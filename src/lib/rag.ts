import { openDB } from 'idb';

/**
 * TurboQuant 4-bit Quantization and MiniFaiss-like Indexing
 */

function seededRandom(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) / 0xffffffff;
  };
}

function l2Norm(v: Float32Array) {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

function normalize(v: Float32Array) {
  const norm = l2Norm(v);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / (norm || 1);
  return { norm, normalized: out };
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function hadamard(v: Float32Array, rng: () => number): Float32Array {
  const d = v.length;
  let n = 1;
  while (n < d) n <<= 1;
  const x = new Float32Array(n);
  for (let i = 0; i < d; i++) x[i] = v[i];
  for (let i = 0; i < n; i++) x[i] *= rng() > 0.5 ? 1 : -1;
  for (let h = 1; h < n; h <<= 1)
    for (let i = 0; i < n; i += h << 1)
      for (let j = i; j < i + h; j++) {
        const a = x[j],
          b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
  const scale = 1 / Math.sqrt(n);
  const out = new Float32Array(d);
  for (let i = 0; i < d; i++) out[i] = x[i] * scale;
  return out;
}

const BOOKS: Record<number, { b: number[]; c: number[] }> = {
  4: {
    b: [
      -1.51, -1.152, -0.896, -0.661, -0.437, -0.218, 0, 0.218, 0.437, 0.661,
      0.896, 1.152, 1.51, 1.894, 2.4,
    ],
    c: [
      -1.894, -1.31, -1.024, -0.778, -0.549, -0.328, -0.11, 0.11, 0.328, 0.549,
      0.778, 1.024, 1.31, 1.632, 2.095, 2.8,
    ],
  },
};

export interface QVec {
  idx: Uint8Array;
  norm: number;
  dim: number;
}

export interface RAGIndexState {
  centroids: Float32Array[];
  lists: [number, string[]][];
  store: [string, { qv: QVec; text: string; centroid: number }][];
}

export interface Result {
  id: string;
  text: string;
  score: number;
  centroid: number;
}

export class TurboQuant {
  quantize(v: Float32Array): QVec {
    const { norm, normalized } = normalize(v);
    const rng = seededRandom(42);
    const rot = hadamard(normalized, rng);
    const idx = new Uint8Array(v.length);
    const { b } = BOOKS[4];
    for (let i = 0; i < v.length; i++) {
      let k = 0;
      while (k < b.length && rot[i] > b[k]) k++;
      idx[i] = k;
    }
    return { idx, norm, dim: v.length };
  }

  ip(query: Float32Array, qv: QVec): number {
    const { normalized: qn } = normalize(query);
    const rot = hadamard(qn, seededRandom(42));
    const { c } = BOOKS[4];
    let dotVal = 0;
    for (let i = 0; i < qv.dim; i++) dotVal += rot[i] * c[qv.idx[i]];
    return dotVal * qv.norm;
  }
}

export class IVF {
  centroids: Float32Array[] = [];
  lists: Map<number, string[]> = new Map();

  constructor(private k: number) {}

  train(vectors: Float32Array[]) {
    if (vectors.length === 0) return;
    this.centroids = vectors.slice(0, Math.min(this.k, vectors.length));
    for (let iter = 0; iter < 5; iter++) {
      const groups: Float32Array[][] = Array.from({ length: this.centroids.length }, () => []);
      for (const v of vectors) {
        const c = this.closestCentroid(v);
        groups[c].push(v);
      }
      this.centroids = groups.map((group, i) => this.mean(group, i));
    }
  }

  assign(id: string, vector: Float32Array): number {
    const c = this.closestCentroid(vector);
    if (!this.lists.has(c)) this.lists.set(c, []);
    this.lists.get(c)!.push(id);
    return c;
  }

  search(query: Float32Array, probes = 2): number[] {
    return this.closestCentroids(query, probes);
  }

  private closestCentroid(v: Float32Array): number {
    let best = 0, bestScore = -Infinity;
    for (let i = 0; i < this.centroids.length; i++) {
      const score = dot(v, this.centroids[i]);
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  }

  private closestCentroids(v: Float32Array, k: number): number[] {
    return this.centroids
      .map((c, i) => ({ i, score: dot(v, c) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((x) => x.i);
  }

  private mean(group: Float32Array[], centroidIdx: number): Float32Array {
    if (group.length === 0) return this.centroids[centroidIdx];
    const dim = group[0].length;
    const out = new Float32Array(dim);
    for (const v of group) for (let i = 0; i < dim; i++) out[i] += v[i];
    for (let i = 0; i < dim; i++) out[i] /= group.length;
    return out;
  }

  clear() {
    this.centroids = [];
    this.lists.clear();
  }
}

export class RAGStore {
  private ivf: IVF;
  private quant = new TurboQuant();
  private store = new Map<string, { qv: QVec; text: string; centroid: number }>();
  private dbName = 'TurboRAG_DB';
  private storeName = 'index_state';

  constructor(k = 8) {
    this.ivf = new IVF(k);
  }

  train(vectors: Float32Array[]) {
    this.ivf.train(vectors);
  }

  add(id: string, text: string, emb: Float32Array) {
    const centroid = this.ivf.assign(id, emb);
    const qv = this.quant.quantize(emb);
    this.store.set(id, { qv, text, centroid });
  }

  search(query: Float32Array, k = 3, probes = 2): Result[] {
    const centroidIndices = this.ivf.search(query, probes);
    const candidates: string[] = [];
    for (const cIdx of centroidIndices) {
      candidates.push(...(this.ivf.lists.get(cIdx) || []));
    }
    return candidates
      .map((id) => {
        const item = this.store.get(id)!;
        return {
          id,
          text: item.text,
          score: this.quant.ip(query, item.qv),
          centroid: item.centroid
        };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  stats() {
    let dims = 0;
    for (const { qv } of this.store.values()) dims += qv.dim;
    const f32 = dims * 4;
    const q4 = Math.ceil((dims * 4) / 8) + this.store.size * 8;
    return {
      docs: this.store.size,
      ratio: (f32 / q4).toFixed(1),
      saved: f32 - q4,
      clusters: this.ivf.centroids.length
    };
  }

  clear() {
    this.store.clear();
    this.ivf.clear();
  }

  exportState(): RAGIndexState {
    return {
      centroids: this.ivf.centroids,
      lists: Array.from(this.ivf.lists.entries()),
      store: Array.from(this.store.entries()),
    };
  }

  importState(state: RAGIndexState) {
    this.clear();
    this.ivf.centroids = state.centroids;
    this.ivf.lists = new Map(state.lists);
    this.store = new Map(state.store);
  }

  async saveToIndexedDB() {
    const db = await openDB(this.dbName, 1, {
      upgrade(db) {
        if (!db.objectStoreNames.contains('index_state')) {
          db.createObjectStore('index_state');
        }
      },
    });
    const state = this.exportState();
    await db.put(this.storeName, state, 'current_index');
    db.close();
  }

  async loadFromIndexedDB(): Promise<boolean> {
    try {
      const db = await openDB(this.dbName, 1, {
        upgrade(db) {
          if (!db.objectStoreNames.contains('index_state')) {
            db.createObjectStore('index_state');
          }
        },
      });
      const state = await db.get(this.storeName, 'current_index') as RAGIndexState | undefined;
      db.close();
      if (state && state.centroids && state.centroids.length > 0) {
        this.importState(state);
        return true;
      }
      return false;
    } catch (err) {
      console.error("Failed to load from IndexedDB:", err);
      return false;
    }
  }

  async deleteDatabase() {
    const db = await openDB(this.dbName, 1);
    await db.clear(this.storeName);
    db.close();
    this.clear();
  }
}
