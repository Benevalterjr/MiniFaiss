import { openDB, IDBPDatabase } from 'idb';
import { QVec, RAGIndexState, Result } from '../types';

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

// Optimized 4-bit centroids for N(0,1)
const BOOKS: Record<number, { b: number[]; c: number[] }> = {
  4: {
    b: [
      -1.894, -1.51, -1.152, -0.896, -0.661, -0.437, -0.218, 0, 0.218, 0.437,
      0.661, 0.896, 1.152, 1.51, 1.894, 2.4,
    ],
    c: [
      -2.095, -1.632, -1.31, -1.024, -0.778, -0.549, -0.328, -0.11, 0.11, 0.328,
      0.549, 0.778, 1.024, 1.31, 1.632, 2.095,
    ],
  },
};

export class TurboQuant {
  // TurboQuant-prod: 2-stage quantization
  quantize(v: Float32Array): QVec {
    const { norm, normalized } = normalize(v);
    const rng = seededRandom(42);
    const rot = hadamard(normalized, rng);
    
    // Stage 1: 4-bit MSE quantization
    const idx = new Uint8Array(v.length);
    const recon = new Float32Array(v.length);
    const { b, c } = BOOKS[4];
    
    // Variance scaling for standard N(0,1) codebook
    const stdFactor = Math.sqrt(v.length);
    for (let i = 0; i < v.length; i++) {
      const scaledVal = rot[i] * stdFactor;
      let k = 0;
      while (k < b.length && scaledVal > b[k]) k++;
      idx[i] = k;
      recon[i] = c[k] / stdFactor; // Scale back
    }
    
    // Stage 2: 1-bit residual quantization for unbiased IP
    const residual = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) {
      residual[i] = rot[i] - recon[i];
    }
    
    const resNorm = l2Norm(residual);
    // Pack 1-bit signs (1 = positive, 0 = negative)
    const resIdxSize = Math.ceil(v.length / 8);
    const resIdx = new Uint8Array(resIdxSize);
    for (let i = 0; i < v.length; i++) {
      if (residual[i] >= 0) {
        resIdx[i >> 3] |= (1 << (i & 7));
      }
    }
    
    return { idx, resIdx, norm, resNorm, dim: v.length };
  }

  ip(query: Float32Array, qv: QVec): number {
    const { normalized: qn } = normalize(query);
    const rng = seededRandom(42);
    const qRot = hadamard(qn, rng);
    const { c } = BOOKS[4];
    
    // MSE Component
    let dotMse = 0;
    const stdFactor = Math.sqrt(qv.dim);
    for (let i = 0; i < qv.dim; i++) {
      dotMse += qRot[i] * (c[qv.idx[i]] / stdFactor);
    }
    
    // Residual Component (Unbiased Correction)
    let dotRes = 0;
    const resNorm = qv.resNorm ?? 0;
    const resIdx = qv.resIdx;
    const resScale = resNorm / Math.sqrt(qv.dim);

    if (resIdx && resIdx.length > 0) {
      for (let i = 0; i < qv.dim; i++) {
        const bit = (resIdx[i >> 3] >> (i & 7)) & 1;
        const sign = bit ? 1 : -1;
        dotRes += qRot[i] * sign;
      }
    }
    
    let score = (dotMse * qv.norm) + (dotRes * resScale * qv.norm);
    if (Number.isNaN(score)) score = 0;
    
    return Math.max(0, Math.min(1, score)); // Clamp for cosine
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
    if (this.centroids.length === 0) return 0;
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
  private store = new Map<string, { qv: QVec; text: string; centroid: number; metadata?: Record<string, unknown> }>();
  private dbName = 'TurboRAG_DB';
  private storeName = 'index_state';

  constructor(k = 8) {
    this.ivf = new IVF(k);
  }

  train(vectors: Float32Array[]) {
    this.ivf.train(vectors);
  }

  add(id: string, text: string, emb: Float32Array, metadata?: Record<string, unknown>) {
    const centroid = this.ivf.assign(id, emb);
    const qv = this.quant.quantize(emb);
    this.store.set(id, { qv, text, centroid, metadata });
  }

  search(query: Float32Array, k = 3, probes = 2): Result[] {
    // Dynamic probing: use more probes for small collections to ensure accuracy
    const actualProbes = this.store.size < 50 ? this.ivf.centroids.length : probes;
    const centroidIndices = this.ivf.search(query, actualProbes);
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
          centroid: item.centroid,
          metadata: item.metadata
        };
      })
      .sort((a, b) => b.score - a.score)
      .filter((v, i, a) => a.findIndex(t => t.id === v.id) === i) // Unique
      .slice(0, k);
  }

  stats() {
    let dims = 0;
    for (const { qv } of this.store.values()) dims += qv.dim;
    // Calculation: 4 bits/dim + 1 bit/dim + 4 bytes norm + 4 bytes resNorm
    const f32 = dims * 4;
    const qBytes = (dims * 5) / 8 + (this.store.size * 8); 
    return {
      docs: this.store.size,
      ratio: (f32 / qBytes).toFixed(1),
      saved: Math.max(0, Math.floor(f32 - qBytes)),
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
    const db: IDBPDatabase = await openDB(this.dbName, 1, {
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
      const db: IDBPDatabase = await openDB(this.dbName, 1, {
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
