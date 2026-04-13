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

const STOPWORDS_PT = new Set(["o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para", "com", "que", "se", "como", "esta", "esse", "isso", "aquele", "aquilo", "este", "tem", "foi", "era", "ser", "eram", "estão", "estava", "estavam", "mais", "muito", "seu", "sua", "seus", "suas", "pode", "podem", "quem", "qual", "quais", "onde", "quando", "como", "mas", "nem", "ou", "então", "desde", "até", "após", "entre", "sobre", "sob", "sem", "cada"]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "") // Remove accents
    .replace(/[^\w\s]/g, "") // Remove punctuation
    .split(/\s+/)
    .filter(t => t.length > 2 && !STOPWORDS_PT.has(t)); // Filter short words and stopwords
}

function lexicalScore(queryTokens: string[], docText: string): number {
  if (queryTokens.length === 0) return 0;
  const docTokens = new Set(tokenize(docText));
  let matches = 0;
  for (const token of queryTokens) {
    if (docTokens.has(token)) matches++;
  }
  return matches / queryTokens.length;
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
    
    // Adaptive Variance scaling
    let variance = 0;
    for (const val of rot) variance += val * val;
    const std = Math.sqrt(variance / v.length) || (1 / Math.sqrt(v.length));
    const scale = 1 / std;

    for (let i = 0; i < v.length; i++) {
      const scaledVal = rot[i] * scale;
      // Binary search for index
      let low = 0, high = b.length;
      while (low < high) {
        let mid = (low + high) >>> 1;
        if (scaledVal > b[mid]) low = mid + 1;
        else high = mid;
      }
      const k = Math.max(0, Math.min(b.length - 1, low));
      idx[i] = k;
      recon[i] = c[k] / scale; 
    }
    
    // Stage 2: 2-bit residual quantization for higher precision
    const residual = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) {
      residual[i] = rot[i] - recon[i];
    }
    
    const resNorm = l2Norm(residual);
    // 2-bit quantization levels: -1.2, -0.4, 0.4, 1.2
    const resIdxSize = Math.ceil(v.length / 4);
    const resIdx = new Uint8Array(resIdxSize);
    const resScale = resNorm / Math.sqrt(v.length);

    for (let i = 0; i < v.length; i++) {
      const val = residual[i] / (resScale || 1e-9);
      let bits = 0;
      if (val < -0.8) bits = 0;      // ~ -1.2
      else if (val < 0) bits = 1;    // ~ -0.4
      else if (val < 0.8) bits = 2;   // ~ 0.4
      else bits = 3;                  // ~ 1.2
      
      const byteIdx = i >> 2;
      const bitShift = (i & 3) << 1;
      resIdx[byteIdx] |= (bits << bitShift);
    }
    
    return { idx, resIdx, norm, resNorm, dim: v.length, scale };
  }

  ip(query: Float32Array, qv: QVec): number {
    const { normalized: qn } = normalize(query);
    const rng = seededRandom(42);
    const qRot = hadamard(qn, rng);
    const { c } = BOOKS[4];
    
    // MSE Component
    let dotMse = 0;
    const scale = qv.scale;
    for (let i = 0; i < qv.dim; i++) {
      dotMse += qRot[i] * (c[qv.idx[i]] / scale);
    }
    
    // Residual Component (Unbiased Correction)
    let dotRes = 0;
    const resNorm = qv.resNorm ?? 0;
    const resIdx = qv.resIdx;
    const resScale = resNorm / Math.sqrt(qv.dim);

    if (resIdx && resIdx.length > 0) {
      const resLevels = [-1.2, -0.4, 0.4, 1.2];
      for (let i = 0; i < qv.dim; i++) {
        const byteIdx = i >> 2;
        const bitShift = (i & 3) << 1;
        const bits = (resIdx[byteIdx] >> bitShift) & 3;
        dotRes += qRot[i] * resLevels[bits];
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
    
    // 1. Ensure vectors are normalized for Cosine Clustering
    const normalizedVectors = vectors.map(v => normalize(v).normalized);
    
    // 2. K-means++ initialization
    this.centroids = this.initializeKMeansPlusPlus(normalizedVectors);
    
    // 3. Training loop with convergence check
    let prevShift = Infinity;
    for (let iter = 0; iter < 20; iter++) {
      const groups: Float32Array[][] = Array.from({ length: this.centroids.length }, () => []);
      for (const v of normalizedVectors) {
        const c = this.closestCentroid(v);
        groups[c].push(v);
      }
      
      const nextCentroids = groups.map((group, i) => this.mean(group, i));
      
      // Calculate total centroid shift for convergence
      let totalShift = 0;
      for (let i = 0; i < this.centroids.length; i++) {
        totalShift += this.distSq(this.centroids[i], nextCentroids[i]);
      }
      
      this.centroids = nextCentroids;
      
      // Stop if shift is minimal or stabilized
      if (totalShift < 1e-6 || Math.abs(prevShift - totalShift) < 1e-9) break;
      prevShift = totalShift;
    }
  }

  private initializeKMeansPlusPlus(vectors: Float32Array[]): Float32Array[] {
    const centroids: Float32Array[] = [];
    const n = vectors.length;
    
    // Pick first centroid randomly
    const rng = seededRandom(Date.now());
    centroids.push(vectors[Math.floor(rng() * n)]);
    
    while (centroids.length < Math.min(this.k, n)) {
      const distances = new Float64Array(n);
      let sumSqDist = 0;
      
      for (let i = 0; i < n; i++) {
        let minDistSq = Infinity;
        for (const c of centroids) {
          const d2 = this.distSq(vectors[i], c);
          if (d2 < minDistSq) minDistSq = d2;
        }
        distances[i] = minDistSq;
        sumSqDist += minDistSq;
      }
      
      // Select next centroid with probability proportional to D(x)^2
      let r = rng() * sumSqDist;
      for (let i = 0; i < n; i++) {
        r -= distances[i];
        if (r <= 0) {
          centroids.push(vectors[i]);
          break;
        }
      }
    }
    
    return centroids;
  }

  private distSq(a: Float32Array, b: Float32Array): number {
    let d2 = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      d2 += diff * diff;
    }
    return d2;
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
  private store = new Map<string, { qv: QVec; text: string; full: Float32Array; centroid: number; metadata?: Record<string, unknown> }>();
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
    this.store.set(id, { qv, text, full: emb, centroid, metadata });
  }

  search(query: Float32Array, queryStr: string, k = 3, probes = 2): Result[] {
    // Stage 1: Fast filtering with IVF + TurboQuant
    const actualProbes = this.store.size < 50 ? this.ivf.centroids.length : probes;
    const centroidIndices = this.ivf.search(query, actualProbes);
    const candidates: string[] = [];
    for (const cIdx of centroidIndices) {
      candidates.push(...(this.ivf.lists.get(cIdx) || []));
    }
    
    // Deduplicate candidates
    const uniqueCandidates = Array.from(new Set(candidates));
    
    // Preliminary scoring with TurboQuant for Top-N candidates
    const preliminaryTopN = uniqueCandidates
      .map((id) => {
        const item = this.store.get(id)!;
        return {
          id,
          qScore: this.quant.ip(query, item.qv)
        };
      })
      .sort((a, b) => b.qScore - a.qScore)
      .slice(0, k * 3); // Take top N for re-ranking

    const queryTokens = tokenize(queryStr);

    // Stage 2: Precision re-ranking using original float32 embeddings + Lexical Boost
    return preliminaryTopN
      .map(({ id }) => {
        const item = this.store.get(id)!;
        const { normalized: qn } = normalize(query);
        const { normalized: en } = normalize(item.full);
        
        const semanticScore = dot(qn, en); // Exact Cosine Similarity
        const lScore = lexicalScore(queryTokens, item.text);
        
        // Hybrid Fusion: 0.7 Semantic + 0.3 Lexical
        const hybridScore = (0.7 * semanticScore) + (0.3 * lScore);

        return {
          id,
          text: item.text,
          score: hybridScore,
          centroid: item.centroid,
          metadata: item.metadata
        };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  stats() {
    let dims = 0;
    for (const { qv } of this.store.values()) dims += qv.dim;
    
    // Calculation: 
    // TurboQuant size (6 bits/dim) + float32 size (32 bits/dim)
    const f32Original = dims * 4;
    const qBytes = (dims * 6) / 8 + (this.store.size * 12); 
    const totalBytes = qBytes + f32Original;
    
    return {
      docs: this.store.size,
      ratio: totalBytes > 0 ? (f32Original / totalBytes).toFixed(1) : "0.0",
      saved: Math.max(0, Math.floor(f32Original - qBytes)),
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
