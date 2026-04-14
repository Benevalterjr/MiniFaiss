import { openDB, IDBPDatabase } from 'idb';
import { QVec, BM25State, RAGIndexState, Result } from '../types';

/**
 * TurboQuant 4-bit Quantization, BM25 Full-Text Search, and MiniFaiss-like Indexing
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

// ============================================================
// Portuguese NLP: Stopwords, Stemmer, Tokenizer
// ============================================================

const STOPWORDS_PT = new Set([
  "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para", "com", "que", "se", "como", "esta", "esse", "isso", "aquele", "aquilo", "este", "tem", "foi", "era", "ser", "eram", "estão", "estava", "estavam", "mais", "muito", "seu", "sua", "seus", "suas", "pode", "podem", "quem", "qual", "quais", "onde", "quando", "como", "mas", "nem", "ou", "então", "desde", "até", "após", "entre", "sobre", "sob", "sem", "cada", "todo", "toda", "todos", "todas", "outro", "outra", "outros", "outras", "esteja", "estejam", "estivesse", "estivessem", "fosse", "fossem", "minha", "minhas", "meu", "meus", "nosso", "nossa", "nossos", "nossas"
]);

/** Light Portuguese stemmer — conservative suffix removal (~30 rules) */
function stemPT(word: string): string {
  if (word.length <= 3) return word;
  // Superlatives / augmentatives
  for (const suf of ['issimo', 'issima', 'mente']) {
    if (word.length > suf.length + 3 && word.endsWith(suf)) return word.slice(0, -suf.length);
  }
  // Verbal suffixes (longest first)
  for (const suf of ['ando', 'endo', 'indo', 'aram', 'eram', 'iram', 'avam', 'ando', 'asse', 'esse', 'isse', 'aria', 'eria', 'iria']) {
    if (word.length > suf.length + 3 && word.endsWith(suf)) return word.slice(0, -suf.length);
  }
  // Nominal suffixes  
  for (const suf of ['ções', 'ção', 'idades', 'idade', 'istas', 'ista', 'ores', 'ador', 'edor', 'eiros', 'eiro', 'eira', 'osas', 'osos', 'oso', 'osa']) {
    if (word.length > suf.length + 3 && word.endsWith(suf)) return word.slice(0, -suf.length);
  }
  // Plural (conservative)
  if (word.length > 4 && word.endsWith('ões')) return word.slice(0, -3);
  if (word.length > 4 && word.endsWith('ães')) return word.slice(0, -3);
  if (word.length > 3 && word.endsWith('es') && !word.endsWith('ões')) return word.slice(0, -2);
  if (word.length > 3 && word.endsWith('s') && !word.endsWith('ss')) return word.slice(0, -1);
  return word;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter(t => t.length > 2 && !STOPWORDS_PT.has(t));
}

function tokenizeAndStem(text: string): string[] {
  return tokenize(text).map(stemPT);
}

// ============================================================
// BM25 Inverted Index (Okapi BM25)
// ============================================================

export class BM25Index {
  // term → Map<docId, termFrequency>
  private postings = new Map<string, Map<string, number>>();
  private docLengths = new Map<string, number>();
  private totalDocs = 0;
  private avgDocLength = 0;
  private k1 = 1.2;
  private b = 0.75;

  addDocument(id: string, text: string): void {
    const tokens = tokenizeAndStem(text);
    this.docLengths.set(id, tokens.length);
    this.totalDocs++;

    // Count term frequencies
    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);

    // Update postings
    for (const [term, count] of tf) {
      if (!this.postings.has(term)) this.postings.set(term, new Map());
      this.postings.get(term)!.set(id, count);
    }

    // Update average
    this.recomputeAvg();
  }

  removeDocument(id: string): void {
    if (!this.docLengths.has(id)) return;
    this.docLengths.delete(id);
    this.totalDocs--;
    // Remove from all posting lists
    for (const [, docs] of this.postings) docs.delete(id);
    this.recomputeAvg();
  }

  private recomputeAvg(): void {
    if (this.totalDocs === 0) { this.avgDocLength = 0; return; }
    let total = 0;
    for (const len of this.docLengths.values()) total += len;
    this.avgDocLength = total / this.totalDocs;
  }

  private idf(term: string): number {
    const df = this.postings.get(term)?.size ?? 0;
    return Math.log((this.totalDocs - df + 0.5) / (df + 0.5) + 1);
  }

  /** BM25 score for a single document given stemmed query tokens */
  score(queryTokens: string[], docId: string): number {
    const dl = this.docLengths.get(docId) ?? 0;
    if (dl === 0) return 0;
    let score = 0;
    for (const qt of queryTokens) {
      const posting = this.postings.get(qt);
      if (!posting) continue;
      const tf = posting.get(docId) ?? 0;
      if (tf === 0) continue;
      const idf = this.idf(qt);
      const num = tf * (this.k1 + 1);
      const den = tf + this.k1 * (1 - this.b + this.b * dl / (this.avgDocLength || 1));
      score += idf * (num / den);
    }
    return score;
  }

  /** Standalone BM25 search: returns top-k doc IDs sorted by score */
  search(query: string, k: number): { id: string; score: number }[] {
    const qTokens = tokenizeAndStem(query);
    if (qTokens.length === 0) return [];

    // Only score docs that contain at least one query term
    const candidateIds = new Set<string>();
    for (const qt of qTokens) {
      const posting = this.postings.get(qt);
      if (posting) for (const id of posting.keys()) candidateIds.add(id);
    }

    const results: { id: string; score: number }[] = [];
    for (const id of candidateIds) {
      results.push({ id, score: this.score(qTokens, id) });
    }
    return results.sort((a, b) => b.score - a.score).slice(0, k);
  }

  clear(): void {
    this.postings.clear();
    this.docLengths.clear();
    this.totalDocs = 0;
    this.avgDocLength = 0;
  }

  exportState(): BM25State {
    return {
      postings: Array.from(this.postings.entries()).map(([term, docs]) => [term, Array.from(docs.entries())]),
      docLengths: Array.from(this.docLengths.entries()),
      avgDocLength: this.avgDocLength,
    };
  }

  importState(state: BM25State): void {
    this.clear();
    for (const [term, docs] of state.postings) {
      this.postings.set(term, new Map(docs));
    }
    this.docLengths = new Map(state.docLengths);
    this.totalDocs = this.docLengths.size;
    this.avgDocLength = state.avgDocLength;
  }
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

function hadamard(v: Float32Array, seed: number): Float32Array {
  const d = v.length;
  const rng = seededRandom(seed);
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

// Exact Lloyd-Max centroids for N(0,1) at 4-bit (16 levels)
// Source: Paez & Glisson 1972, Max 1960 — verified standard values
const LLOYD_MAX_16 = [
  -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
   0.1284,  0.3881,  0.6568,  0.9423,  1.2562,  1.6180,  2.0690,  2.7326,
];

// Paper Section 3.1: centroids scaled by 1/sqrt(d) for unit sphere coordinates
// After random rotation, coordinates ~ N(0, 1/d), so centroids = LLOYD_MAX * 1/sqrt(d)
function computeCodebook(dim: number): { centroids: number[]; boundaries: number[] } {
  const s = 1.0 / Math.sqrt(dim);
  const centroids = LLOYD_MAX_16.map(c => c * s);
  // Boundaries = midpoints between consecutive centroids
  const boundaries: number[] = [];
  for (let i = 0; i < centroids.length - 1; i++) {
    boundaries.push((centroids[i] + centroids[i + 1]) / 2);
  }
  return { centroids, boundaries };
}

export class TurboQuant {
  private codebooks = new Map<number, { centroids: number[]; boundaries: number[] }>();
  
  private getCodebook(dim: number) {
    if (!this.codebooks.has(dim)) {
      this.codebooks.set(dim, computeCodebook(dim));
    }
    return this.codebooks.get(dim)!;
  }

  // Stage 1: MSE-optimal quantization via random rotation + scalar Lloyd-Max
  // Stage 2: 1-bit QJL on residual for unbiased inner product estimation
  quantize(v: Float32Array): QVec {
    const dim = v.length;
    const { norm, normalized } = normalize(v);
    
    // Random rotation (Hadamard with random signs as approximation)
    const rot = hadamard(normalized, 42);
    
    // Stage 1: 4-bit scalar quantization per coordinate
    const { centroids, boundaries } = this.getCodebook(dim);
    const idx = new Uint8Array(dim);
    const recon = new Float32Array(dim);
    
    for (let i = 0; i < dim; i++) {
      // Find nearest centroid via binary search on boundaries
      let k = 0;
      for (let j = 0; j < boundaries.length; j++) {
        if (rot[i] > boundaries[j]) k = j + 1;
        else break;
      }
      idx[i] = k;
      recon[i] = centroids[k];
    }
    
    // Compute residual r = rot - recon
    const residual = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      residual[i] = rot[i] - recon[i];
    }
    const resNorm = l2Norm(residual);
    
    // Stage 2: 1-bit QJL on residual
    // QJL: store sign(S · r) where S is a random projection
    // For efficiency, we use a seeded random projection
    const resIdxSize = Math.ceil(dim / 8); // 1 bit per dimension
    const resIdx = new Uint8Array(resIdxSize);
    const rng = seededRandom(7919); // Different seed for QJL projection
    
    for (let i = 0; i < dim; i++) {
      // Compute projection: dot product of random vector with residual
      // Simplified: use random sign flipping (equivalent to diagonal random matrix)
      const projected = residual[i] * (rng() > 0.5 ? 1 : -1);
      const bit = projected >= 0 ? 1 : 0;
      const byteIdx = i >> 3;
      const bitShift = i & 7;
      resIdx[byteIdx] |= (bit << bitShift);
    }
    
    // resScale stores gamma = ||r|| for QJL dequantization
    const resScale = resNorm;
    
    return { idx, resIdx, norm, resNorm, dim, scale: 0, resScale };
  }

  // Compute rotated query ONCE, then reuse for all candidates
  rotateQuery(query: Float32Array): Float32Array {
    const { normalized: qn } = normalize(query);
    return hadamard(qn, 42);
  }

  // Inner product estimation: MSE reconstruction + QJL correction
  // Paper Eq: <y, x_tilde> = <y, x_mse> + gamma * <y, x_qjl>
  ip(qRot: Float32Array, qv: QVec): number {
    const dim = qv.dim;
    const { centroids } = this.getCodebook(dim);
    
    // Stage 1: <qRot, recon> = MSE component
    let dotMse = 0;
    for (let i = 0; i < dim; i++) {
      dotMse += qRot[i] * centroids[qv.idx[i]];
    }
    
    // Stage 2: QJL correction on residual  
    // DeQuant_qjl: sqrt(pi/2) * sign_vector / d
    // <qRot, x_qjl> = gamma * sqrt(pi/2) / d * sum(qRot[i] * sign[i] * randomSign[i])
    let dotQjl = 0;
    const resIdx = qv.resIdx;
    const gamma = qv.resScale || 0;
    
    if (resIdx && resIdx.length > 0 && gamma > 0) {
      const rng = seededRandom(7919); // Same seed as quantization
      const qjlScale = gamma * Math.sqrt(Math.PI / 2) / dim;
      
      for (let i = 0; i < dim; i++) {
        const byteIdx = i >> 3;
        const bitShift = i & 7;
        const bit = (resIdx[byteIdx] >> bitShift) & 1;
        const sign = bit === 1 ? 1 : -1;
        const randomSign = rng() > 0.5 ? 1 : -1;
        dotQjl += qRot[i] * sign * randomSign * qjlScale;
      }
    }
    
    const score = dotMse + dotQjl;
    return Number.isNaN(score) ? 0 : score;
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
  private bm25 = new BM25Index();
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
    this.bm25.addDocument(id, text);
  }

  /**
   * Hybrid search: dual retrieval (Semantic + BM25) fused via Reciprocal Rank Fusion.
   * 
   * Pipeline:
   *  1. Semantic: IVF probe → TurboQuant IP → rank by cosine-like score
   *  2. Lexical: BM25 inverted index → rank by BM25 score
   *  3. Fusion: RRF(k=60) merges both rankings
   */
  search(query: Float32Array, queryStr: string, options: { k?: number; probes?: number; hybridMode?: 'rrf' | 'weighted' } = {}): Result[] {
    const { k = 3, probes = 2, hybridMode = 'rrf' } = options;
    const RRF_K = 60; // Standard RRF constant
    const CANDIDATE_POOL = Math.max(k * 5, 50); // Retrieve more candidates for better fusion

    // ---- Stage 1: Semantic Retrieval (TurboQuant + IVF) ----
    const actualProbes = this.store.size < 50 ? this.ivf.centroids.length : probes;
    const centroidIndices = this.ivf.search(query, actualProbes);
    const semanticCandidates: string[] = [];
    for (const cIdx of centroidIndices) {
      semanticCandidates.push(...(this.ivf.lists.get(cIdx) || []));
    }
    const uniqueSemantic = Array.from(new Set(semanticCandidates));

    // Score semantic candidates
    const qRot = this.quant.rotateQuery(query);
    const semanticScores = new Map<string, number>();
    const semanticScored = uniqueSemantic
      .map(id => {
        const score = this.quant.ip(qRot, this.store.get(id)!.qv);
        semanticScores.set(id, score);
        return { id, score };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, CANDIDATE_POOL);

    // ---- Stage 2: Lexical Retrieval (BM25) ----
    const bm25Results = this.bm25.search(queryStr, CANDIDATE_POOL);
    const bm25Scores = new Map(bm25Results.map(r => [r.id, r.score]));

    // ---- Stage 3: Fusion ----
    const finalScores = new Map<string, number>();
    const candidateIds = new Set([...semanticScored.map(r => r.id), ...bm25Results.map(r => r.id)]);

    if (hybridMode === 'rrf') {
      // Reciprocal Rank Fusion
      for (let rank = 0; rank < semanticScored.length; rank++) {
        const { id } = semanticScored[rank];
        finalScores.set(id, (finalScores.get(id) || 0) + 1 / (RRF_K + rank + 1));
      }
      for (let rank = 0; rank < bm25Results.length; rank++) {
        const { id } = bm25Results[rank];
        finalScores.set(id, (finalScores.get(id) || 0) + 1 / (RRF_K + rank + 1));
      }
    } else {
      // Weighted Fusion (fallback/baseline)
      // Normalize BM25 scores roughly using sigmoid to match TurboQuant [-1, 1] scale
      for (const id of candidateIds) {
        const sem = semanticScores.get(id) || 0;
        const bm25Raw = bm25Scores.get(id) || 0;
        const bm25Norm = 1 / (1 + Math.exp(-bm25Raw * 0.5)); // smooth sigmoid
        const weighted = (0.75 * sem) + (0.25 * bm25Norm);
        finalScores.set(id, weighted);
      }
    }

    // Build final results sorted by fusion score
    return Array.from(finalScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, k)
      .map(([id, score]) => {
        const item = this.store.get(id)!;
        return {
          id,
          text: item.text,
          score,
          semanticScore: semanticScores.get(id) || 0,
          bm25Score: bm25Scores.get(id) || 0,
          centroid: item.centroid,
          metadata: item.metadata
        };
      });
  }

  stats() {
    let totalDims = 0;
    const totalDocs = this.store.size;
    
    for (const { qv } of this.store.values()) {
      totalDims += qv.dim;
    }

    // Real memory calculation: 5 bits/dim + per-doc overhead
    // (4-bit MSE index + 1-bit QJL sign)
    const bitsPerVector = 5;
    const qBytes = Math.ceil((totalDims * bitsPerVector) / 8) + (totalDocs * 24); 
    const originalBytes = totalDims * 4;
    
    return {
      docs: totalDocs,
      dimensions: totalDims,
      memoryMB: (qBytes / (1024 * 1024)).toFixed(2),
      ratio: originalBytes > 0 ? (originalBytes / qBytes).toFixed(1) + "x" : "0.0x",
      saved: Math.max(0, Math.floor(originalBytes - qBytes)),
      clusters: this.ivf.centroids.length
    };
  }

  clear() {
    this.store.clear();
    this.ivf.clear();
    this.bm25.clear();
  }

  exportState(): RAGIndexState {
    return {
      centroids: this.ivf.centroids,
      lists: Array.from(this.ivf.lists.entries()),
      store: Array.from(this.store.entries()),
      bm25: this.bm25.exportState(),
    };
  }

  importState(state: RAGIndexState) {
    this.clear();
    this.ivf.centroids = state.centroids;
    this.ivf.lists = new Map(state.lists);
    this.store = new Map(state.store);
    if (state.bm25) {
      this.bm25.importState(state.bm25);
    } else {
      // Backward compat: rebuild BM25 from stored texts
      for (const [id, { text }] of this.store) {
        this.bm25.addDocument(id, text);
      }
    }
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
