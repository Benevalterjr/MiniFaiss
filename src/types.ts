export interface QVec {
  idx: Uint8Array;
  resIdx: Uint8Array;
  norm: number;
  resNorm: number;
  dim: number;
  scale: number;
  resScale: number;
}

export interface Document {
  id: string;
  text: string;
  source: string;
  type: 'txt' | 'pdf';
  metadata?: Record<string, unknown>;
}

export interface Result {
  id: string;
  text: string;
  score: number;
  semanticScore?: number;
  bm25Score?: number;
  centroid: number;
  metadata?: Record<string, unknown>;
}

export interface BM25State {
  postings: [string, [string, number][]][];  // term → [docId, tf][]
  docLengths: [string, number][];             // docId → token count
  avgDocLength: number;
}

export interface RAGIndexState {
  centroids: Float32Array[];
  lists: [number, string[]][];
  store: [string, { qv: QVec; text: string; centroid: number; metadata?: Record<string, unknown> }][];
  bm25?: BM25State;
}

export interface EngineStats {
  docs: number;
  dimensions: number;
  memoryMB: string;
  ratio: string;
  saved: number;
  clusters: number;
}
