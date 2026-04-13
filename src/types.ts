export interface QVec {
  idx: Uint8Array;
  resIdx: Uint8Array;
  norm: number;
  resNorm: number;
  dim: number;
  scale: number;
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
  centroid: number;
  metadata?: Record<string, unknown>;
}

export interface RAGIndexState {
  centroids: Float32Array[];
  lists: [number, string[]][];
  store: [string, { qv: QVec; text: string; full: Float32Array; centroid: number; metadata?: Record<string, unknown> }][];
}

export interface EngineStats {
  docs: number;
  ratio: string;
  saved: number;
  clusters: number;
}
