import { expose } from "comlink";
import { pipeline, env, FeatureExtractionPipeline, Tensor } from "@huggingface/transformers";
import { RAGStore } from "./lib/rag";
import { EngineStats } from "./types";

// Configure transformers for browser
env.allowLocalModels = false;
env.useBrowserCache = true;

// Fix for WASM Out of Memory errors
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;
}

export class TurboRAGWorker {
  private extractor: FeatureExtractionPipeline | null = null;
  private store: RAGStore = new RAGStore();
  private modelName = "Xenova/multilingual-e5-small";

  async init(progressCallback?: (p: { status: string; progress: number }) => void) {
    if (!this.extractor) {
      this.extractor = (await pipeline("feature-extraction", this.modelName, {
        progress_callback: (p: any) => {
          if (p.status === 'progress' && progressCallback) {
            progressCallback(p);
          }
        }
      })) as FeatureExtractionPipeline;
    }
    
    // Try to load from IndexedDB
    const hasCache = await this.store.loadFromIndexedDB();
    return { hasCache };
  }

  async embed(text: string): Promise<Float32Array> {
    if (!this.extractor) throw new Error("Extractor not initialized");
    
    // E5 models require 'query: ' or 'passage: ' prefixes
    const out = (await this.extractor(text, { pooling: "mean", normalize: true })) as Tensor;
    const data = out.data;
    return data instanceof Float32Array ? data : new Float32Array(Array.from(data as Iterable<number>));
  }

  async indexDocuments(docs: { id: string, text: string, metadata?: Record<string, unknown> }[], progressCallback: (p: number) => void): Promise<EngineStats> {
    this.store.clear();
    const embeddings: Float32Array[] = [];
    
    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      const emb = await this.embed(`passage: ${doc.text}`);
      embeddings.push(emb);
      progressCallback(Math.round(((i + 1) / docs.length) * 50));
    }

    this.store.train(embeddings);
    progressCallback(75);

    for (let i = 0; i < docs.length; i++) {
      this.store.add(docs[i].id, docs[i].text, embeddings[i], docs[i].metadata);
    }
    
    await this.store.saveToIndexedDB();
    progressCallback(100);
    return this.store.stats();
  }

  async search(query: string, k = 3, probes = 2) {
    const qEmb = await this.embed(`query: ${query}`);
    return this.store.search(qEmb, query, { k, probes });
  }

  async getStats() {
    return this.store.stats();
  }

  async clearCache() {
    await this.store.deleteDatabase();
  }
}

expose(new TurboRAGWorker());
