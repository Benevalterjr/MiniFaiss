/**
 * MiniFaiss HF Dataset Benchmark v2
 * - Streams SQuAD pairs from HuggingFace
 * - Caches embeddings to disk for fast re-runs
 * - Measures Recall@10, Spearman Rank Correlation, and latency
 */
import { pipeline, env } from "@huggingface/transformers";
import { TurboQuant, RAGStore } from "../src/lib/rag";
import https from "https";
import http from "http";
import zlib from "zlib";
import readline from "readline";
import fs from "fs";

env.allowLocalModels = false;

const CACHE_DIR = "../.benchmark_cache";
const DOC_CACHE = `${CACHE_DIR}/squad_doc_embeddings.json`;
const QUERY_CACHE = `${CACHE_DIR}/squad_query_embeddings.json`;

// ============================================================
// Helpers
// ============================================================
function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

function spearmanCorrelation(exact: number[], approx: number[]): number {
  const n = exact.length;
  if (n < 2) return 0;

  function rankArray(arr: number[]): number[] {
    const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
    const ranks = new Array(n);
    for (let i = 0; i < n; i++) ranks[sorted[i].i] = i + 1;
    return ranks;
  }

  const rankE = rankArray(exact);
  const rankA = rankArray(approx);

  let d2sum = 0;
  for (let i = 0; i < n; i++) {
    const d = rankE[i] - rankA[i];
    d2sum += d * d;
  }

  return 1 - (6 * d2sum) / (n * (n * n - 1));
}

// ============================================================
// Dataset Streaming (with redirect support)
// ============================================================
function fetchUrl(url: string): Promise<http.IncomingMessage> {
  return new Promise((resolve, reject) => {
    const mod = url.startsWith("https") ? https : http;
    mod.get(url, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        const loc = res.headers.location;
        if (loc) return resolve(fetchUrl(loc));
      }
      if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode}`));
      resolve(res);
    }).on("error", reject);
  });
}

async function fetchDatasetPairs(url: string, limit: number): Promise<{ query: string; doc: string }[]> {
  console.log(`Downloading dataset (streaming up to ${limit} pairs)...`);
  const res = await fetchUrl(url);
  const gunzip = zlib.createGunzip();
  res.pipe(gunzip);

  const rl = readline.createInterface({ input: gunzip, crlfDelay: Infinity });
  const pairs: { query: string; doc: string }[] = [];

  for await (const line of rl) {
    if (pairs.length >= limit) break;
    try {
      if (!line.trim()) continue;
      const parsed = JSON.parse(line);
      if (Array.isArray(parsed) && parsed.length >= 2) {
        pairs.push({ query: parsed[0], doc: parsed[1] });
      }
    } catch { /* skip */ }
  }
  return pairs;
}

// ============================================================
// Embedding Cache (saves ~3min on re-runs)
// ============================================================
function saveEmbeddings(path: string, embeddings: number[][]): void {
  if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
  fs.writeFileSync(path, JSON.stringify(embeddings));
  console.log(`  💾 Saved ${embeddings.length} embeddings to ${path}`);
}

function loadEmbeddings(path: string): Float32Array[] | null {
  if (!fs.existsSync(path)) return null;
  console.log(`  💾 Loading cached embeddings from ${path}...`);
  const raw: number[][] = JSON.parse(fs.readFileSync(path, "utf-8"));
  return raw.map(a => new Float32Array(a));
}

// ============================================================
// Main Benchmark
// ============================================================
async function main() {
  const K = 10;
  const NUM_DOCS = 1000;
  const NUM_QUERIES = 100;
  const DATASET_URL = "https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/squad_pairs.jsonl.gz";

  console.log("🚀 MiniFaiss HuggingFace SQuAD Benchmark v2");
  console.log(`   ${NUM_DOCS} docs | ${NUM_QUERIES} queries | k=${K}\n`);

  // ---- 1. Load dataset ----
  const pairs = await fetchDatasetPairs(DATASET_URL, NUM_DOCS);
  console.log(`✅ ${pairs.length} pairs loaded\n`);
  if (pairs.length === 0) return console.error("Empty dataset");

  const DOCUMENTS = pairs.map(p => p.doc);
  const QUERIES = pairs.slice(0, NUM_QUERIES).map(p => p.query);

  // ---- 2. Embeddings (cached) ----
  let docEmbeddings = loadEmbeddings(DOC_CACHE);
  let queryEmbeddings = loadEmbeddings(QUERY_CACHE);

  if (!docEmbeddings || !queryEmbeddings) {
    console.log("Loading multilingual-e5-small model...");
    const extractor = await pipeline("feature-extraction", "Xenova/multilingual-e5-small");
    console.log("✅ Model loaded\n");

    if (!docEmbeddings) {
      console.log("Generating document embeddings...");
      docEmbeddings = [];
      for (let i = 0; i < DOCUMENTS.length; i++) {
        const out = await extractor(`passage: ${DOCUMENTS[i]}`, { pooling: "mean", normalize: true });
        const d = (out as any).data;
        docEmbeddings.push(d instanceof Float32Array ? d : new Float32Array(Array.from(d)));
        if ((i + 1) % 50 === 0) process.stdout.write(`  ${i + 1}/${DOCUMENTS.length}\r`);
      }
      console.log(`\n✅ ${DOCUMENTS.length} doc embeddings (dim=${docEmbeddings[0].length})`);
      saveEmbeddings(DOC_CACHE, docEmbeddings.map(e => Array.from(e)));
    }

    if (!queryEmbeddings) {
      console.log("Generating query embeddings...");
      queryEmbeddings = [];
      for (const q of QUERIES) {
        const out = await extractor(`query: ${q}`, { pooling: "mean", normalize: true });
        const d = (out as any).data;
        queryEmbeddings.push(d instanceof Float32Array ? d : new Float32Array(Array.from(d)));
      }
      console.log(`✅ ${QUERIES.length} query embeddings`);
      saveEmbeddings(QUERY_CACHE, queryEmbeddings.map(e => Array.from(e)));
    }
  }

  console.log();

  // ---- 3. Index ----
  const NUM_CLUSTERS = Math.max(16, Math.floor(Math.sqrt(DOCUMENTS.length)));
  console.log(`Training IVF (${NUM_CLUSTERS} clusters) + TurboQuant...`);
  const store = new RAGStore(NUM_CLUSTERS);
  store.train(docEmbeddings);
  for (let i = 0; i < DOCUMENTS.length; i++) {
    store.add(i.toString(), DOCUMENTS[i], docEmbeddings[i]);
  }
  console.log("✅ Indexed\n");

  // ---- 4. Benchmark ----
  console.log("=======================================");
  console.log("         BENCHMARK RESULTS");
  console.log("=======================================\n");

  const probeLevels = [
    Math.max(1, Math.floor(NUM_CLUSTERS * 0.1)),
    Math.max(1, Math.floor(NUM_CLUSTERS * 0.25)),
    Math.max(1, Math.floor(NUM_CLUSTERS * 0.5)),
    NUM_CLUSTERS, // exhaustive IVF
  ];

  for (const probes of probeLevels) {
    let totalRecall = 0;
    let totalDivergence = 0;
    let totalTime = 0;
    let allExact: number[] = [];
    let allApprox: number[] = [];

    for (let q = 0; q < NUM_QUERIES; q++) {
      const queryEmb = queryEmbeddings[q];

      // Ground truth: exact cosine top-k
      const exactScores = docEmbeddings.map((v, i) => ({ id: i.toString(), score: cosine(queryEmb, v) }));
      exactScores.sort((a, b) => b.score - a.score);
      const gtIds = new Set(exactScores.slice(0, K).map(t => t.id));

      // TurboQuant search
      const start = performance.now();
      const results = store.search(queryEmb, QUERIES[q], { k: K, probes, hybridMode: 'rrf' });
      totalTime += performance.now() - start;

      // Recall & Divergence
      const resIds = new Set(results.map(r => r.id));
      let hits = 0;
      for (const id of gtIds) if (resIds.has(id)) hits++;
      totalRecall += hits / K;
      totalDivergence += (K - hits);

      // Collect scores for Spearman (all docs, not just top-k)
      allExact.push(...exactScores.slice(0, 50).map(e => e.score));
    }

    const avgRecall = (totalRecall / NUM_QUERIES * 100).toFixed(2);
    const avgDivergence = (totalDivergence / NUM_QUERIES).toFixed(1);
    const avgTime = (totalTime / NUM_QUERIES).toFixed(2);
    const pct = ((probes / NUM_CLUSTERS) * 100).toFixed(0);
    console.log(`Probes=${probes} (${pct}% of ${NUM_CLUSTERS}): Recall@${K} (vs E5)=${avgRecall}% | Diverged=${avgDivergence} docs | ${avgTime} ms/q`);
  }

  // ---- 5. TurboQuant Exhaustive + Spearman ----
  console.log("\n--- TurboQuant IP Quality (Exhaustive) ---");
  const quant = new TurboQuant();
  const qvecs = docEmbeddings.map(v => quant.quantize(v));

  let totalExhaustiveRecall = 0;
  let globalExact: number[] = [];
  let globalApprox: number[] = [];

  for (let q = 0; q < NUM_QUERIES; q++) {
    const queryEmb = queryEmbeddings[q];
    const qRot = quant.rotateQuery(queryEmb);

    const exact = docEmbeddings.map((v, i) => ({ id: i, score: cosine(queryEmb, v) }));
    exact.sort((a, b) => b.score - a.score);
    const gtIds = new Set(exact.slice(0, K).map(t => t.id));

    const approx = qvecs.map((qv, i) => ({ id: i, score: quant.ip(qRot, qv) }));
    approx.sort((a, b) => b.score - a.score);
    const approxIds = new Set(approx.slice(0, K).map(r => r.id));

    let hits = 0;
    for (const id of gtIds) if (approxIds.has(id)) hits++;
    totalExhaustiveRecall += hits / K;

    // Spearman: compare full ranking for top-50
    const top50exact = exact.slice(0, 50);
    const approxMap = new Map(approx.map(a => [a.id, a.score]));
    for (const e of top50exact) {
      globalExact.push(e.score);
      globalApprox.push(approxMap.get(e.id) || 0);
    }
  }

  const exhaustiveRecall = (totalExhaustiveRecall / NUM_QUERIES * 100).toFixed(2);
  const spearman = spearmanCorrelation(globalExact, globalApprox).toFixed(4);

  console.log(`Exhaustive Recall@${K}: ${exhaustiveRecall}%`);
  console.log(`Spearman Rank Correlation (top-50): ${spearman}`);

  // ---- 6. Stats ----
  console.log("\n--- Engine Stats ---");
  console.dir(store.stats(), { depth: null });
  console.log("\n✅ Benchmark completo");
}

main().catch(console.error);
