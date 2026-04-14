import { TurboQuant, RAGStore } from '../src/lib/rag';

function generateRandomNormalizedVector(dim: number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = Math.random() * 2 - 1;
  const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  for (let i = 0; i < dim; i++) v[i] /= norm || 1;
  return v;
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

// Spearman rank correlation
function spearmanCorrelation(a: number[], b: number[]): number {
  const n = a.length;
  const rankA = getRanks(a);
  const rankB = getRanks(b);
  let sumD2 = 0;
  for (let i = 0; i < n; i++) {
    const d = rankA[i] - rankB[i];
    sumD2 += d * d;
  }
  return 1 - (6 * sumD2) / (n * (n * n - 1));
}

function getRanks(arr: number[]): number[] {
  const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array(arr.length);
  sorted.forEach((item, rank) => { ranks[item.i] = rank; });
  return ranks;
}

async function runRecallBenchmark() {
  const dim = 384;
  const numVectors = 2000;
  const kSearch = 10;
  const numQueries = 100;

  console.log(`🚀 MiniFaiss Recall@${kSearch} Benchmark`);
  console.log(`Vectors: ${numVectors} | Dim: ${dim} | Queries: ${numQueries}\n`);

  const vectors = Array.from({ length: numVectors }, () => generateRandomNormalizedVector(dim));

  // ===== TEST 1: IP Quality (Correlation & Recall) =====
  console.log("--- Test 1: TurboQuant IP quality (exhaustive, no IVF) ---");
  const quant = new TurboQuant();
  const qvecs = vectors.map(v => quant.quantize(v));

  let recallQuant = 0;
  let totalCorrelation = 0;

  for (let q = 0; q < numQueries; q++) {
    const query = generateRandomNormalizedVector(dim);
    const qRot = quant.rotateQuery(query);

    const exactScores = vectors.map((v, i) => ({ id: i, score: cosine(query, v) }));
    const approxScores = qvecs.map((qv, i) => ({ id: i, score: quant.ip(qRot, qv) }));

    // Spearman correlation between exact and approx scores
    totalCorrelation += spearmanCorrelation(
      exactScores.map(s => s.score),
      approxScores.map(s => s.score)
    );

    // Recall
    const gtIds = new Set(
      exactScores.sort((a, b) => b.score - a.score).slice(0, kSearch).map(t => t.id)
    );
    const approxIds = new Set(
      approxScores.sort((a, b) => b.score - a.score).slice(0, kSearch).map(r => r.id)
    );
    let hits = 0;
    for (const id of gtIds) if (approxIds.has(id)) hits++;
    recallQuant += hits / kSearch;
  }
  console.log(`TurboQuant Recall@${kSearch}: ${(recallQuant / numQueries * 100).toFixed(2)}%`);
  console.log(`Spearman Rank Correlation: ${(totalCorrelation / numQueries).toFixed(4)}`);
  console.log(`(1.0 = perfect ranking, expected ~0.35-0.55 for 4+1 bit on random vectors)\n`);

  // ===== TEST 2: Full pipeline =====
  for (const probes of [8, 16, 32]) {
    const store = new RAGStore(32);
    store.train(vectors);
    for (let i = 0; i < numVectors; i++) {
      store.add(i.toString(), `doc ${i}`, vectors[i]);
    }

    let recallFull = 0;
    let totalTime = 0;
    for (let q = 0; q < numQueries; q++) {
      const query = generateRandomNormalizedVector(dim);
      const start = performance.now();

      const groundTruth = vectors
        .map((v, i) => ({ id: i.toString(), score: cosine(query, v) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, kSearch);
      const gtIds = new Set(groundTruth.map(t => t.id));

      const results = store.search(query, "", { k: kSearch, probes });
      const resIds = new Set(results.map(r => r.id));

      let hits = 0;
      for (const id of gtIds) if (resIds.has(id)) hits++;
      recallFull += hits / kSearch;
      totalTime += performance.now() - start;
    }
    console.log(`IVF Probes=${probes}: Recall@${kSearch}=${(recallFull / numQueries * 100).toFixed(2)}% | ${(totalTime / numQueries).toFixed(2)} ms/query`);
  }

  // Stats
  const store = new RAGStore(32);
  store.train(vectors);
  for (let i = 0; i < numVectors; i++) store.add(i.toString(), `doc ${i}`, vectors[i]);
  console.log("\nMemory Stats:", store.stats());
  console.log("\n✅ Done");
}

runRecallBenchmark().catch(console.error);
