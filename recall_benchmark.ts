import { TurboQuant, IVF, RAGStore } from './src/lib/rag';

function generateRandomVector(dim: number) {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = Math.random() * 2 - 1;
  return v;
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function l2Norm(v: Float32Array) {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

function cosine(a: Float32Array, b: Float32Array): number {
  return dot(a, b) / (l2Norm(a) * l2Norm(b));
}

async function runRecallBenchmark() {
  const dim = 384;
  const numVectors = 1000;
  const kSearch = 10;
  const numQueries = 50;
  
  console.log(`🚀 Measuring Recall@${kSearch} for ${numVectors} vectors (${dim} dim)\n`);
  
  const vectors = Array.from({ length: numVectors }, () => generateRandomVector(dim));
  const store = new RAGStore(32); // 32 clusters
  
  // Training
  store.train(vectors);
  for (let i = 0; i < numVectors; i++) {
    store.add(i.toString(), `doc_${i}`, vectors[i]);
  }

  let totalRecall = 0;

  for (let q = 0; q < numQueries; q++) {
    const query = generateRandomVector(dim);
    
    // Ground Truth (Linear Search Float32)
    const groundTruth = vectors
      .map((v, i) => ({ id: i.toString(), score: cosine(query, v) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, kSearch);
    const gtIds = new Set(groundTruth.map(t => t.id));

    // Approximate Search (TurboQuant + IVF)
    const results = store.search(query, kSearch, 32); // Exhaustive probes
    const resIds = results.map(r => r.id);

    // Calculate Intersection
    let hits = 0;
    for (const id of resIds) {
      if (gtIds.has(id)) hits++;
    }
    totalRecall += (hits / kSearch);
  }

  console.log(`--- Statistics (${numQueries} queries) ---`);
  console.log(`Recall@${kSearch}: ${(totalRecall / numQueries * 100).toFixed(2)}%`);
  console.log(`Config: IVF(32 clusters), Probes=4, TurboQuant(4-bit+1-bit)`);
  
  console.log("\n✅ Done");
}

runRecallBenchmark().catch(console.error);
