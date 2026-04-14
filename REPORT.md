# MiniFaiss / TurboRAG — Relatório Completo do Projeto

> **⚠️ INSTRUÇÃO PERMANENTE**: Este relatório DEVE ser atualizado sempre que houver mudanças significativas no projeto (motor, interfaces, UI, benchmarks, bugs). Atualizar ambas as cópias:  
> 1. Knowledge Item: `knowledge/minifaiss-status/artifacts/report.md`  
> 2. Projeto: `g:/dyad-apps/dyad-apps/MiniFaiss/REPORT.md`

> **Data**: 2026-04-14  
> **Workspace**: `g:/dyad-apps/dyad-apps/MiniFaiss`  
> **Stack**: React 19 + Vite + TypeScript + Tailwind CSS v4 + Comlink (Web Worker)  
> **Paper Base**: TurboQuant (arXiv 2504.19874) — Zandieh et al., ICLR 2026  

---

## 1. Visão Geral do Projeto

MiniFaiss é um **motor de busca vetorial browser-native** (100% client-side) que implementa:
- Quantização vetorial baseada no paper TurboQuant
- Indexação IVF (Inverted File Index) com K-means++
- Busca híbrida (semântica + léxica)
- Embeddings via `multilingual-e5-small` (HuggingFace Transformers.js)
- Persistência local via IndexedDB
- Suporte a ingestão de PDF e TXT
- Interface premium com glassmorphism, animações e Tailwind v4

---

## 2. Arquitetura Atual

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
│   App.tsx    │────▶│  worker.ts   │────▶│   src/lib/rag.ts     │
│  (React UI) │     │ (Web Worker) │     │  TurboQuant + IVF +  │
│             │     │  via Comlink │     │  RAGStore             │
└─────────────┘     └──────────────┘     └──────────────────────┘
                                                │
                                         ┌──────┴──────┐
                                         │  IndexedDB  │
                                         │ (TurboRAG_DB)│
                                         └─────────────┘
```

### Arquivos Principais

| Arquivo | Função |
|---------|--------|
| `src/lib/rag.ts` | Motor principal: TurboQuant, IVF, RAGStore |
| `src/types.ts` | Interfaces TypeScript (QVec, Result, EngineStats, etc.) |
| `src/App.tsx` | Interface React com dashboard de métricas |
| `src/worker.ts` | Web Worker que roda embeddings + indexação off-thread |
| `src/lib/pdf.ts` | Extração de texto de PDFs via pdfjs-dist |
| `recall_benchmark.ts` | Benchmark de recall e correlação de ranking |

---

## 3. Todas as Modificações Realizadas (Sessão 2026-04-14)

### 3.1 Remoção de Vetores Float32 (Otimização de Memória)

**Problema**: O `RAGStore` guardava AMBOS o vetor original (`full: Float32Array`) e a versão quantizada. A função `stats()` mentia sobre a economia de memória.

**Solução**:
- Removido `full: Float32Array` do store
- Atualizado `RAGIndexState` em `types.ts`
- Re-ranking agora usa scores quantizados + léxico (sem float32 exato)
- `stats()` reescrito com cálculo honesto de memória

**Arquivos alterados**: `src/lib/rag.ts`, `src/types.ts`, `src/App.tsx`

---

### 3.2 Reescrita do TurboQuant (Conforme Paper arXiv 2504.19874)

**Problema**: Implementação anterior usava centroids arbitrários, adaptive variance scaling incorreto, e 2-bit residual ad-hoc.

**Solução baseada no paper**:

#### Stage 1: MSE-Optimal Quantization
- Random rotation via **Hadamard transform** com sinais aleatórios (seed=42)
- **16 centroids Lloyd-Max** pré-computados para N(0, 1/d), escalados por `1/√d`
- Boundaries = midpoints entre centroids consecutivos
- Codebooks cacheados por dimensão

```typescript
// Centroids escalados para d=384:
const s = 1.0 / Math.sqrt(384); // ≈ 0.051
const centroids = [
  -2.733*s, -2.069*s, ..., 2.069*s, 2.733*s  // 16 valores
];
```

#### Stage 2: QJL (Quantized Johnson-Lindenstrauss) no Residual
- Conforme Seção 3.2 do paper: aplica 1-bit QJL no residual `r = rot - recon`
- Armazena `sign(S·r)` onde S é projeção aleatória diagonal (seed=7919)
- Dequantização: `gamma * √(π/2) / d * Σ(qRot[i] * sign[i] * randomSign[i])`
- Resultado: estimador **unbiased** de inner product

#### IP Estimation
```
<y, x_tilde> = <y, x_mse> + gamma * <y, x_qjl>
```
Onde:
- `dotMse = Σ qRot[i] * centroids[idx[i]]`
- `dotQjl = gamma * √(π/2)/d * Σ qRot[i] * sign[i] * randomSign[i]`

**API redesenhada**:
- `quant.rotateQuery(query)` → computa rotação UMA vez
- `quant.ip(qRot, qv)` → recebe rotação pré-computada (reutilizada para todos candidatos)

---

### 3.3 Interface QVec Atualizada

```typescript
export interface QVec {
  idx: Uint8Array;      // 4-bit centroid indices
  resIdx: Uint8Array;   // 1-bit QJL sign bits (1 bit/dim)
  norm: number;         // Original vector L2 norm
  resNorm: number;      // Residual L2 norm
  dim: number;          // Vector dimensionality
  scale: number;        // Unused (kept for compat, always 0)
  resScale: number;     // gamma = ||residual|| for QJL
}
```

---

### 3.4 EngineStats Honesto

```typescript
export interface EngineStats {
  docs: number;         // Total de documentos indexados
  dimensions: number;   // Total de dimensões (docs × dim_per_doc)
  memoryMB: string;     // Memória real alocada em MB
  ratio: string;        // Compressão real vs float32 (ex: "4.9x")
  saved: number;        // Bytes economizados vs float32
  clusters: number;     // Centroids IVF
}
```

Cálculo: `qBytes = ceil((totalDims * 6) / 8) + (totalDocs * 24)` onde 6 = 4-bit MSE + ~1-bit QJL + overhead.

---

### 3.5 Busca Híbrida

- **Pesos**: 0.82 Semantic (TurboQuant IP) + 0.18 Lexical
- **Lexical Score**: Ponderado pelo comprimento da palavra (`weight = min(token.length/8, 2)`)
- **Stopwords**: ~90 termos em Português Brasileiro
- **Tokenização**: NFD normalize → remove accents → remove punctuation → split → filter

---

### 3.6 Dashboard UI

- Métrica "Compressão" mostra ratio real (ex: "4.9x")
- Métrica "Memória Alocada" mostra MB reais ocupados
- Estado inicial corrigido para o novo `EngineStats`

---

### 3.7 VS Code Settings

Criado `.vscode/settings.json` com `"css.lint.unknownAtRules": "ignore"` para suprimir avisos de `@theme` e `@apply` do Tailwind CSS v4.

---

### 3.8 Benchmark Fix

- Corrigido tipo em `recall_benchmark.ts`: adicionado `queryStr` missing no `store.search()`
- Benchmark agora usa vetores normalizados (simula E5)
- Adicionada medição de correlação de Spearman
- Separação de testes: TurboQuant puro vs pipeline completo (IVF + TurboQuant)

---

## 4. Resultados dos Benchmarks (2026-04-14)

### Tabela Comparativa

| Dataset | Tipo | # Docs | # Queries | Recall@10 (max) | Tempo max | Spearman | Compressão |
|---------|------|--------|-----------|------------------|-----------|----------|------------|
| Vetores Aleatórios | Stress Test | 2000 | 100 | 49.6% | 24.4 ms | 0.8927 | 4.9x |
| SQuAD pairs (v1) | Semantic Only | 1000 | 100 | 62.0% | 41.45 ms | 0.8452 | **5.8x** |
| SQuAD pairs (v2) | **BM25 + Semantic (RRF)** | 1000 | 100 | **69.4%** | **8.87 ms** | 0.8452 | **5.8x** |

### Detalhes SQuAD 1000 docs (BM25 + Semantic via RRF)

*Pipeline: IVF (Semantic) + Inverted Index (BM25) fusionados com Reciprocal Rank Fusion (k=60).*

| Teste | Recall@10 | Tempo / Query |
|-------|-----------|---------------|
| IVF Probes=3 (10%) | 61.30% | 1.63 ms |
| IVF Probes=7 (23%) | 68.20% | 2.81 ms |
| IVF Probes=15 (48%) | 69.40% | 4.32 ms |
| IVF Probes=31 (100%) | 69.40% | 8.87 ms |
| **TurboQuant Exaustivo** | **76.60%** | — |

### Interpretação (Impacto do BM25 + RRF)
- **Quebra do "Teto de Vidro" do IVF:** Na v1 (apenas semantic), o IVF limitava o recall a 62%. Com a **dupla recuperação** (Semantic IVF + Lexical BM25) unida por RRF, o framework "resgata" candidatos perdidos, elevando o recall prático limit de 62.0% para **69.4%** (+7.4% de ganho real absoluto).
- **Latência ~5x Menor:** Trocar a tokenização no momento da query (que era O(n) por candidato) por um **Inverted Index BM25 clássico** derrubou drasticamente os tempos: uma busca completa na tabela (probes=31) caiu de 41.5ms para **menos de 9ms**. Usando Probes=3 alcançamos >61% de recall em incríveis **1.6ms** por query (perfeito para *search-as-you-type* 60fps no browser).
- **Compressão Intacta:** O Inverted Index e os dicionários são extremamente leves na memória (comparados a vetores em F32), portanto mantemos a métrica mágica de hospedar 1k docs ocupando ~0.25 MB de RAM para o search vetorial completo.

### Arquivos de Benchmark
| Arquivo | Função |
|---------|--------|
| `recall_benchmark.ts` | Stress test com vetores aleatórios + Spearman |
| `e5_benchmark.ts` | 100 docs PT-BR manualmente criados |
| `hf_benchmark.ts` | **SQuAD 1000 docs via HuggingFace** (com cache de embeddings e Spearman) |

### Como Avaliar Busca Híbrida Corretamente
A métrica de "Recall@10" baseada exclusivamente em embeddings (a nossa *ground truth* atual) pune o BM25 quando ele traz resultados perfeitinhos lexicalmente, mas diferentes dos top-10 do modelo E5. 

**Próximos Passos Recomendados para Benchmarking Real:**
1. **Human Evaluation (nDCG@10)**: Criar 30-50 queries em PT-BR e julgar manualmente a relevância dos top-10 reais.
2. **LLM-as-a-judge**: Usar um LLM (ou o próprio modelo embedding E5 via cross-encoder) para pontuar a relevância dos pares Query/Documento trazidos pela fusão híbrida.
*Nota: A busca híbrida foi otimizada (agora exporta `semanticScore`, `bm25Score` e expõe um parâmetro `hybridMode: 'rrf' | 'weighted'` no método `search`) para facilitar essa experimentação no futuro.*

---

## 5. Status Atual (Compilação & Lint)

- ✅ `npm run lint` (`tsc --noEmit`): **0 erros**
- ✅ `npm run dev`: rodando sem erros
- ✅ `hf_benchmark.ts` v2: funcional (1k docs, 76.6% exaustivo, Spearman 0.85, cache de embeddings)
- ✅ Centroids Lloyd-Max corrigidos para valores exatos (Paez & Glisson 1972)
- ✅ `stats()` corrigido para 5 bits/dim (compressão real 5.8x)

---

## 6. Bugs Corrigidos Nesta Sessão

| Bug | Causa | Fix |
|-----|-------|-----|
| TS2345 em `recall_benchmark.ts` | Faltava arg `queryStr` em `store.search()` | Adicionado `""` como segundo argumento |
| Memória superestimada | `stats()` contava float32 como "economia" mas armazenava ambos | Removido `full: Float32Array`, cálculo honesto |
| IP scaling errado | `ip()` multiplicava por `qv.norm`, clampava [0,1] | Removido, usar score direto |
| Cache poisoning | Query rotation cache com key de 2 floats → colisões | Removido cache, rotação direta |
| Search cache stale | Cache baseado só em queryStr → vetores diferentes mesmos resultados | Removido searchCache |
| Quantization off-by-one | Binary search retornava `low-1` em vez de `low` | Corrigido para `low` |
| Centroids inexatos | Lloyd-Max internos desviavam ~0.01 dos valores ótimos | Corrigidos para valores exatos (Max 1960) |
| bits/dim errado | `stats()` usava 6 bits/dim mas real é 5 (4 MSE + 1 QJL) | Corrigido para 5 bits/dim |

---

## 7. Estrutura de Dados Persistidos (IndexedDB)

- **DB Name**: `TurboRAG_DB`
- **Object Store**: `index_state`
- **Key**: `current_index`
- **Value**: `RAGIndexState` (centroids IVF + lists + store com QVec)

---

## 8. Dependências Principais

```json
{
  "@huggingface/transformers": "^4.0.1",  // Embeddings E5-small
  "comlink": "^4.4.2",                     // Web Worker bridge
  "idb": "^8.0.3",                         // IndexedDB wrapper
  "pdfjs-dist": "^5.6.205",               // PDF parsing
  "motion": "^12.38.0",                    // Animations (Framer Motion)
  "lucide-react": "^0.546.0",             // Icons
  "tailwindcss": "^4.1.14",               // CSS framework (v4)
  "react": "^19.0.0",                     // UI framework
  "vite": "^6.2.0"                        // Build tool
}
```

---

## 9. Próximos Passos

1. ~~Testar com embeddings E5 reais~~ ✅ Feito
2. ~~Benchmark maior (1000+ docs)~~ ✅ Feito — 76.6% exaustivo, 62% IVF
3. ~~Corrigir stats() para 5 bits/dim~~ ✅ Feito — compressão real 5.8x
4. **Melhorar IVF**: O gap exaustivo (76.6%) vs IVF (62%) sugere que o IVF está perdendo ~15% de recall. Possíveis soluções:
   - Aumentar clusters (atualmente √n = 31)
   - Implementar multi-probe LSH
   - Re-ranking com mais candidatos
5. **QJL Gaussiano completo**: Implementar projeção gaussiana densa S ∈ R^{m×d} conforme paper Definition 1 (atualmente usa diagonal)
6. **Benchmark PT-BR em escala**: Usar dataset FaQuAD para 1000+ docs em Português
7. **Entropy encoding**: Paper seção 3.1 menciona ~5% de redução com prefix codes
8. **UI/UX**: Mostrar Spearman e recall estimado no dashboard

---

## 10. Referência Rápida do Paper

**TurboQuant (arXiv 2504.19874)**:
- MSE distortion para b-bit: `D_mse ≤ √3·π/2 · 1/4^b` (~2.7× do lower bound)
- Para b=4: `D_mse ≈ 0.009` (quase imperceptível)
- Inner Product TurboQuant: MSE quantizer (b-1 bits) + 1-bit QJL no residual → **unbiased**
- IP distortion: `D_prod ≈ 0.047/d` para b=4
- Centroids exatos para b=4 (16 níveis): `{±0.1284, ±0.3881, ±0.6568, ±0.9423, ±1.2562, ±1.6180, ±2.0690, ±2.7326} * 1/√d`
