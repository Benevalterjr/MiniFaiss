# Technical Specification: TurboRAG Engine

**Version:** 1.0  
**Technology Stack:** React, TypeScript, Vite, Comlink, Transformers.js, PDF.js, IDB.

---

## 1. System Architecture
TurboRAG follows a **Local-First, Worker-Centric** architecture.

- **Main Thread (UI)**: React application handling user interactions, document management, and visualization.
- **Worker Thread (Engine)**: Offloads heavy computation (Transformers.js inference, Vector Quantization, IVF Training).
- **Communication Layer**: `comlink` for RPC-style communication between threads.
- **Storage Layer**: IndexedDB via `idb` for persisting vector indices and document contents.

## 2. The TurboQuant-prod Algorithm
The core of the engine implements the research from **"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)**.

### 2.1 Quantization Workflow
1.  **Normalization**: Input vector $x$ is normalized to unit length.
2.  **Hadamard Rotation**: A Randomized Walsh-Hadamard Transform (RWHT) is applied to spread information evenly across all dimensions ($x_{rot} = H \cdot D \cdot x$).
3.  **Variance Scaling**: Rotated values are scaled by $\sqrt{dim}$ to align with the standard Normal distribution ($\mathcal{N}(0,1)$).
4.  **4-bit Quantization (MSE)**: Each dimension is mapped to one of 16 centroids optimized for MSE.
5.  **1-bit Residual (Unbiased)**: The residual error $r = x_{scaled} - \hat{x}_{scaled}$ is captured as a bitmask (signs only).

### 2.2 Inner Product Estimation
The estimated inner product $IP(q, x)$ is calculated as:
$$IP \approx (q_{rot} \cdot \hat{x}_{rot}) + (\text{resScale} \cdot (q_{rot} \cdot \text{sgn}(r)))$$
Where `resScale` is the stored L2-norm of the residual vector.

## 3. Indexing Strategy: IVF (Inverted File)
To ensure sub-linear search performance as the database grows:
- **Centroids**: Calculated using K-Means clustering.
- **Assignment**: Each document is assigned to its nearest centroid.
- **Search**: The query is compared against centroids, and only documents in the nearest $P$ (probes) clusters are scanned.
- **Dynamic Probing**: For small datasets ($N < 50$), the engine automatically increases probes to include all clusters (Brute Force fallback) to ensure 100% recall.

## 4. Text Processing
- **Model**: `Xenova/multilingual-e5-small`.
- **Dimensions**: 384.
- **Prefixes**: `query: ` for searches, `passage: ` for indexing.
- **Chunking**: Narrative splitting with 1000-character target, splitting by sentence boundaries for large documents.

## 5. Security & Privacy
- **Zero Cloud Leakage**: No telemetry or embedding data is sent to external servers.
- **In-Memory Volatility**: Search queries are processed in memory and never persisted.
- **IndexedDB**: Standard browser security model applies.
