# ⚡ MiniFaiss
### High-Performance, Browser-Native Vector Database & RAG Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E)](https://vitejs.dev/)


---

## 🚀 Overview

**MiniFaiss** is a state-of-the-art vector search engine built to run entirely in the browser. It implements the cutting-edge **TurboQuant (ICLR 2026)** quantization research, providing massive memory savings (6x+) with near-zero accuracy loss.

Optimized for **Brazilian Portuguese (PT-BR)**, MiniFaiss enables privacy-first, local-only RAG applications without any server dependency.

## ✨ Key Features

- **TurboQuant-prod Engine**: 4-bit MSE quantization with 1-bit residual correction for unbiased inner product estimation.
- **Hybrid Search (BM25 + Semantic)**: Dual retrieval pipeline combining sub-linear IVF spatial clustering with an Okapi TF-IDF Inverted Index, merged via Reciprocal Rank Fusion (RRF).
- **PT-BR Optimized**: Light Stemming optimized for Portuguese and powered by `multilingual-e5-small`.
- **Web Worker Architecture**: Offloads embedding and index training to background threads ensuring 60fps UI performance.
- **Format Support**: Native extraction and chunking for **PDF** and **TXT** files.
- **Local Persistence**: Automatically saves and loads your semantic + lexical indices from **IndexedDB**.
- **Real-time Metrics**: Visualization of compression ratios, bitrate, and search confidence.
- **Premium UI**: Modern dark-mode dashboard with advanced glassmorphism aesthetics.

## 🛠️ Technology Stack

- **Core Logic**: TypeScript + Vanilla JS (TurboQuant Math)
- **Frontend**: React + Motion + Lucide Icons
- **ML Inference**: Transformers.js (ONNX Runtime)
- **PDF Processing**: PDF.js
- **State Management**: Comlink (Worker RPC) & IndexedDB

## 📦 Getting Started

### Prerequisites

- Node.js (v18+)
- NPM or PNPM

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Benevalterjr/MiniFaiss.git
   cd MiniFaiss
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## 📐 Architecture & Research

MiniFaiss implements a Randomized Walsh-Hadamard Transform (RWHT) to distribute vector information before quantization. This approach, combined with dynamic variance scaling, allows high-dimensional embeddings (384-dim) to be compressed into just a few bits per dimension while maintaining ranking precision.

For more details, see our internal documentation:
- 📄 [Product Requirements Document (PRD)](PRD.md)
- ⚙️ [Technical Specification](TECHNICAL_SPEC.md)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Built with ❤️ for the Local AI Community</p>
</div>
