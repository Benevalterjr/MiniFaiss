import { useState, useEffect, useRef, useMemo, FormEvent, ChangeEvent } from 'react';
import { pipeline, env } from "@huggingface/transformers";
import { motion, AnimatePresence } from "motion/react";
import { 
  Search, 
  Database, 
  Zap, 
  Cpu, 
  Info, 
  CheckCircle2, 
  Loader2, 
  Terminal,
  ArrowRight,
  RefreshCw,
  Layers,
  ShieldCheck,
  Upload,
  FileText,
  X,
  Plus
} from "lucide-react";
import { RAGStore, Result } from "./lib/rag";
import { cn } from "./lib/utils";

// Configure transformers for browser
env.allowLocalModels = false;
env.useBrowserCache = true;

// Fix for WASM Out of Memory errors
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;
}

interface Document {
  id: string;
  text: string;
  source?: string;
}

const INITIAL_DOCS: Document[] = [
  { id: "1", text: "O Brasil é o maior país da América do Sul, com área de 8,5 milhões de km².", source: "Brasil Info" },
  { id: "2", text: "A Amazônia abriga a maior floresta tropical do mundo e é fundamental para o clima global.", source: "Natureza" },
  { id: "3", text: "O carnaval brasileiro é uma das maiores festas populares do mundo, celebrado em fevereiro ou março.", source: "Cultura" },
  { id: "4", text: "O futebol é a paixão nacional do Brasil, com cinco títulos mundiais conquistados pela seleção.", source: "Esporte" },
  { id: "5", text: "São Paulo é a maior cidade do Brasil e centro financeiro da América Latina.", source: "Cidades" },
];

export default function App() {
  const [status, setStatus] = useState<'idle' | 'loading-model' | 'indexing' | 'ready' | 'error'>('idle');
  const [progress, setProgress] = useState(0);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [documents, setDocuments] = useState<Document[]>(INITIAL_DOCS);
  const [isDragging, setIsDragging] = useState(false);
  
  const extractorRef = useRef<any>(null);
  const storeRef = useRef<RAGStore>(new RAGStore());
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addLog = (msg: string) => {
    setLogs(prev => [...prev.slice(-4), msg]);
  };

  const initModel = async () => {
    try {
      setStatus('loading-model');
      addLog("Iniciando pipeline de extração...");
      
      const hasCache = await storeRef.current.loadFromIndexedDB();
      if (hasCache) {
        addLog("Índice carregado do cache local.");
      }

      const modelName = "Xenova/all-MiniLM-L6-v2";
      addLog(`Carregando ${modelName}...`);

      const extractor = await pipeline("feature-extraction", modelName, { 
        progress_callback: (p: any) => {
          if (p.status === 'progress') {
            setProgress(Math.round(p.progress));
          }
        }
      });
      
      extractorRef.current = extractor;
      addLog(`Modelo ${modelName} pronto.`);
      
      if (hasCache) {
        setStatus('ready');
      } else {
        await indexDocuments(INITIAL_DOCS);
      }
    } catch (err) {
      console.error(err);
      setStatus('error');
      addLog("Erro de Memória/WASM.");
    }
  };

  const embed = async (text: string): Promise<Float32Array> => {
    const out: any = await extractorRef.current(text, { pooling: "mean", normalize: true });
    const data = out.data ?? out[0]?.data;
    return data instanceof Float32Array ? data : new Float32Array(Array.from(data));
  };

  const indexDocuments = async (docsToIndex: Document[]) => {
    setStatus('indexing');
    addLog(`Indexando ${docsToIndex.length} documentos...`);
    
    storeRef.current.clear();
    const embeddings: Float32Array[] = [];
    
    for (let i = 0; i < docsToIndex.length; i++) {
      const doc = docsToIndex[i];
      const emb = await embed(`passage: ${doc.text}`);
      embeddings.push(emb);
      setProgress(Math.round(((i + 1) / docsToIndex.length) * 50));
    }

    addLog("Treinando clusters IVF...");
    storeRef.current.train(embeddings);
    setProgress(75);

    addLog("Finalizando indexação...");
    for (let i = 0; i < docsToIndex.length; i++) {
      storeRef.current.add(docsToIndex[i].id, docsToIndex[i].text, embeddings[i]);
    }
    
    await storeRef.current.saveToIndexedDB();
    
    setProgress(100);
    setStatus('ready');
    addLog("Motor MiniFaiss pronto.");
  };

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    
    const newDocs: Document[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.type !== 'text/plain' && !file.name.endsWith('.txt')) continue;
      
      const text = await file.text();
      // Split by double newlines to create chunks if the file is large
      const chunks = text.split(/\n\s*\n/).filter(c => c.trim().length > 10);
      
      chunks.forEach((chunk, idx) => {
        newDocs.push({
          id: `upload-${Date.now()}-${i}-${idx}`,
          text: chunk.trim(),
          source: file.name
        });
      });
    }

    if (newDocs.length > 0) {
      const updatedDocs = [...documents, ...newDocs];
      setDocuments(updatedDocs);
      await indexDocuments(updatedDocs);
    }
  };

  const clearCache = async () => {
    if (confirm("Deseja limpar o cache do índice? Isso forçará uma nova indexação.")) {
      await storeRef.current.deleteDatabase();
      window.location.reload();
    }
  };

  const handleSearch = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!query.trim() || status !== 'ready' || isSearching) return;

    setIsSearching(true);
    try {
      const qEmb = await embed(`query: ${query}`);
      const searchResults = storeRef.current.search(qEmb, 3, 2);
      setResults(searchResults);
      addLog(`Busca concluída.`);
    } catch (err) {
      console.error(err);
      addLog("Erro na busca.");
    } finally {
      setIsSearching(false);
    }
  };

  const stats = useMemo(() => storeRef.current.stats(), [status, documents]);

  useEffect(() => {
    initModel();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center p-4 md:p-8 selection:bg-brand-500/30">
      {/* Header */}
      <header className="w-full max-w-5xl flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-brand-500/10 rounded-xl border border-brand-500/20">
              <Zap className="w-6 h-6 text-brand-400" />
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white">
              Turbo<span className="text-brand-400">RAG</span>
            </h1>
          </div>
          <p className="text-slate-400 max-w-md">
            Busca vetorial ultra-comprimida com suporte a upload de arquivos TXT.
          </p>
        </div>

        <div className="flex items-center gap-4">
          <button 
            onClick={clearCache}
            className="p-2 text-slate-500 hover:text-red-400 transition-colors"
            title="Limpar Cache"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <div className="flex flex-col items-end">
            <div className="flex items-center gap-2 text-xs font-mono text-slate-500 uppercase tracking-wider">
              <Cpu className="w-3 h-3" />
              Status do Motor
            </div>
            <div className="flex items-center gap-2 mt-1">
              <div className={cn(
                "w-2 h-2 rounded-full animate-pulse",
                status === 'ready' ? "bg-emerald-500" : status === 'error' ? "bg-red-500" : "bg-brand-500"
              )} />
              <span className="text-sm font-medium text-slate-200">
                {status === 'loading-model' && "Carregando Modelo..."}
                {status === 'indexing' && "Indexando..."}
                {status === 'ready' && "Motor Pronto"}
                {status === 'error' && "Erro Crítico"}
                {status === 'idle' && "Iniciando..."}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Sidebar */}
        <div className="lg:col-span-4 space-y-6">
          {/* Upload Area */}
          <section 
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); handleFileUpload(e.dataTransfer.files); }}
            className={cn(
              "bg-slate-900/50 border-2 border-dashed rounded-2xl p-6 backdrop-blur-sm transition-all text-center group cursor-pointer",
              isDragging ? "border-brand-500 bg-brand-500/5" : "border-slate-800 hover:border-slate-700"
            )}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept=".txt" 
              multiple 
              onChange={(e) => handleFileUpload(e.target.files)}
            />
            <div className="flex flex-col items-center gap-3">
              <div className="p-3 bg-slate-800 rounded-xl group-hover:bg-brand-500/10 group-hover:text-brand-400 transition-colors">
                <Upload className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">Upload de TXT</h3>
                <p className="text-xs text-slate-500 mt-1">Arraste ou clique para adicionar documentos</p>
              </div>
            </div>
          </section>

          {/* Stats Card */}
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-6 flex items-center gap-2">
              <Layers className="w-4 h-4" />
              Métricas do Índice
            </h2>
            
            <div className="space-y-6">
              <div className="flex items-end justify-between">
                <div>
                  <div className="text-3xl font-bold text-white">{stats.ratio}×</div>
                  <div className="text-xs text-slate-500 mt-1">Taxa de Compressão</div>
                </div>
                <div className="text-right">
                  <div className="text-xl font-semibold text-brand-400">4-bit</div>
                  <div className="text-xs text-slate-500 mt-1">Quantização</div>
                </div>
              </div>

              <div className="pt-6 border-t border-slate-800/50 space-y-4">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Documentos</span>
                  <span className="text-slate-200 font-mono">{stats.docs}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Clusters IVF</span>
                  <span className="text-brand-400 font-mono">{stats.clusters}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Memória Salva</span>
                  <span className="text-emerald-400 font-mono">+{stats.saved.toLocaleString()} bytes</span>
                </div>
              </div>
            </div>
          </section>

          {/* Console Card */}
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Terminal className="w-4 h-4" />
              Console
            </h2>
            <div className="bg-black/40 rounded-lg p-4 font-mono text-xs space-y-2 min-h-[120px]">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-brand-500/50 shrink-0">›</span>
                  <span className={cn(
                    i === logs.length - 1 ? "text-brand-400" : "text-slate-500"
                  )}>{log}</span>
                </div>
              ))}
              {status === 'loading-model' || status === 'indexing' ? (
                <div className="pt-2">
                  <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                    <motion.div 
                      className="bg-brand-500 h-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                    />
                  </div>
                  <div className="text-[10px] text-slate-600 mt-1 text-right">{progress}%</div>
                </div>
              ) : null}
            </div>
          </section>
        </div>

        {/* Search & Results */}
        <div className="lg:col-span-8 space-y-6">
          {/* Search Bar */}
          <form onSubmit={handleSearch} className="relative group">
            <div className="absolute inset-y-0 left-5 flex items-center pointer-events-none">
              {isSearching ? (
                <Loader2 className="w-5 h-5 text-brand-400 animate-spin" />
              ) : (
                <Search className="w-5 h-5 text-slate-500 group-focus-within:text-brand-400 transition-colors" />
              )}
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Perquise nos documentos..."
              disabled={status !== 'ready'}
              className="w-full bg-slate-900 border border-slate-800 text-white rounded-2xl py-5 pl-14 pr-6 focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500/50 transition-all placeholder:text-slate-600 disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={status !== 'ready' || !query.trim() || isSearching}
              className="absolute right-3 inset-y-3 px-4 bg-brand-600 hover:bg-brand-500 disabled:bg-slate-800 text-white rounded-xl font-medium text-sm transition-colors flex items-center gap-2"
            >
              Buscar
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          {/* Results Area */}
          <div className="space-y-4">
            <AnimatePresence mode="popLayout">
              {results.length > 0 ? (
                results.map((result, idx) => (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ delay: idx * 0.1 }}
                    className="bg-slate-900/40 border border-slate-800/60 hover:border-brand-500/30 rounded-2xl p-6 transition-all group"
                  >
                    <div className="flex items-start justify-between gap-4 mb-3">
                      <div className="flex items-center gap-2">
                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-brand-500/10 text-brand-400 text-xs font-bold border border-brand-500/20">
                          {idx + 1}
                        </span>
                        <div className="flex flex-col">
                          <span className="text-xs font-mono text-slate-500 uppercase tracking-widest">Documento #{result.id.slice(-4)}</span>
                          {documents.find(d => d.id === result.id)?.source && (
                            <span className="text-[10px] text-slate-600 font-medium truncate max-w-[150px]">Fonte: {documents.find(d => d.id === result.id)?.source}</span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1.5 px-2 py-1 bg-brand-500/10 border border-brand-500/20 rounded-md">
                          <Database className="w-3 h-3 text-brand-400" />
                          <span className="text-[10px] font-bold text-brand-400 uppercase tracking-tighter">Cluster {result.centroid}</span>
                        </div>
                        <div className="flex items-center gap-1.5 px-2 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-md">
                          <ShieldCheck className="w-3 h-3 text-emerald-400" />
                          <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-tighter">Score {result.score.toFixed(4)}</span>
                        </div>
                      </div>
                    </div>
                    <p className="text-slate-200 leading-relaxed">
                      {result.text}
                    </p>
                  </motion.div>
                ))
              ) : status === 'ready' && !isSearching ? (
                <div className="flex flex-col items-center justify-center py-20 text-center space-y-4 opacity-40">
                  <div className="p-4 bg-slate-900 rounded-full">
                    <Database className="w-12 h-12 text-slate-700" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium text-slate-400">Nenhum resultado ainda</h3>
                    <p className="text-sm text-slate-500">Digite uma pergunta acima para iniciar a busca vetorial.</p>
                  </div>
                </div>
              ) : isSearching ? (
                <div className="space-y-4">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="bg-slate-900/20 border border-slate-800/40 rounded-2xl p-6 animate-pulse">
                      <div className="h-4 bg-slate-800 rounded w-1/4 mb-4" />
                      <div className="h-4 bg-slate-800 rounded w-full mb-2" />
                      <div className="h-4 bg-slate-800 rounded w-2/3" />
                    </div>
                  ))}
                </div>
              ) : null}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* Footer Info */}
      <footer className="mt-auto pt-12 pb-8 w-full max-w-5xl">
        <div className="bg-brand-500/5 border border-brand-500/10 rounded-2xl p-6 flex flex-col md:flex-row items-center gap-6">
          <div className="p-3 bg-brand-500/10 rounded-xl">
            <Info className="w-6 h-6 text-brand-400" />
          </div>
          <div className="flex-1 text-center md:text-left">
            <h4 className="text-sm font-semibold text-white mb-1">Como funciona?</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Este app utiliza o modelo <code className="text-brand-300">all-MiniLM-L6-v2</code> para gerar embeddings. 
              O <strong>TurboQuant</strong> comprime esses vetores para 4-bit, e o <strong>IVF</strong> organiza em clusters para busca rápida.
              Você pode subir seus próprios arquivos TXT para testar a busca em tempo real.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
