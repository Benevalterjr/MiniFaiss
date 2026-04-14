import { useState, useEffect, useRef, useMemo, FormEvent } from 'react';
import { wrap, proxy } from "comlink";
import { motion, AnimatePresence } from "motion/react";
import { 
  Search, 
  Database, 
  Zap, 
  Cpu, 
  Info, 
  Loader2, 
  Terminal,
  ArrowRight,
  RefreshCw,
  Layers,
  ShieldCheck,
  Upload,
  FileText,
  X,
  FileSearch,
  CheckCircle2,
  AlertTriangle
} from "lucide-react";
import { cn } from "./lib/utils";
import { extractTextFromPDF } from "./lib/pdf";
import { Remote } from "comlink";
import { TurboRAGWorker } from "./worker";
import { Document, EngineStats, Result } from "./types";

const INITIAL_DOCS: Document[] = [
  { id: "1", text: "O MiniFaiss é uma técnica de quantização de vetores apresentada no ICLR 2026 que atinge distorção quase ótima.", source: "Artigo MiniFaiss", type: 'txt' },
  { id: "2", text: "O uso de quantização em dois estágios (MSE + Residual) permite estimativas de produto escalar não tendenciosas.", source: "Artigo MiniFaiss", type: 'txt' },
  { id: "3", text: "Este projeto utiliza o modelo multilingual-e5-small para suporte superior ao Português do Brasil.", source: "Sistema", type: 'txt' },
];

export default function App() {
  const [status, setStatus] = useState<'idle' | 'loading-model' | 'indexing' | 'ready' | 'error' | 'processing'>('idle');
  const [progress, setProgress] = useState(0);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [documents, setDocuments] = useState<Document[]>(INITIAL_DOCS);
  const [isDragging, setIsDragging] = useState(false);
  const [stats, setStats] = useState<EngineStats>({ 
    docs: 0, 
    dimensions: 0,
    memoryMB: "0.00",
    ratio: "0.0x", 
    saved: 0, 
    clusters: 0 
  });
  const [showWipeConfirm, setShowWipeConfirm] = useState(false);
  
  const workerApiRef = useRef<Remote<TurboRAGWorker> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addLog = (msg: string) => {
    setLogs(prev => [...prev.slice(-8), msg]);
  };

  useEffect(() => {
    const init = async () => {
      try {
        setStatus('loading-model');
        addLog("Iniciando MiniFaiss Worker...");
        
        const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
        const api = wrap<TurboRAGWorker>(worker);
        workerApiRef.current = api;

        const { hasCache } = await api.init(proxy((p: { status: string; progress: number }) => {
          if (p.status === 'progress') setProgress(Math.round(p.progress));
        }));

        if (hasCache) {
          addLog("Índice carregado do IndexedDB.");
          const currentStats = await api.getStats();
          setStats(currentStats);
          setStatus('ready');
        } else {
          await indexDocuments(INITIAL_DOCS);
        }
      } catch (err) {
        console.error(err);
        setStatus('error');
        addLog("Erro ao inicializar worker.");
      }
    };

    init();
  }, []);

  const indexDocuments = async (docsToIndex: Document[]) => {
    if (!workerApiRef.current) return;
    setStatus('indexing');
    addLog(`Indexando ${docsToIndex.length} documentos com MiniFaiss-prod...`);
    
    try {
      const newStats = await workerApiRef.current.indexDocuments(
        docsToIndex.map(d => ({ id: d.id, text: d.text, metadata: { source: d.source, type: d.type } })),
        proxy((p: number) => setProgress(p))
      );
      
      setStats(newStats);
      setStatus('ready');
      addLog("Motor pronto e calibrado.");
    } catch (err) {
      console.error(err);
      setStatus('error');
      addLog("Erro na indexação.");
    }
  };

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0 || status !== 'ready') return;
    
    setStatus('processing');
    const newDocs: Document[] = [];
    addLog(`Iniciando pipeline para ${files.length} arquivo(s)...`);

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        addLog(`[${i + 1}/${files.length}] Extraindo: ${file.name}...`);
        let text = "";
        let type: 'txt' | 'pdf' = 'txt';

        if (file.name.endsWith('.pdf')) {
          text = await extractTextFromPDF(file);
          type = 'pdf';
        } else if (file.name.endsWith('.txt')) {
          text = await file.text();
          type = 'txt';
        } else continue;

        // Professional Chunking: target ~1000 chars per chunk
        const targetSize = 1000;
        const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 0);
        
        paragraphs.forEach((para) => {
          const p = para.trim();
          if (p.length <= targetSize + 200) {
            // Paragraph is small enough, keep as one chunk
            if (p.length > 10) {
              newDocs.push({
                id: `doc-${Date.now()}-${i}-${newDocs.length}`,
                text: p,
                source: file.name,
                type
              });
            }
          } else {
            // Paragraph is too large, split by sentences or blocks
            const sentences = p.split(/([.!?]\s+)/);
            let currentChunk = "";
            
            for (const part of sentences) {
              if ((currentChunk + part).length > targetSize && currentChunk.length > 0) {
                newDocs.push({
                  id: `doc-${Date.now()}-${i}-${newDocs.length}`,
                  text: currentChunk.trim(),
                  source: file.name,
                  type
                });
                currentChunk = "";
              }
              currentChunk += part;
            }
            if (currentChunk.trim().length > 10) {
              newDocs.push({
                id: `doc-${Date.now()}-${i}-${newDocs.length}`,
                text: currentChunk.trim(),
                source: file.name,
                type
              });
            }
          }
        });
        addLog(`✓ ${file.name} fragmentado.`);
      }

      if (newDocs.length > 0) {
        addLog(`Total: ${newDocs.length} novos fragmentos detectados.`);
        setDocuments(prev => {
          const updated = [...prev, ...newDocs];
          // Trigger re-indexing with the updated combined set
          indexDocuments(updated);
          return updated;
        });
      } else {
        addLog("Aviso: Nenhum texto extraído do arquivo.");
        setStatus('ready');
      }
    } catch (err) {
      console.error(err);
      setStatus('error');
      addLog("Erro no processamento de arquivos.");
    }
  };

  const handleSearch = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!query.trim() || status !== 'ready' || isSearching) return;

    setIsSearching(true);
    addLog(`Buscando: "${query.substring(0, 20)}..."`);
    try {
      const searchResults = await workerApiRef.current.search(query, 4, 3);
      setResults(searchResults as Result[]);
    } catch (err) {
      console.error(err);
      addLog("Erro na busca.");
    } finally {
      setIsSearching(false);
    }
  };

  const removeDoc = async (source: string) => {
    const updatedDocs = documents.filter(d => d.source !== source);
    setDocuments(updatedDocs);
    await indexDocuments(updatedDocs);
  };

  const clearCache = async () => {
    if (!workerApiRef.current) return;
    addLog("Limpando banco de dados...");
    await workerApiRef.current.clearCache();
    window.location.reload();
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 selection:bg-brand-500/30 font-sans">
      <div className="max-w-[1400px] mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Brand & Controls */}
        <aside className="lg:col-span-3 space-y-8">
          <header className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-brand-500/10 rounded-xl border border-brand-500/20 shadow-lg shadow-brand-500/5">
                <Zap className="w-8 h-8 text-brand-400" />
              </div>
              <div>
                <h1 className="text-xl font-black tracking-tighter text-white">MiniFaiss</h1>
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500 mt-1 font-mono">ICLR 2026 Engine</p>
              </div>
            </div>
          </header>

          {/* Engine Status */}
          <section className="glass-card p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5" /> Motor
              </span>
              <div className={cn(
                "w-2 h-2 rounded-full",
                status === 'ready' ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-brand-500 animate-pulse"
              )} />
            </div>
            
            <div className="space-y-1">
              <p className="text-sm font-medium text-white">
                {status === 'loading-model' && "Carregando E5-Small..."}
                {status === 'processing' && "Lendo Arquivos..."}
                {status === 'indexing' && "Quantizando Vetores..."}
                {status === 'ready' && "TurboQuant Ativo"}
                {status === 'error' && "Falha Crítica"}
                {status === 'idle' && "Iniciando..."}
              </p>
              {status === 'indexing' && (
                <div className="w-full bg-slate-800 h-1 rounded-full mt-2 overflow-hidden">
                  <motion.div 
                    className="bg-brand-500 h-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                  />
                </div>
              )}
            </div>
          </section>

          {/* Upload Area */}
          <section 
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); handleFileUpload(e.dataTransfer.files); }}
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              "group relative overflow-hidden border-2 border-dashed rounded-2xl p-8 transition-all cursor-pointer text-center",
              isDragging ? "border-brand-500 bg-brand-500/5" : "border-slate-800 hover:border-slate-700 hover:bg-slate-900/40",
              (status === 'indexing' || status === 'processing') && "pointer-events-none opacity-50"
            )}
          >
            <input type="file" ref={fileInputRef} className="hidden" accept=".txt,.pdf" multiple onChange={(e) => handleFileUpload(e.target.files)} />
            <AnimatePresence>
              {(status === 'indexing' || status === 'processing') ? (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center gap-3 py-2"
                >
                  <Loader2 className="w-8 h-8 text-brand-400 animate-spin" />
                  <p className="text-[10px] text-brand-400 font-bold uppercase tracking-widest animate-pulse">
                    Processando...
                  </p>
                </motion.div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <Upload className="w-8 h-8 text-slate-500 group-hover:text-brand-400 transition-colors" />
                  <div>
                    <h3 className="text-sm font-bold text-white uppercase tracking-wide">Importar Docs</h3>
                    <p className="text-[10px] text-slate-500 mt-1">PDF ou TXT (Arraste aqui)</p>
                  </div>
                </div>
              )}
            </AnimatePresence>
          </section>

          {/* Stats */}
          <section className="glass-card p-5 space-y-6">
            <div className="flex items-baseline justify-between">
              <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                <Layers className="w-3.5 h-3.5" /> Métricas
              </h2>
              <span className="text-[10px] font-mono text-brand-400 bg-brand-400/10 px-1.5 py-0.5 rounded">Prod-Ready</span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="text-2xl font-bold text-white">{stats.ratio}</div>
                <div className="text-[9px] text-slate-500 uppercase font-bold tracking-tighter">Compressão</div>
              </div>
              <div className="space-y-1 text-right">
                <div className="text-2xl font-bold text-emerald-400">{stats.memoryMB}<span className="text-[10px] ml-0.5">MB</span></div>
                <div className="text-[9px] text-slate-500 uppercase font-bold tracking-tighter">Memória Alocada</div>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-800/50 space-y-3">
              <div className="flex justify-between text-[11px]">
                <span className="text-slate-400">Total Vetores</span>
                <span className="text-white font-mono">{stats.docs}</span>
              </div>
              <div className="flex justify-between text-[11px]">
                <span className="text-slate-400">Centroids IVF</span>
                <span className="text-white font-mono">{stats.clusters}</span>
              </div>
            </div>
          </section>

          {/* Wipe Database Button */}
          <section className="space-y-3">
            <AnimatePresence mode="wait">
              {!showWipeConfirm ? (
                <motion.button
                  key="wipe-btn"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  onClick={() => setShowWipeConfirm(true)}
                  className="w-full py-3 px-4 rounded-xl border border-slate-800 text-slate-500 hover:text-red-400 hover:border-red-400/30 hover:bg-red-400/5 transition-all flex items-center justify-center gap-2 group"
                >
                  <RefreshCw className="w-4 h-4 group-hover:rotate-180 transition-transform duration-500" />
                  <span className="text-[11px] font-bold uppercase tracking-widest">Limpar Base Local</span>
                </motion.button>
              ) : (
                <motion.div
                  key="wipe-confirm"
                  initial={{ scale: 0.95, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.95, opacity: 0 }}
                  className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-center space-y-3"
                >
                  <div className="flex justify-center text-red-400">
                    <AlertTriangle className="w-6 h-6 animate-pulse" />
                  </div>
                  <p className="text-[10px] text-red-200 font-medium leading-tight">
                    Isso apagará permanentemente todos os índices e documentos locais. Continuar?
                  </p>
                  <div className="flex gap-2">
                    <button 
                      onClick={clearCache}
                      className="flex-1 py-1.5 bg-red-600 hover:bg-red-500 text-white text-[10px] font-bold uppercase rounded-lg"
                    >
                      Sim, Apagar
                    </button>
                    <button 
                      onClick={() => setShowWipeConfirm(false)}
                      className="flex-1 py-1.5 bg-slate-800 hover:bg-slate-700 text-white text-[10px] font-bold uppercase rounded-lg"
                    >
                      Cancelar
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        </aside>

        {/* Middle Column: Search & Results */}
        <main className="lg:col-span-6 space-y-8">
          {/* Search Bar */}
          <form onSubmit={handleSearch} className="sticky top-8 z-10">
            <div className="relative group">
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
                placeholder="Pergunte ao seu conhecimento local (PT-BR)..."
                disabled={status !== 'ready'}
                className="w-full glass-input pl-14 pr-32 py-5 text-lg shadow-2xl shadow-black/20"
              />
              <button
                type="submit"
                disabled={status !== 'ready' || !query.trim() || isSearching}
                className="absolute right-3 inset-y-3 px-5 bg-brand-600 hover:bg-brand-500 disabled:bg-slate-800 text-white rounded-xl font-bold text-xs uppercase tracking-widest transition-all hover:scale-105 active:scale-95"
              >
                Buscar
              </button>
            </div>
          </form>

          {/* Results */}
          <div className="space-y-6">
            <AnimatePresence mode="popLayout">
              {results.length > 0 ? (
                results.map((result, idx) => (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ delay: idx * 0.05, type: "spring", damping: 20 }}
                    className="glass-card p-6 relative group"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-brand-500/10 flex items-center justify-center border border-brand-500/20 text-brand-400 font-black text-xs">
                          {idx + 1}
                        </div>
                        <div>
                          <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">
                            {result.metadata?.source || "Documento"}
                          </p>
                          <div className="flex items-center gap-3 mt-1">
                             <div className="flex items-center gap-1">
                               <Database className="w-3 h-3 text-slate-600" />
                               <span className="text-[10px] font-bold text-slate-500 uppercase">Cluster {result.centroid}</span>
                             </div>

                             {/* Semantic Score */}
                             {result.semanticScore !== undefined && result.semanticScore > 0 && (
                               <div className={cn("flex items-center gap-1", result.semanticScore > 0.75 ? "text-emerald-400" : "text-yellow-400")}>
                                 <Zap className="w-3 h-3" />
                                 <span className="text-[10px] font-bold uppercase">Semântica {(result.semanticScore * 100).toFixed(1)}%</span>
                               </div>
                             )}

                             {/* BM25 Score */}
                             {result.bm25Score !== undefined && result.bm25Score > 0 && (
                               <div className="flex items-center gap-1 text-fuchsia-400">
                                 <Terminal className="w-3 h-3" />
                                 <span className="text-[10px] font-bold uppercase">Léxico ({result.bm25Score.toFixed(1)})</span>
                               </div>
                             )}
                          </div>
                        </div>
                      </div>
                      <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                         {result.metadata?.type === 'pdf' && <FileSearch className="w-4 h-4 text-slate-600" />}
                      </div>
                    </div>
                    <p className="text-slate-100 text-sm leading-relaxed font-medium">
                      {result.text}
                    </p>
                  </motion.div>
                ))
              ) : status === 'ready' && !isSearching ? (
                <div className="flex flex-col items-center justify-center py-32 text-center opacity-30 grayscale hover:grayscale-0 transition-all cursor-default">
                  <FileText className="w-16 h-16 text-slate-500 mb-6" />
                  <h3 className="text-xl font-bold text-slate-400 uppercase tracking-widest">Aguardando Consulta</h3>
                  <p className="text-sm text-slate-600 mt-2 max-w-xs font-medium">Use a barra acima para pesquisar nos documentos indexados em 4-bit.</p>
                </div>
              ) : isSearching ? (
                <div className="space-y-6">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="glass-card p-8 animate-pulse-soft">
                      <div className="h-3 bg-slate-800 rounded w-1/4 mb-6" />
                      <div className="space-y-3">
                        <div className="h-3 bg-slate-800 rounded w-full" />
                        <div className="h-3 bg-slate-800 rounded w-4/5" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
            </AnimatePresence>
          </div>
        </main>

        {/* Right Column: Console & Sources */}
        <aside className="lg:col-span-3 space-y-8">
          {/* Active Sources */}
          <section className="glass-card p-5 space-y-6">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest flex items-center justify-between">
              <span>Fontes Ativas</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 bg-slate-800 rounded">{Array.from(new Set(documents.map(d => d.source))).length} font</span>
            </h2>
            
            <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
              {(Array.from(new Set(documents.map(d => d.source))) as string[]).map(source => (
                <div key={source} className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/40 hover:bg-slate-800/60 transition-colors group">
                  <div className="flex items-center gap-2.5 overflow-hidden">
                    {documents.find(d => d.source === source)?.type === 'pdf' ? 
                      <FileSearch className="w-3.5 h-3.5 text-red-400/70" /> : 
                      <FileText className="w-3.5 h-3.5 text-brand-400/70" />
                    }
                    <span className="text-[11px] font-medium text-slate-300 truncate">{source}</span>
                  </div>
                  <button 
                    onClick={() => removeDoc(source)}
                    className="p-1 hover:text-red-400 text-slate-600 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </section>

          {/* Console */}
          <section className="glass-card p-5 bg-black/60">
            <div className="flex items-center justify-between mb-4">
               <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
                 <Terminal className="w-3 h-3" /> Kernel Logs
               </h2>
               <button onClick={clearCache} className="text-[10px] flex items-center gap-1 text-slate-600 hover:text-brand-400 uppercase font-bold tracking-tighter transition-colors">
                  <RefreshCw className="w-2.5 h-2.5" /> Wipe
               </button>
            </div>
            <div className="font-mono text-[9px] space-y-2 p-1">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-brand-500/30">[$]</span>
                  <span className={cn(
                    i === logs.length - 1 ? "text-brand-400" : "text-slate-600"
                  )}>{log}</span>
                </div>
              ))}
              {isSearching && (
                <div className="text-emerald-400 flex items-center gap-1.5 animate-pulse mt-2">
                  <span className="w-1 h-1 rounded-full bg-emerald-400" />
                  Calculando Inner Product não tendencioso...
                </div>
              )}
            </div>
          </section>

          {/* Info Card */}
          <div className="p-5 bg-gradient-to-br from-brand-600/10 to-indigo-600/10 border border-brand-500/10 rounded-2xl relative overflow-hidden group">
             <div className="absolute -right-4 -bottom-4 opacity-5 transform group-hover:scale-110 transition-transform">
                <Info className="w-24 h-24 text-white" />
             </div>
             <p className="text-[11px] text-slate-400 leading-relaxed font-medium">
               Este motor utiliza quantização de <strong>4-bit + 1-bit Residual</strong> para compressão ultra-eficiente sem perder precisão matemática em Português.
             </p>
          </div>
        </aside>

      </div>
    </div>
  );
}
