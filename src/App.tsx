import { useState, useEffect, useRef, FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useRAGGeneration } from './hooks/useRAGGeneration';
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
  AlertTriangle,
  Sparkles
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
  const [geminiApiKey, setGeminiApiKey] = useState(() => sessionStorage.getItem("gemini_api_key") || "");
  const [groqApiKey, setGroqApiKey] = useState(() => sessionStorage.getItem("groq_api_key") || "");
  
  const rag = useRAGGeneration();
  
  useEffect(() => {
    if (geminiApiKey) {
      sessionStorage.setItem("gemini_api_key", geminiApiKey);
    } else {
      sessionStorage.removeItem("gemini_api_key");
    }
  }, [geminiApiKey]);

  useEffect(() => {
    if (groqApiKey) {
      sessionStorage.setItem("groq_api_key", groqApiKey);
    } else {
      sessionStorage.removeItem("groq_api_key");
    }
  }, [groqApiKey]);
  
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
      
      if (geminiApiKey || groqApiKey) {
        rag.generateAnswer(query, searchResults as Result[], geminiApiKey, groqApiKey);
      }
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
    <div className="min-h-screen bg-surface-50 text-gray-700 selection:bg-brand-100 font-sans">
      <div className="max-w-[1400px] mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Brand & Controls */}
        <aside className="lg:col-span-3 space-y-8">
          <header className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-brand-50 rounded-xl border border-brand-100 shadow-sm shadow-brand-100">
                <Zap className="w-8 h-8 text-brand-500" />
              </div>
              <div>
                <h1 className="text-xl font-black tracking-tighter text-gray-900">MiniFaiss</h1>
                <p className="text-[10px] uppercase tracking-[0.2em] text-brand-600 mt-1 font-mono">ICLR 2026 Engine</p>
              </div>
            </div>
          </header>

          {/* Engine Status */}
          <section className="glass-card p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5" /> Motor
              </span>
              <div className={cn(
                "w-2 h-2 rounded-full",
                status === 'ready' ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]" : "bg-brand-400 animate-pulse"
              )} />
            </div>
            
            <div className="space-y-1">
              <p className="text-sm font-medium text-gray-800">
                {status === 'loading-model' && "Carregando E5-Small..."}
                {status === 'processing' && "Lendo Arquivos..."}
                {status === 'indexing' && "Quantizando Vetores..."}
                {status === 'ready' && "TurboQuant Ativo"}
                {status === 'error' && "Falha Crítica"}
                {status === 'idle' && "Iniciando..."}
              </p>
              {status === 'indexing' && (
                <div className="w-full bg-surface-200 h-1 rounded-full mt-2 overflow-hidden">
                  <motion.div 
                    className="bg-brand-400 h-full"
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
              "group relative overflow-hidden border-2 border-dashed rounded-2xl p-8 transition-all cursor-pointer text-center bg-white/50",
              isDragging ? "border-brand-400 bg-brand-50" : "border-surface-300 hover:border-brand-300 hover:bg-white",
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
                  <p className="text-[10px] text-brand-500 font-bold uppercase tracking-widest animate-pulse">
                    Processando...
                  </p>
                </motion.div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <Upload className="w-8 h-8 text-gray-400 group-hover:text-brand-500 transition-colors" />
                  <div>
                    <h3 className="text-sm font-bold text-gray-700 uppercase tracking-wide">Importar Docs</h3>
                    <p className="text-[10px] text-gray-500 mt-1">PDF ou TXT (Arraste aqui)</p>
                  </div>
                </div>
              )}
            </AnimatePresence>
          </section>

          {/* Stats */}
          <section className="glass-card p-5 space-y-6">
            <div className="flex items-baseline justify-between">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest flex items-center gap-2">
                <Layers className="w-3.5 h-3.5" /> Métricas
              </h2>
              <span className="text-[10px] font-mono text-brand-600 bg-brand-50 px-1.5 py-0.5 rounded border border-brand-100">Prod-Ready</span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <div className="text-2xl font-bold text-gray-800">{stats.ratio}</div>
                <div className="text-[9px] text-gray-500 uppercase font-bold tracking-tighter">Compressão</div>
              </div>
              <div className="space-y-1 text-right">
                <div className="text-2xl font-bold text-emerald-500">{stats.memoryMB}<span className="text-[10px] ml-0.5">MB</span></div>
                <div className="text-[9px] text-gray-500 uppercase font-bold tracking-tighter">Memória Alocada</div>
              </div>
            </div>

            <div className="pt-4 border-t border-surface-200 space-y-3">
              <div className="flex justify-between text-[11px]">
                <span className="text-gray-500">Total Vetores</span>
                <span className="text-gray-800 font-mono font-semibold">{stats.docs}</span>
              </div>
              <div className="flex justify-between text-[11px]">
                <span className="text-gray-500">Centroids IVF</span>
                <span className="text-gray-800 font-mono font-semibold">{stats.clusters}</span>
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
                  className="w-full py-3 px-4 rounded-xl border border-surface-300 text-gray-500 hover:text-red-500 hover:border-red-200 hover:bg-red-50 transition-all flex items-center justify-center gap-2 group bg-white shadow-sm"
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
                  className="p-4 rounded-xl bg-red-50 border border-red-200 text-center space-y-3 shadow-sm"
                >
                  <div className="flex justify-center text-red-500">
                    <AlertTriangle className="w-6 h-6 animate-pulse" />
                  </div>
                  <p className="text-[10px] text-red-700 font-medium leading-tight">
                    Isso apagará permanentemente todos os índices e documentos locais. Continuar?
                  </p>
                  <div className="flex gap-2">
                    <button 
                      onClick={clearCache}
                      className="flex-1 py-1.5 bg-red-500 hover:bg-red-600 text-white text-[10px] font-bold uppercase rounded-lg shadow-sm"
                    >
                      Sim, Apagar
                    </button>
                    <button 
                      onClick={() => setShowWipeConfirm(false)}
                      className="flex-1 py-1.5 bg-white hover:bg-surface-50 text-gray-700 border border-surface-300 text-[10px] font-bold uppercase rounded-lg shadow-sm"
                    >
                      Cancelar
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>

          {/* LLM API Keys */}
          <section className="glass-card p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest flex items-center gap-2">
                <Sparkles className="w-3.5 h-3.5" /> Geração RAG
              </h2>
              {(geminiApiKey || groqApiKey) ? (
                <span className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-wider text-emerald-500 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100">
                  <ShieldCheck className="w-3 h-3" /> {[geminiApiKey && 'Gemini', groqApiKey && 'Groq'].filter(Boolean).join(' + ')}
                </span>
              ) : (
                <span className="text-[9px] font-bold uppercase tracking-wider text-gray-400">Desconectado</span>
              )}
            </div>

            {/* Gemini Key */}
            <div className="space-y-1">
              <label className="text-[9px] text-gray-500 uppercase tracking-wider font-bold">Google Gemini</label>
              <div className="relative">
                <input 
                  type="password"
                  value={geminiApiKey}
                  onChange={e => setGeminiApiKey(e.target.value)}
                  placeholder="Gemini API Key..."
                  className="w-full bg-white border border-surface-300 rounded-lg px-3 py-2 pr-8 text-xs text-gray-800 focus:outline-none focus:ring-2 focus:ring-brand-200 focus:border-brand-300 transition-all placeholder:text-gray-400"
                />
                {geminiApiKey && (
                  <button onClick={() => setGeminiApiKey("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-red-500 transition-colors" title="Limpar">
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>

            {/* Groq Key */}
            <div className="space-y-1">
              <label className="text-[9px] text-gray-500 uppercase tracking-wider font-bold">Groq (Fallback)</label>
              <div className="relative">
                <input 
                  type="password"
                  value={groqApiKey}
                  onChange={e => setGroqApiKey(e.target.value)}
                  placeholder="Groq API Key..."
                  className="w-full bg-white border border-surface-300 rounded-lg px-3 py-2 pr-8 text-xs text-gray-800 focus:outline-none focus:ring-2 focus:ring-brand-200 focus:border-brand-300 transition-all placeholder:text-gray-400"
                />
                {groqApiKey && (
                  <button onClick={() => setGroqApiKey("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-red-500 transition-colors" title="Limpar">
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>

            <div className="flex items-start gap-2 p-2.5 bg-brand-50 rounded-lg border border-brand-100">
              <ShieldCheck className="w-3.5 h-3.5 text-brand-500 mt-0.5 shrink-0" />
              <p className="text-[9px] text-gray-600 leading-relaxed font-medium">
                <strong className="text-brand-600">BYOK</strong> — Chaves em <code className="text-brand-600 font-bold">sessionStorage</code>. Cadeia: Gemini 2.5 → 2.0 → Groq Llama 3.3 70B.
              </p>
            </div>
          </section>
        </aside>

        {/* Middle Column: Search & Results */}
        <main className="lg:col-span-6 space-y-8">
          {/* Search Bar */}
          <form onSubmit={handleSearch} className="sticky top-8 z-10">
            <div className="relative group">
              <div className="absolute inset-y-0 left-5 flex items-center pointer-events-none">
                {isSearching ? (
                  <Loader2 className="w-5 h-5 text-brand-500 animate-spin" />
                ) : (
                  <Search className="w-5 h-5 text-gray-400 group-focus-within:text-brand-500 transition-colors" />
                )}
              </div>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Pergunte ao seu conhecimento local (PT-BR)..."
                disabled={status !== 'ready'}
                className="w-full glass-input pl-14 pr-32 py-5 text-lg shadow-lg shadow-brand-100/50"
              />
              <button
                type="submit"
                disabled={status !== 'ready' || !query.trim() || isSearching}
                className="absolute right-3 inset-y-3 px-5 bg-brand-500 hover:bg-brand-600 disabled:bg-surface-300 disabled:text-gray-400 text-white rounded-xl font-bold text-xs uppercase tracking-widest transition-all hover:scale-105 active:scale-95 shadow-sm"
              >
                Buscar
              </button>
            </div>
          </form>

          {/* Orama-style Chat Answer */}
          <AnimatePresence>
            {rag.state !== 'idle' && (
              <motion.div
                key="rag-answer"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="glass-card overflow-hidden relative"
              >
                <div className="p-6 space-y-4 max-h-[500px] overflow-y-auto custom-scrollbar">
                  {/* User Prompt Bubble */}
                  <div className="flex justify-end">
                    <div className="max-w-[75%] px-4 py-3 bg-gradient-to-r from-brand-100 to-indigo-100 border border-brand-200/50 rounded-2xl rounded-br-md shadow-sm">
                      <p className="text-sm text-gray-800 font-medium">{query}</p>
                    </div>
                  </div>

                  {/* Skeleton Loader */}
                  {rag.state === 'generating' && !rag.answer && (
                    <div className="animate-pulse space-y-3 py-2">
                      <div className="h-3 bg-surface-200 rounded w-3/4" />
                      <div className="h-3 bg-surface-200 rounded w-1/2" />
                      <div className="h-3 bg-surface-200 rounded w-5/6" />
                    </div>
                  )}

                  {/* Sources Strip (horizontal scrolling) */}
                  {results.length > 0 && (
                    <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
                      {results.slice(0, 4).map((r, i) => (
                        <div
                          key={r.id}
                          className="flex-shrink-0 inline-flex items-center gap-2 bg-white border border-surface-300 rounded-lg px-3 py-2 hover:bg-surface-50 transition-colors max-w-[200px] shadow-sm"
                        >
                          <div className={cn(
                            "w-2 h-2 rounded-full shrink-0",
                            i === 0 ? "bg-brand-400" : i === 1 ? "bg-emerald-400" : i === 2 ? "bg-amber-400" : "bg-fuchsia-400"
                          )} />
                          <div className="flex flex-col min-w-0">
                            <span className="text-[10px] font-semibold text-gray-700 truncate">{String(r.metadata?.source || 'Documento')}</span>
                            <span className="text-[9px] text-gray-500 truncate">{r.text.substring(0, 40)}...</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* AI Response */}
                  {rag.answer && (
                    <div className="p-4 bg-brand-50/50 border border-brand-100 rounded-xl shadow-sm">
                      <div className="flex items-center gap-2 mb-3">
                        <Sparkles className="w-4 h-4 text-brand-500" />
                        <span className="text-[11px] font-bold text-brand-600 uppercase tracking-wider">MiniFaiss AI</span>
                        {rag.activeModel && (
                          <span className="text-[9px] font-mono text-brand-600 bg-brand-100 px-1.5 py-0.5 rounded border border-brand-200">{rag.activeModel}</span>
                        )}
                        {rag.state === 'generating' && (
                          <span className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />
                        )}
                      </div>
                      <div className="text-sm text-gray-700 leading-relaxed prose prose-sm max-w-none [&>p]:mb-3 [&>ul]:list-disc [&>ul]:ml-4 [&>ul]:mb-3 [&>ol]:list-decimal [&>ol]:ml-4 [&>ol]:mb-3 [&>strong]:text-brand-700 [&>strong]:font-bold [&>h1]:text-lg [&>h2]:text-base [&>h3]:text-sm">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{rag.answer}</ReactMarkdown>
                      </div>
                    </div>
                  )}

                  {/* Error */}
                  {rag.state === 'error' && (
                    <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-xs flex items-center gap-2 shadow-sm">
                      <AlertTriangle className="w-4 h-4" />
                      {rag.error}
                    </div>
                  )}

                  {/* Actions (Orama-style) */}
                  {rag.answer && rag.state === 'success' && (
                    <div className="flex items-center gap-3 pt-2 border-t border-surface-200">
                      <button
                        onClick={() => navigator.clipboard.writeText(rag.answer)}
                        className="text-[10px] text-gray-500 hover:text-brand-500 transition-colors flex items-center gap-1 uppercase tracking-wider font-bold"
                      >
                        <ArrowRight className="w-3 h-3" /> Copiar
                      </button>
                      <button
                        onClick={() => rag.reset()}
                        className="text-[10px] text-gray-500 hover:text-red-500 transition-colors flex items-center gap-1 uppercase tracking-wider font-bold"
                      >
                        <RefreshCw className="w-3 h-3" /> Limpar
                      </button>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

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
                        <div className="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center border border-brand-100 text-brand-600 font-black text-xs shadow-sm">
                          {idx + 1}
                        </div>
                        <div>
                          <p className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">
                            {String(result.metadata?.source || "Documento")}
                          </p>
                          <div className="flex items-center gap-3 mt-1">
                             <div className="flex items-center gap-1">
                               <Database className="w-3 h-3 text-gray-400" />
                               <span className="text-[10px] font-bold text-gray-500 uppercase">Cluster {result.centroid}</span>
                             </div>

                             {/* Semantic Score */}
                             {result.semanticScore !== undefined && result.semanticScore > 0 && (
                               <div className={cn("flex items-center gap-1", result.semanticScore > 0.75 ? "text-emerald-500" : "text-amber-500")}>
                                 <Zap className="w-3 h-3" />
                                 <span className="text-[10px] font-bold uppercase">Semântica {(result.semanticScore * 100).toFixed(1)}%</span>
                               </div>
                             )}

                             {/* BM25 Score */}
                             {result.bm25Score !== undefined && result.bm25Score > 0 && (
                               <div className="flex items-center gap-1 text-brand-500">
                                 <Terminal className="w-3 h-3" />
                                 <span className="text-[10px] font-bold uppercase">Léxico ({result.bm25Score.toFixed(1)})</span>
                               </div>
                             )}
                          </div>
                        </div>
                      </div>
                      <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                         {result.metadata?.type === 'pdf' && <FileSearch className="w-4 h-4 text-gray-400" />}
                      </div>
                    </div>
                    <p className="text-gray-700 text-sm leading-relaxed font-medium break-words whitespace-pre-wrap">
                      {result.text}
                    </p>
                  </motion.div>
                ))
              ) : status === 'ready' && !isSearching ? (
                <div className="flex flex-col items-center justify-center py-32 text-center opacity-50 hover:opacity-100 transition-all cursor-default">
                  <FileText className="w-16 h-16 text-surface-400 mb-6" />
                  <h3 className="text-xl font-bold text-gray-400 uppercase tracking-widest">Aguardando Consulta</h3>
                  <p className="text-sm text-gray-500 mt-2 max-w-xs font-medium">Use a barra acima para pesquisar nos documentos indexados em 4-bit.</p>
                </div>
              ) : isSearching ? (
                <div className="space-y-6">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="glass-card p-8 animate-pulse">
                      <div className="h-3 bg-surface-200 rounded w-1/4 mb-6" />
                      <div className="space-y-3">
                        <div className="h-3 bg-surface-200 rounded w-full" />
                        <div className="h-3 bg-surface-200 rounded w-4/5" />
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
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest flex items-center justify-between">
              <span>Fontes Ativas</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 bg-surface-200 text-gray-600 rounded">{Array.from(new Set(documents.map(d => d.source))).length} font</span>
            </h2>
            
            <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
              {(Array.from(new Set(documents.map(d => d.source))) as string[]).map(source => (
                <div key={source} className="flex items-center justify-between p-2.5 rounded-xl bg-white border border-surface-200 shadow-sm hover:border-brand-200 transition-colors group">
                  <div className="flex items-center gap-2.5 overflow-hidden">
                    {documents.find(d => d.source === source)?.type === 'pdf' ? 
                      <FileSearch className="w-3.5 h-3.5 text-red-400" /> : 
                      <FileText className="w-3.5 h-3.5 text-brand-500" />
                    }
                    <span className="text-[11px] font-medium text-gray-700 truncate">{source}</span>
                  </div>
                  <button 
                    onClick={() => removeDoc(source)}
                    className="p-1 hover:text-red-500 text-gray-400 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </section>

          {/* Console */}
          <section className="glass-card p-5 bg-surface-800 border-surface-900">
            <div className="flex items-center justify-between mb-4">
               <h2 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2">
                 <Terminal className="w-3 h-3" /> Kernel Logs
               </h2>
               <button onClick={clearCache} className="text-[10px] flex items-center gap-1 text-gray-400 hover:text-brand-500 uppercase font-bold tracking-tighter transition-colors">
                  <RefreshCw className="w-2.5 h-2.5" /> Wipe
               </button>
            </div>
            <div className="font-mono text-[9px] space-y-2 p-1">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-brand-300">[$]</span>
                  <span className={cn(
                    i === logs.length - 1 ? "text-brand-600 font-semibold" : "text-gray-500"
                  )}>{log}</span>
                </div>
              ))}
              {isSearching && (
                <div className="text-emerald-500 flex items-center gap-1.5 animate-pulse mt-2">
                  <span className="w-1 h-1 rounded-full bg-emerald-500" />
                  Calculando Inner Product não tendencioso...
                </div>
              )}
            </div>
          </section>

          {/* Info Card */}
          <div className="p-5 bg-gradient-to-br from-brand-50 to-indigo-50 border border-brand-100 rounded-2xl relative overflow-hidden group shadow-sm">
             <div className="absolute -right-4 -bottom-4 opacity-5 transform group-hover:scale-110 transition-transform">
                <Info className="w-24 h-24 text-brand-500" />
             </div>
             <p className="text-[11px] text-brand-800/80 leading-relaxed font-medium">
               Este motor utiliza quantização de <strong>4-bit + 1-bit Residual</strong> para compressão ultra-eficiente sem perder precisão matemática em Português.
             </p>
          </div>
        </aside>

      </div>
    </div>
  );
}
