import { useState, useCallback, useRef } from "react";
import { streamRAGAnswer } from "../services/geminiService";
import { Result } from "../types";

export type RAGGenerationState = 'idle' | 'generating' | 'success' | 'error';

export function useRAGGeneration() {
  const [state, setState] = useState<RAGGenerationState>('idle');
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState<string | null>(null);
  const isCanceledRef = useRef(false);

  const generateAnswer = useCallback(async (query: string, results: Result[], apiKey: string) => {
    // Reset state
    setState('generating');
    setAnswer("");
    setError(null);
    isCanceledRef.current = false;

    if (results.length === 0) {
      setAnswer("Não foram encontrados documentos relevantes na base local para gerar uma resposta.");
      setState('success');
      return;
    }

    try {
      const texts = results.map(r => r.text);
      const stream = streamRAGAnswer(query, texts, apiKey);

      for await (const chunk of stream) {
        if (isCanceledRef.current) break;
        setAnswer(prev => prev + chunk);
      }
      
      setState('success');
    } catch (err: any) {
      if (!isCanceledRef.current) {
        setState('error');
        setError(err.message || "Falha na geração");
      }
    }
  }, []);

  const cancelGeneration = useCallback(() => {
    isCanceledRef.current = true;
    if (state === 'generating') {
      setState('idle');
    }
  }, [state]);

  const reset = useCallback(() => {
    setState('idle');
    setAnswer("");
    setError(null);
  }, []);

  return {
    state,
    answer,
    error,
    generateAnswer,
    cancelGeneration,
    reset
  };
}
