import { GoogleGenAI } from "@google/genai";

/**
 * Modelos com quota separada na free tier.
 * Se o primeiro esgotar, tentamos o fallback automaticamente.
 */
const MODEL_CHAIN = ["gemini-2.5-flash", "gemini-2.0-flash"] as const;

/**
 * Serviço que encapsula a interação com o Google AI via SDK.
 * Mantemos separado da UI e focado estritamente no domínio de LLM.
 *
 * Estratégia de resiliência:
 * 1. Tenta o modelo primário (gemini-2.5-flash).
 * 2. Se receber 429 (RESOURCE_EXHAUSTED), faz fallback para gemini-2.0-flash.
 * 3. Se ambos falharem, propaga o erro para o hook tratar na UI.
 */
export async function* streamRAGAnswer(
  query: string,
  contextTexts: string[],
  apiKey: string
): AsyncGenerator<string, void, unknown> {
  if (!apiKey || apiKey.trim() === "") {
    throw new Error("Chave de API do Gemini ausente ou inválida.");
  }

  const ai = new GoogleGenAI({ apiKey });
  const contextBlock = contextTexts.map((c, i) => `[Referência ${i + 1}]:\n${c}`).join('\n\n---\n\n');

  const prompt = `Você é o MiniFaiss AI, um assistente RAG altamente preciso.
SUAS INSTRUÇÕES:
1. Baseie sua resposta EXCLUSIVAMENTE nas referências fornecidas no contexto abaixo.
2. Não utilize conhecimento prévio ou externo.
3. Se a informação não estiver no contexto, diga: "Não possuo informações suficientes no contexto para responder."
4. Seja claro, conciso e use formatação Markdown com elegância.

CONTEXTO:
====================
${contextBlock}
====================

PERGUNTA: ${query}`;

  let lastError: Error | null = null;

  for (const model of MODEL_CHAIN) {
    try {
      console.log(`[GeminiService] Tentando modelo: ${model}`);

      const responseStream = await ai.models.generateContentStream({
        model,
        contents: prompt,
        config: {
          temperature: 0.1,
        }
      });

      for await (const chunk of responseStream) {
        if (chunk.text) {
          yield chunk.text;
        }
      }

      return; // Sucesso — encerra o generator
    } catch (error: any) {
      const isQuotaError = error?.message?.includes("429") ||
                           error?.message?.includes("RESOURCE_EXHAUSTED") ||
                           error?.status === 429;

      if (isQuotaError && model !== MODEL_CHAIN[MODEL_CHAIN.length - 1]) {
        console.warn(`[GeminiService] Quota excedida para ${model}. Tentando próximo modelo...`);
        lastError = error;
        continue; // Tenta o próximo modelo
      }

      // Erro não-recuperável ou último modelo da cadeia
      console.error(`[GeminiService] Erro no modelo ${model}:`, error);
      lastError = error;
      break;
    }
  }

  // Extrair mensagem amigável do erro
  const rawMsg = lastError?.message || "Falha de comunicação com a IA.";
  if (rawMsg.includes("429") || rawMsg.includes("RESOURCE_EXHAUSTED")) {
    throw new Error("Limite de requisições gratuitas atingido. Aguarde ~1 minuto ou verifique o billing da sua API Key no Google AI Studio.");
  }

  throw new Error(rawMsg);
}
