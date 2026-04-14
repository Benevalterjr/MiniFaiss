import { GoogleGenAI } from "@google/genai";

/**
 * Cadeia de modelos com fallback automático.
 * Gemini 2.5 Flash → Gemini 2.0 Flash → Groq (Llama 3.3 70B)
 */

type ProviderConfig = {
  provider: 'gemini' | 'groq';
  model: string;
};

const MODEL_CHAIN: ProviderConfig[] = [
  { provider: 'gemini', model: 'gemini-2.5-flash' },
  { provider: 'gemini', model: 'gemini-2.0-flash' },
  { provider: 'groq',   model: 'llama-3.3-70b-versatile' },
];

function buildSystemPrompt(contextTexts: string[], query: string): string {
  const contextBlock = contextTexts.map((c, i) => `[Referência ${i + 1}]:\n${c}`).join('\n\n---\n\n');

  return `Você é o MiniFaiss AI, um assistente de análise documental preciso e direto.

REGRAS:
1. Responda EXCLUSIVAMENTE com base nas referências fornecidas abaixo.
2. Interprete os dados com inteligência: sinônimos, variações e termos equivalentes DEVEM ser tratados como a mesma coisa. Não questione se um termo equivale a outro — afirme com confiança.
3. Quando os dados forem semi-estruturados (listas, campos separados por vírgulas, códigos), extraia e organize as informações de forma clara.
4. Seja direto, assertivo e objetivo. Nunca diga "não está claro se..." quando a correspondência for óbvia. Use tabelas Markdown quando houver múltiplos itens.
5. Se realmente não houver NENHUMA informação relevante, diga apenas: "Sem dados suficientes no contexto."
6. Responda em Português do Brasil.

CONTEXTO:
====================
${contextBlock}
====================

PERGUNTA: ${query}`;
}

/** Stream via Google GenAI SDK */
async function* streamGemini(
  apiKey: string, model: string, prompt: string
): AsyncGenerator<string, void, unknown> {
  const ai = new GoogleGenAI({ apiKey });
  const responseStream = await ai.models.generateContentStream({
    model,
    contents: prompt,
    config: { temperature: 0.1 }
  });

  for await (const chunk of responseStream) {
    if (chunk.text) yield chunk.text;
  }
}

/** Stream via Groq REST API (OpenAI-compatible) */
async function* streamGroq(
  apiKey: string, model: string, prompt: string
): AsyncGenerator<string, void, unknown> {
  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: "Você é o MiniFaiss AI, um assistente RAG preciso. Responda exclusivamente com base no contexto fornecido pelo usuário. Use Markdown." },
        { role: "user", content: prompt }
      ],
      temperature: 0.1,
      stream: true,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Groq ${response.status}: ${errorBody}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("Groq: stream indisponível.");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ") || trimmed === "data: [DONE]") continue;

      try {
        const json = JSON.parse(trimmed.slice(6));
        const delta = json.choices?.[0]?.delta?.content;
        if (delta) yield delta;
      } catch {
        // Ignore malformed chunks
      }
    }
  }
}

function isQuotaError(error: any): boolean {
  const msg = error?.message || "";
  return msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED") || error?.status === 429;
}

/**
 * Serviço principal de streaming RAG com fallback automático.
 *
 * @param geminiApiKey - Chave do Google AI Studio
 * @param groqApiKey - Chave do Groq (opcional, usado como último fallback)
 * @param onModelSelected - Callback chamado quando um modelo é selecionado
 */
export async function* streamRAGAnswer(
  query: string,
  contextTexts: string[],
  geminiApiKey: string,
  groqApiKey?: string,
  onModelSelected?: (model: string) => void,
): AsyncGenerator<string, void, unknown> {
  if (!geminiApiKey && !groqApiKey) {
    throw new Error("Nenhuma chave de API configurada. Insira uma chave Gemini ou Groq.");
  }

  const prompt = buildSystemPrompt(contextTexts, query);
  let lastError: Error | null = null;

  for (const config of MODEL_CHAIN) {
    // Pular Gemini se não tiver key
    if (config.provider === 'gemini' && !geminiApiKey) continue;
    // Pular Groq se não tiver key
    if (config.provider === 'groq' && !groqApiKey) continue;

    try {
      console.log(`[LLMService] Tentando: ${config.provider}/${config.model}`);
      onModelSelected?.(`${config.provider}/${config.model}`);

      const stream = config.provider === 'gemini'
        ? streamGemini(geminiApiKey, config.model, prompt)
        : streamGroq(groqApiKey!, config.model, prompt);

      for await (const chunk of stream) {
        yield chunk;
      }

      return; // Sucesso
    } catch (error: any) {
      lastError = error;

      if (isQuotaError(error)) {
        console.warn(`[LLMService] Quota excedida para ${config.provider}/${config.model}. Tentando próximo...`);
        continue;
      }

      // Erro não-recuperável — tenta próximo modelo mesmo assim
      console.error(`[LLMService] Erro em ${config.provider}/${config.model}:`, error);
      continue;
    }
  }

  // Todos falharam
  const rawMsg = lastError?.message || "Falha de comunicação com a IA.";
  if (rawMsg.includes("429") || rawMsg.includes("RESOURCE_EXHAUSTED")) {
    throw new Error("Limite de requisições atingido em todos os provedores. Aguarde alguns minutos.");
  }

  throw new Error(rawMsg);
}
