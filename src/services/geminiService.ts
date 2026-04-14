import { GoogleGenAI } from "@google/genai";

/**
 * Serviço que encapsula a interação com o Google AI via SDK.
 * Mantemos separado da UI e focado estritamente no domínio de LLM.
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

  try {
    const responseStream = await ai.models.generateContentStream({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        temperature: 0.1, // Minimizar alucinações (Greedy decoding preference)
      }
    });

    for await (const chunk of responseStream) {
      if (chunk.text) {
        yield chunk.text;
      }
    }
  } catch (error: any) {
    console.error("[GeminiService] Ocorreu um erro no streaming:", error);
    throw new Error(error.message || "Falha de comunicação com a IA.");
  }
}
