import * as pdfjs from "pdfjs-dist";
// @ts-ignore
import pdfWorker from "pdfjs-dist/build/pdf.worker.mjs?url";

// Set worker source for pdfjs
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorker;

export async function extractTextFromPDF(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;
  let fullText = "";

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const strings = content.items.map((item: unknown) => (item as { str: string }).str);
    fullText += strings.join(" ") + "\n\n";
  }

  return fullText;
}
