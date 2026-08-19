import "./globals.css";

export const metadata = {
  title: "Voice-RAG | MSMARCO-XI Voice Assistant",
  description: "Ultra-fast voice-enabled Retrieval-Augmented Generation powered by Sarvam STT, BGE-small, Qdrant, BM25, and Llama 3.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
