import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI/ML Knowledge RAG Assistant",
  description:
    "Grounded AI/ML document assistant with Jina embeddings, Supabase vector search, Groq answers, and citations"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
