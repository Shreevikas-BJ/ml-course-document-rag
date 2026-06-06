import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI/ML Knowledge RAG Assistant",
  description: "Grounded AI/ML document assistant with FAISS retrieval and citations"
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
