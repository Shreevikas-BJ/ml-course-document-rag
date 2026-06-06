import { AskBox } from "../components/AskBox";

export default function HomePage() {
  return (
    <main className="page-shell">
      <section className="hero-band">
        <div className="hero-copy">
          <p className="eyebrow">Portfolio RAG System</p>
          <h1>AI/ML Knowledge RAG Assistant</h1>
        </div>
      </section>

      <AskBox />
    </main>
  );
}
