import { AskBox } from "../components/AskBox";

export default function HomePage() {
  return (
    <main className="page-shell">
      <section className="hero-band">
        <div className="hero-copy">
          <p className="eyebrow">Portfolio RAG System</p>
          <h1>AI/ML Knowledge RAG Assistant</h1>
          <p className="hero-subtitle">
            Grounded in Stanford CS229, Cornell CS4780, ISLR, ESL, scikit-learn
            docs, and modern AI/RAG research papers.
          </p>
          <a
            className="linkedin-link"
            href="https://www.linkedin.com/in/shreevikasbj/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Suggest changes or ask about the architecture on LinkedIn
          </a>
        </div>
      </section>

      <AskBox />
    </main>
  );
}
