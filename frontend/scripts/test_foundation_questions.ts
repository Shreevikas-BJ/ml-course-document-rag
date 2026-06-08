const API_BASE_URL = process.env.API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:3000";
const ASK_DELAY_MS = Number(process.env.ASK_DELAY_MS ?? 12000);
const REQUEST_RETRIES = Number(process.env.REQUEST_RETRIES ?? 4);
const REFUSAL_MESSAGE =
  "Not enough information in the indexed AI/ML documents to answer confidently.";

type AskResponse = {
  answer?: string;
  citations?: unknown[];
  retrieved_chunks?: unknown[];
  similarity_scores?: number[];
  refusal?: boolean;
  best_score?: number | null;
};

const questions = [
  "What is linear regression?",
  "Explain logistic regression.",
  "What is gradient boosting?",
  "What is the difference between bagging and boosting?",
  "What is random forest?",
  "What is decision tree impurity?",
  "What is regularization?",
  "What is ridge regression?",
  "What is lasso regression?",
  "What is PCA?",
  "What is k-means clustering?",
  "What is cross-validation?",
  "What is the bias-variance tradeoff?",
  "What is overfitting?",
  "What is precision and recall?"
];

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function ask(question: string) {
  for (let attempt = 1; attempt <= REQUEST_RETRIES; attempt += 1) {
    const response = await fetch(`${API_BASE_URL}/api/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question })
    });

    if (response.ok) {
      return (await response.json()) as AskResponse;
    }

    const errorText = await response.text();
    const isRateLimit = /rate[_ -]?limit/i.test(errorText);
    if (!isRateLimit || attempt === REQUEST_RETRIES) {
      throw new Error(`${response.status} ${errorText}`);
    }

    const backoffMs = ASK_DELAY_MS * attempt;
    console.warn(`Rate limit retry ${attempt} for "${question}" after ${backoffMs}ms`);
    await sleep(backoffMs);
  }

  throw new Error("Request failed after retries.");
}

function validate(question: string, response: AskResponse) {
  const citations = response.citations ?? [];
  const chunks = response.retrieved_chunks ?? [];
  const scores = response.similarity_scores ?? [];
  const failures: string[] = [];

  if (response.refusal || response.answer === REFUSAL_MESSAGE || response.answer?.startsWith("Not enough information")) {
    failures.push("returned refusal");
  }
  if (chunks.length < 1) {
    failures.push("no retrieved chunk passed threshold");
  }
  if (citations.length < 1) {
    failures.push("no citations returned");
  }
  if (scores.length < 1) {
    failures.push("no similarity scores returned");
  }

  return {
    question,
    ok: failures.length === 0,
    failures,
    citations: citations.length,
    chunks: chunks.length,
    best_score: response.best_score ?? null
  };
}

async function main() {
  const results = [];

  for (const question of questions) {
    const response = await ask(question);
    const result = validate(question, response);
    results.push(result);
    const status = result.ok ? "PASS" : "FAIL";
    console.log(
      `${status} | ${question} | citations=${result.citations} chunks=${result.chunks} best_score=${result.best_score ?? "n/a"}`
    );
    if (!result.ok) {
      console.log(`  ${result.failures.join("; ")}`);
    }
    await sleep(ASK_DELAY_MS);
  }

  const failures = results.filter((result) => !result.ok);
  console.log(`Foundation question tests: ${results.length - failures.length}/${results.length} passed`);

  if (failures.length > 0) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
