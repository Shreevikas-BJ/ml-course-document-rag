import { config } from "dotenv";

config({ path: ".env.local", quiet: true });

const API_BASE_URL =
  process.env.API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:3000";
const SEMANTIC_THRESHOLD = Number(
  process.env.SEMANTIC_CACHE_THRESHOLD ?? 0.9
);

type CacheHitType = "none" | "exact" | "semantic";

type AskResponse = {
  answer?: string;
  citations?: unknown[];
  retrieved_chunks?: unknown[];
  refusal?: boolean;
  cache_hit?: boolean;
  cache_hit_type?: CacheHitType;
  embedding_cache_hit?: boolean | "skipped";
  semantic_cache_score?: number;
  matched_cached_question?: string;
  timings?: {
    embedding_ms: number;
    retrieval_ms: number;
    generation_ms: number;
    total_ms: number;
  };
};

async function ask(question: string) {
  const response = await fetch(`${API_BASE_URL}/api/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ question })
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }

  return (await response.json()) as AskResponse;
}

function validateGrounded(label: string, response: AskResponse) {
  const failures: string[] = [];

  if (response.refusal) {
    failures.push("returned refusal");
  }
  if (!response.answer) {
    failures.push("missing answer");
  }
  if (!response.citations?.length) {
    failures.push("missing citations");
  }
  if (!response.retrieved_chunks?.length) {
    failures.push("missing retrieved chunks");
  }
  if (!response.timings) {
    failures.push("missing timings");
  }

  if (failures.length) {
    throw new Error(`${label} failed: ${failures.join("; ")}`);
  }
}

async function testExactCache() {
  const question = "What is linear regression?";
  await ask(question);
  const second = await ask(question);
  validateGrounded("exact cache", second);

  if (!second.cache_hit || second.cache_hit_type !== "exact") {
    throw new Error(
      `Expected exact cache hit, got cache_hit=${Boolean(second.cache_hit)} type=${second.cache_hit_type ?? "missing"}`
    );
  }

  if (
    second.embedding_cache_hit !== "skipped" ||
    second.timings?.embedding_ms !== 0 ||
    second.timings?.retrieval_ms !== 0 ||
    second.timings?.generation_ms !== 0
  ) {
    throw new Error("Exact cache hit did not skip embedding, retrieval, and Groq.");
  }

  console.log(
    `PASS | exact cache | type=${second.cache_hit_type} total_ms=${second.timings?.total_ms ?? "n/a"}`
  );
}

async function testSemanticCache() {
  const original = "Explain the bias-variance tradeoff.";
  const paraphrase = "tell me the tradeoff of bias-variance.";

  await ask(original);
  const second = await ask(paraphrase);
  validateGrounded("semantic cache", second);

  if (!second.cache_hit || second.cache_hit_type !== "semantic") {
    throw new Error(
      `Expected semantic cache hit, got cache_hit=${Boolean(second.cache_hit)} type=${second.cache_hit_type ?? "missing"}`
    );
  }

  if ((second.semantic_cache_score ?? 0) < SEMANTIC_THRESHOLD) {
    throw new Error(
      `Semantic cache score ${second.semantic_cache_score ?? "missing"} is below ${SEMANTIC_THRESHOLD}`
    );
  }

  if (!second.matched_cached_question) {
    throw new Error("Semantic cache response did not identify the matched question.");
  }

  if (
    second.timings?.retrieval_ms !== 0 ||
    second.timings?.generation_ms !== 0
  ) {
    throw new Error("Semantic cache hit did not skip retrieval and Groq.");
  }

  console.log(
    [
      "PASS | semantic cache",
      `type=${second.cache_hit_type}`,
      `score=${second.semantic_cache_score?.toFixed(3) ?? "n/a"}`,
      `embedding_cache=${second.embedding_cache_hit ?? "miss"}`,
      `total_ms=${second.timings?.total_ms ?? "n/a"}`
    ].join(" | ")
  );
}

async function main() {
  await testExactCache();
  await testSemanticCache();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
