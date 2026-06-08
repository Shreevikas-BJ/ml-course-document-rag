const API_BASE_URL =
  process.env.API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:3000";
const QUESTION = process.env.CACHE_TEST_QUESTION ?? "What is linear regression?";

type AskResponse = {
  answer?: string;
  citations?: unknown[];
  retrieved_chunks?: unknown[];
  refusal?: boolean;
  cache_hit?: boolean;
  embedding_cache_hit?: boolean;
  timings?: {
    embedding_ms: number;
    retrieval_ms: number;
    generation_ms: number;
    total_ms: number;
  };
};

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

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

function validateResponse(label: string, response: AskResponse) {
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

async function main() {
  const first = await ask(QUESTION);
  validateResponse("first request", first);

  await sleep(750);

  const second = await ask(QUESTION);
  validateResponse("second request", second);

  if (!second.cache_hit) {
    throw new Error(
      "Expected second response to have cache_hit=true. Run frontend/supabase/performance_cache.sql in Supabase and try again."
    );
  }

  if (
    second.timings &&
    (second.timings.embedding_ms !== 0 ||
      second.timings.retrieval_ms !== 0 ||
      second.timings.generation_ms !== 0)
  ) {
    throw new Error(
      "Expected cached response to skip embedding, retrieval, and generation timings."
    );
  }

  console.log(
    `PASS | cache test | first_cache_hit=${Boolean(first.cache_hit)} second_cache_hit=${Boolean(second.cache_hit)} total_ms=${second.timings?.total_ms ?? "n/a"}`
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
