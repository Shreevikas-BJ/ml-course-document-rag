import { createClient } from "@supabase/supabase-js";
import { config } from "dotenv";

config({ path: ".env.local", quiet: true });
config({ quiet: true });

type CleanupSummary = {
  deleted_query_cache_count: number;
  deleted_embedding_cache_count: number;
  remaining_query_cache_count: number;
  remaining_embedding_cache_count: number;
};

function env(name: string) {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`${name} is required`);
  }
  return value;
}

function positiveIntegerEnv(name: string, fallback: number) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isFinite(value) || value < 1) {
    throw new Error(`${name} must be a positive number`);
  }
  return Math.floor(value);
}

async function main() {
  const retentionDays = positiveIntegerEnv("CACHE_RETENTION_DAYS", 15);
  const maxRows = positiveIntegerEnv("CACHE_MAX_ROWS", 2000);
  const supabase = createClient(
    env("SUPABASE_URL"),
    env("SUPABASE_SERVICE_ROLE_KEY"),
    {
      auth: {
        persistSession: false,
        autoRefreshToken: false
      }
    }
  );

  const { data, error } = await supabase.rpc("cleanup_rag_cache", {
    retention_days: retentionDays,
    max_rows: maxRows
  });

  if (error) {
    throw new Error(
      `Cache cleanup failed: ${error.message}. Run frontend/supabase/cache_retention.sql in Supabase first.`
    );
  }

  const summary = ((data ?? []) as CleanupSummary[])[0];
  if (!summary) {
    throw new Error("Cache cleanup completed without a summary result.");
  }

  console.log(`Retention days: ${retentionDays}`);
  console.log(`Maximum cached questions: ${maxRows}`);
  console.log(`Deleted query-cache rows: ${summary.deleted_query_cache_count}`);
  console.log(
    `Deleted embedding-cache rows: ${summary.deleted_embedding_cache_count}`
  );
  console.log(`Remaining query-cache rows: ${summary.remaining_query_cache_count}`);
  console.log(
    `Remaining embedding-cache rows: ${summary.remaining_embedding_cache_count}`
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
