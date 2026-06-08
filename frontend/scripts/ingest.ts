import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createClient } from "@supabase/supabase-js";
import { config } from "dotenv";
import { parse } from "node-html-parser";

type SourceEntry = {
  title: string;
  authors_or_organization?: string;
  year?: string;
  url: string;
  download_url?: string;
  local_filename?: string;
  license_or_access_note?: string;
  access_note?: string;
  category: string;
  topics?: string[];
  source_type?: "pdf" | "html";
};

type PageRecord = {
  source: SourceEntry;
  pageNumber: number | null;
  text: string;
};

type ChunkRecord = {
  content: string;
  content_hash: string;
  legacy_content_hash?: string;
  embedding_input: string;
  source_title: string;
  source_url: string;
  page_start: number | null;
  page_end: number | null;
  category: string;
};

type FailedSource = {
  title: string;
  reason: string;
};

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_DIR = path.resolve(__dirname, "..");
const MANIFEST_PATH = path.join(FRONTEND_DIR, "data", "source_manifest.json");
const PDF_DIR = path.join(FRONTEND_DIR, "data", "raw_pdfs");
const HTML_DIR = path.join(FRONTEND_DIR, "data", "raw_html");
const JINA_EMBEDDING_DIMENSIONS = 1024;

config({ path: path.join(FRONTEND_DIR, ".env.local"), quiet: true });
config({ quiet: true });

const CHUNK_SIZE_CHARS = Number(process.env.CHUNK_SIZE_CHARS ?? 4000);
const CHUNK_OVERLAP_CHARS = Number(process.env.CHUNK_OVERLAP_CHARS ?? 800);
const EMBEDDING_BATCH_SIZE = Number(process.env.EMBEDDING_BATCH_SIZE ?? 8);
const EXISTING_HASH_BATCH_SIZE = Number(process.env.EXISTING_HASH_BATCH_SIZE ?? 100);
const INSERT_BATCH_SIZE = Number(process.env.INSERT_BATCH_SIZE ?? 50);
const EMBEDDING_DELAY_MS = Number(process.env.EMBEDDING_DELAY_MS ?? 8000);
const RATE_LIMIT_BACKOFF_MS = Number(process.env.RATE_LIMIT_BACKOFF_MS ?? 70000);

function env(name: string, fallback?: string) {
  const value = process.env[name]?.trim() || fallback;
  if (!value) {
    throw new Error(`${name} is required`);
  }
  return value;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sha256(value: string | Uint8Array) {
  return createHash("sha256").update(value).digest("hex");
}

function cleanText(text: string) {
  return text.replace(/\0/g, " ").replace(/\s+/g, " ").trim();
}

function sanitizeFilename(value: string) {
  return value
    .toLowerCase()
    .replace(/https?:\/\//, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 120);
}

function sourceType(source: SourceEntry) {
  if (source.source_type) {
    return source.source_type;
  }
  return source.url.toLowerCase().endsWith(".html") ? "html" : "pdf";
}

function localFilename(source: SourceEntry) {
  if (source.local_filename) {
    return source.local_filename;
  }
  const extension = sourceType(source) === "html" ? "html" : "pdf";
  return `${sanitizeFilename(source.title)}.${extension}`;
}

function sourceTopics(source: SourceEntry) {
  return source.topics?.filter(Boolean) ?? [];
}

function sourceAccessNote(source: SourceEntry) {
  return source.access_note ?? source.license_or_access_note ?? "Open-access source.";
}

async function readManifest() {
  const raw = await readFile(MANIFEST_PATH, "utf-8");
  return JSON.parse(raw) as SourceEntry[];
}

function metaRefreshUrl(html: string) {
  const match = html.match(/url=['"]?([^'";>]+)['"]?/i);
  return match?.[1]?.replace(/&amp;/g, "&");
}

async function fetchBytes(url: string) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status}`);
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("text/html")) {
    const html = await response.text();
    const refreshUrl = metaRefreshUrl(html);
    if (refreshUrl) {
      return fetchBytes(refreshUrl);
    }
    return new TextEncoder().encode(html);
  }

  return new Uint8Array(await response.arrayBuffer());
}

async function downloadPdf(source: SourceEntry) {
  await mkdir(PDF_DIR, { recursive: true });
  const destination = path.join(PDF_DIR, localFilename(source));

  try {
    await readFile(destination);
    console.log(`Reusing ${localFilename(source)}`);
    return destination;
  } catch {
    // File is missing; download it below.
  }

  console.log(`Downloading ${source.title}`);
  const bytes = await fetchBytes(source.download_url ?? source.url);
  await writeFile(destination, bytes);
  return destination;
}

async function downloadHtml(source: SourceEntry) {
  await mkdir(HTML_DIR, { recursive: true });
  const destination = path.join(HTML_DIR, localFilename(source));

  try {
    await readFile(destination, "utf-8");
    console.log(`Reusing ${localFilename(source)}`);
    return destination;
  } catch {
    // File is missing; download it below.
  }

  console.log(`Downloading ${source.title}`);
  const bytes = await fetchBytes(source.download_url ?? source.url);
  await writeFile(destination, bytes);
  return destination;
}

async function extractPages(pdfPath: string, source: SourceEntry): Promise<PageRecord[]> {
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const data = new Uint8Array(await readFile(pdfPath));
  const document = await pdfjs.getDocument({
    data,
    disableFontFace: true,
    useSystemFonts: true
  }).promise;

  const pages: PageRecord[] = [];
  for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber += 1) {
    const page = await document.getPage(pageNumber);
    const content = await page.getTextContent();
    const text = cleanText(content.items.map((item: any) => item.str || "").join(" "));
    if (text) {
      pages.push({ source, pageNumber, text });
    }
  }
  return pages;
}

async function extractHtmlPages(htmlPath: string, source: SourceEntry): Promise<PageRecord[]> {
  const html = await readFile(htmlPath, "utf-8");
  const root = parse(html);

  root.querySelectorAll("script, style, nav, footer, header, aside, pre, code, .highlight").forEach((node) => {
    node.remove();
  });

  const blocks = root
    .querySelectorAll("h1, h2, h3, h4, p, li, dt, dd, th, td")
    .map((node) => cleanText(node.text))
    .filter((text) => text.length > 40);

  const text = cleanText(blocks.join(" "));
  return text ? [{ source, pageNumber: null, text }] : [];
}

function embeddingInput(source: SourceEntry, content: string) {
  const topics = sourceTopics(source);
  const metadata = [
    `Title: ${source.title}`,
    `Category: ${source.category}`,
    topics.length ? `Topics: ${topics.join(", ")}` : null,
    `Access: ${sourceAccessNote(source)}`
  ].filter(Boolean);

  return `${metadata.join("\n")}\n\n${content}`;
}

function chunkPage(page: PageRecord): ChunkRecord[] {
  const chunks: ChunkRecord[] = [];
  let start = 0;
  let chunkIndex = 0;

  while (start < page.text.length) {
    const end = Math.min(start + CHUNK_SIZE_CHARS, page.text.length);
    const content = page.text.slice(start, end).trim();
    if (content) {
      const contentDigest = sha256(content);
      const pageLabel = page.pageNumber ?? "html";
      const stableId = [
        page.source.url,
        pageLabel,
        chunkIndex,
        contentDigest
      ].join(":");
      const legacyStableId = [
        localFilename(page.source),
        pageLabel,
        chunkIndex,
        contentDigest
      ].join(":");
      chunks.push({
        content,
        content_hash: sha256(stableId),
        legacy_content_hash: sha256(legacyStableId),
        embedding_input: embeddingInput(page.source, content),
        source_title: page.source.title,
        source_url: page.source.url,
        page_start: page.pageNumber,
        page_end: page.pageNumber,
        category: page.source.category
      });
    }

    chunkIndex += 1;
    if (end === page.text.length) {
      break;
    }
    start = Math.max(0, end - CHUNK_OVERLAP_CHARS);
  }

  return chunks;
}

type JinaEmbeddingResponse = {
  data: Array<{
    index: number;
    embedding: number[];
  }>;
};

async function embedTexts(texts: string[]) {
  const provider = process.env.EMBEDDING_PROVIDER?.trim().toLowerCase() || "jina";
  if (provider !== "jina") {
    throw new Error("Only EMBEDDING_PROVIDER=jina is supported by this ingestion script.");
  }

  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      const response = await fetch("https://api.jina.ai/v1/embeddings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${env("JINA_API_KEY")}`
        },
        body: JSON.stringify({
          model: env("JINA_EMBEDDING_MODEL", "jina-embeddings-v3"),
          task: "retrieval.passage",
          dimensions: JINA_EMBEDDING_DIMENSIONS,
          input: texts
        })
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload = (await response.json()) as JinaEmbeddingResponse;
      return payload.data
        .sort((left, right) => left.index - right.index)
        .map((item) => item.embedding);
    } catch (error) {
      if (attempt === 5) {
        throw error;
      }
      const message = error instanceof Error ? error.message : String(error);
      const isRateLimit = /rate[_ -]?limit|RATE_TOKEN_LIMIT/i.test(message);
      const backoffMs = isRateLimit ? RATE_LIMIT_BACKOFF_MS : attempt * attempt * 1000;
      console.warn(`Embedding retry ${attempt} after ${backoffMs}ms`);
      await sleep(backoffMs);
    }
  }

  throw new Error("Embedding failed after retries.");
}

async function existingHashes(
  supabase: ReturnType<typeof createClient>,
  hashes: string[]
) {
  const found = new Set<string>();
  for (let start = 0; start < hashes.length; start += EXISTING_HASH_BATCH_SIZE) {
    const batch = hashes.slice(start, start + EXISTING_HASH_BATCH_SIZE);
    const { data, error } = await supabase
      .from("documents")
      .select("content_hash")
      .in("content_hash", batch);

    if (error) {
      throw new Error(error.message);
    }

    data?.forEach((row) => found.add(row.content_hash));
  }
  return found;
}

async function ingestChunks(chunks: ChunkRecord[]) {
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
  const existing = await existingHashes(
    supabase,
    chunks.flatMap((chunk) => [
      chunk.content_hash,
      ...(chunk.legacy_content_hash ? [chunk.legacy_content_hash] : [])
    ])
  );
  const pending = chunks.filter(
    (chunk) =>
      !existing.has(chunk.content_hash) &&
      !(chunk.legacy_content_hash && existing.has(chunk.legacy_content_hash))
  );
  const skipped = chunks.length - pending.length;
  let inserted = 0;

  console.log(`Chunks total: ${chunks.length}`);
  console.log(`Skipped duplicate chunks: ${skipped}`);
  console.log(`Pending upload: ${pending.length}`);

  for (let start = 0; start < pending.length; start += EMBEDDING_BATCH_SIZE) {
    const batch = pending.slice(start, start + EMBEDDING_BATCH_SIZE);
    const embeddings = await embedTexts(
      batch.map((chunk) => chunk.embedding_input)
    );

    const rows = batch.map((chunk, index) => {
      const { embedding_input, legacy_content_hash, ...row } = chunk;
      return {
        ...row,
        embedding: embeddings[index]
      };
    });

    for (let insertStart = 0; insertStart < rows.length; insertStart += INSERT_BATCH_SIZE) {
      const insertBatch = rows.slice(insertStart, insertStart + INSERT_BATCH_SIZE);
      const { error } = await supabase
        .from("documents")
        .upsert(insertBatch, {
          onConflict: "content_hash",
          ignoreDuplicates: true
        });

      if (error) {
        if (error.message.includes("expected 1536 dimensions")) {
          throw new Error(
            [
              "Supabase documents.embedding is still vector(1536).",
              "Jina embeddings are 1024 dimensions.",
              "Run frontend/supabase/reset_jina_schema.sql in the Supabase SQL editor, then rerun npm run ingest."
            ].join(" ")
          );
        }
        throw new Error(error.message);
      }
    }

    inserted += rows.length;
    console.log(`Uploaded ${Math.min(start + batch.length, pending.length)} / ${pending.length}`);
    await sleep(EMBEDDING_DELAY_MS);
  }

  console.log(`Inserted chunks: ${inserted}`);
}

async function main() {
  const manifest = await readManifest();
  const allChunks: ChunkRecord[] = [];
  const failedSources: FailedSource[] = [];

  for (const source of manifest) {
    try {
      const pages =
        sourceType(source) === "html"
          ? await extractHtmlPages(await downloadHtml(source), source)
          : await extractPages(await downloadPdf(source), source);
      const chunks = pages.flatMap(chunkPage);
      allChunks.push(...chunks);
      console.log(`${source.title}: ${pages.length} pages, ${chunks.length} chunks`);
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unknown error";
      failedSources.push({ title: source.title, reason });
      console.warn(`Skipped ${source.title}: ${reason}`);
    }
  }

  if (failedSources.length > 0) {
    console.warn(`Failed documents: ${failedSources.length}`);
    failedSources.forEach((source) => {
      console.warn(`- ${source.title}: ${source.reason}`);
    });
  } else {
    console.log("Failed documents: 0");
  }

  if (allChunks.length === 0) {
    throw new Error("No chunks were extracted from the source manifest.");
  }

  await ingestChunks(allChunks);
  console.log("Ingestion complete.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
