import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createClient } from "@supabase/supabase-js";
import { config } from "dotenv";

type SourceEntry = {
  title: string;
  authors_or_organization: string;
  year: string;
  url: string;
  local_filename: string;
  license_or_access_note: string;
  category: string;
};

type PageRecord = {
  source: SourceEntry;
  pageNumber: number;
  text: string;
};

type ChunkRecord = {
  content: string;
  content_hash: string;
  source_title: string;
  source_url: string;
  page_start: number;
  page_end: number;
  category: string;
};

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_DIR = path.resolve(__dirname, "..");
const MANIFEST_PATH = path.join(FRONTEND_DIR, "data", "source_manifest.json");
const PDF_DIR = path.join(FRONTEND_DIR, "data", "raw_pdfs");
const JINA_EMBEDDING_DIMENSIONS = 1024;

config({ path: path.join(FRONTEND_DIR, ".env.local") });
config();

const CHUNK_SIZE_CHARS = Number(process.env.CHUNK_SIZE_CHARS ?? 4000);
const CHUNK_OVERLAP_CHARS = Number(process.env.CHUNK_OVERLAP_CHARS ?? 800);
const EMBEDDING_BATCH_SIZE = Number(process.env.EMBEDDING_BATCH_SIZE ?? 24);
const INSERT_BATCH_SIZE = Number(process.env.INSERT_BATCH_SIZE ?? 50);
const EMBEDDING_DELAY_MS = Number(process.env.EMBEDDING_DELAY_MS ?? 500);

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

async function readManifest() {
  const raw = await readFile(MANIFEST_PATH, "utf-8");
  return JSON.parse(raw) as SourceEntry[];
}

async function downloadPdf(source: SourceEntry) {
  await mkdir(PDF_DIR, { recursive: true });
  const destination = path.join(PDF_DIR, source.local_filename);

  try {
    await readFile(destination);
    console.log(`Reusing ${source.local_filename}`);
    return destination;
  } catch {
    // File is missing; download it below.
  }

  console.log(`Downloading ${source.title}`);
  const response = await fetch(source.url);
  if (!response.ok) {
    throw new Error(`Failed to download ${source.url}: ${response.status}`);
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
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

function chunkPage(page: PageRecord): ChunkRecord[] {
  const chunks: ChunkRecord[] = [];
  let start = 0;
  let chunkIndex = 0;

  while (start < page.text.length) {
    const end = Math.min(start + CHUNK_SIZE_CHARS, page.text.length);
    const content = page.text.slice(start, end).trim();
    if (content) {
      const stableId = [
        page.source.local_filename,
        page.pageNumber,
        chunkIndex,
        sha256(content)
      ].join(":");
      chunks.push({
        content,
        content_hash: sha256(stableId),
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
      const backoffMs = attempt * attempt * 1000;
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
  for (let start = 0; start < hashes.length; start += 500) {
    const batch = hashes.slice(start, start + 500);
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
    chunks.map((chunk) => chunk.content_hash)
  );
  const pending = chunks.filter((chunk) => !existing.has(chunk.content_hash));

  console.log(`Chunks total: ${chunks.length}`);
  console.log(`Already in Supabase: ${existing.size}`);
  console.log(`Pending upload: ${pending.length}`);

  for (let start = 0; start < pending.length; start += EMBEDDING_BATCH_SIZE) {
    const batch = pending.slice(start, start + EMBEDDING_BATCH_SIZE);
    const embeddings = await embedTexts(
      batch.map((chunk) => chunk.content)
    );

    const rows = batch.map((chunk, index) => ({
      ...chunk,
      embedding: embeddings[index]
    }));

    for (let insertStart = 0; insertStart < rows.length; insertStart += INSERT_BATCH_SIZE) {
      const insertBatch = rows.slice(insertStart, insertStart + INSERT_BATCH_SIZE);
      const { error } = await supabase
        .from("documents")
        .upsert(insertBatch, {
          onConflict: "content_hash",
          ignoreDuplicates: true
        });

      if (error) {
        throw new Error(error.message);
      }
    }

    console.log(`Uploaded ${Math.min(start + batch.length, pending.length)} / ${pending.length}`);
    await sleep(EMBEDDING_DELAY_MS);
  }
}

async function main() {
  const manifest = await readManifest();
  const allChunks: ChunkRecord[] = [];

  for (const source of manifest) {
    const pdfPath = await downloadPdf(source);
    const pages = await extractPages(pdfPath, source);
    const chunks = pages.flatMap(chunkPage);
    allChunks.push(...chunks);
    console.log(`${source.title}: ${pages.length} pages, ${chunks.length} chunks`);
  }

  await ingestChunks(allChunks);
  console.log("Ingestion complete.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
