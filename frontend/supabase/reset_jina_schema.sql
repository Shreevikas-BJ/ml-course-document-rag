-- Destructive reset for migrating an empty/old OpenAI embedding table to Jina.
-- Run this in the Supabase SQL editor only when you are ready to delete
-- existing rows from public.documents and rebuild them with `npm run ingest`.

drop function if exists public.match_documents(vector, float, int);
drop table if exists public.documents;

create extension if not exists vector;
create extension if not exists pgcrypto;

create table public.documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  content_hash text not null unique,
  source_title text not null,
  source_url text,
  page_start int,
  page_end int,
  category text,
  embedding vector(1024) not null,
  created_at timestamptz not null default now()
);

create index documents_embedding_idx
on public.documents
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

create index documents_source_title_idx
on public.documents (source_title);

create or replace function public.match_documents(
  query_embedding vector(1024),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  content text,
  source_title text,
  source_url text,
  page_start int,
  page_end int,
  category text,
  similarity float
)
language sql
stable
as $$
  select
    documents.id,
    documents.content,
    documents.source_title,
    documents.source_url,
    documents.page_start,
    documents.page_end,
    documents.category,
    1 - (documents.embedding <=> query_embedding) as similarity
  from public.documents
  where 1 - (documents.embedding <=> query_embedding) >= match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;
