create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists public.query_cache (
  id uuid primary key default gen_random_uuid(),
  question_hash text unique not null,
  normalized_question text not null,
  answer text not null,
  citations jsonb not null,
  retrieved_chunks jsonb,
  refusal boolean not null default false,
  created_at timestamptz default now()
);

create index if not exists query_cache_question_hash_idx
  on public.query_cache (question_hash);

create index if not exists query_cache_created_at_idx
  on public.query_cache (created_at desc);

create table if not exists public.embedding_cache (
  id uuid primary key default gen_random_uuid(),
  question_hash text unique not null,
  normalized_question text not null,
  embedding vector(1024) not null,
  created_at timestamptz default now()
);

create index if not exists embedding_cache_question_hash_idx
  on public.embedding_cache (question_hash);

create index if not exists embedding_cache_created_at_idx
  on public.embedding_cache (created_at desc);

notify pgrst, 'reload schema';
