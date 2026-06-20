create extension if not exists vector;

alter table public.query_cache
  add column if not exists question_embedding vector(1024),
  add column if not exists cache_source text not null default 'rag',
  add column if not exists updated_at timestamptz not null default now();

update public.query_cache as query_rows
set
  question_embedding = embedding_rows.embedding,
  updated_at = now()
from public.embedding_cache as embedding_rows
where query_rows.question_hash = embedding_rows.question_hash
  and query_rows.question_embedding is null;

create index if not exists query_cache_embedding_idx
  on public.query_cache
  using hnsw (question_embedding vector_cosine_ops)
  where question_embedding is not null;

create or replace function public.match_query_cache(
  query_embedding vector(1024),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  normalized_question text,
  answer text,
  citations jsonb,
  retrieved_chunks jsonb,
  refusal boolean,
  similarity float
)
language sql
stable
as $$
  select
    query_rows.id,
    query_rows.normalized_question,
    query_rows.answer,
    query_rows.citations,
    query_rows.retrieved_chunks,
    query_rows.refusal,
    1 - (query_rows.question_embedding <=> query_embedding) as similarity
  from public.query_cache as query_rows
  where query_rows.question_embedding is not null
    and 1 - (query_rows.question_embedding <=> query_embedding) >= match_threshold
  order by query_rows.question_embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

notify pgrst, 'reload schema';
