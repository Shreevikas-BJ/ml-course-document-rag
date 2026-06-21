create or replace function public.cleanup_rag_cache(
  retention_days integer default 15,
  max_rows integer default 2000
)
returns table (
  deleted_query_cache_count bigint,
  deleted_embedding_cache_count bigint,
  remaining_query_cache_count bigint,
  remaining_embedding_cache_count bigint
)
language plpgsql
security definer
set search_path = public
as $$
declare
  effective_retention_days integer := greatest(coalesce(retention_days, 15), 1);
  effective_max_rows integer := greatest(coalesce(max_rows, 2000), 1);
  affected_rows bigint := 0;
begin
  deleted_query_cache_count := 0;
  deleted_embedding_cache_count := 0;

  delete from public.query_cache
  where created_at < now() - make_interval(days => effective_retention_days);
  get diagnostics affected_rows = row_count;
  deleted_query_cache_count := deleted_query_cache_count + affected_rows;

  delete from public.embedding_cache
  where created_at < now() - make_interval(days => effective_retention_days);
  get diagnostics affected_rows = row_count;
  deleted_embedding_cache_count := deleted_embedding_cache_count + affected_rows;

  with excess_query_rows as (
    select id
    from public.query_cache
    order by created_at desc nulls last, id desc
    offset effective_max_rows
  )
  delete from public.query_cache as query_rows
  using excess_query_rows
  where query_rows.id = excess_query_rows.id;
  get diagnostics affected_rows = row_count;
  deleted_query_cache_count := deleted_query_cache_count + affected_rows;

  delete from public.embedding_cache as embedding_rows
  where not exists (
    select 1
    from public.query_cache as query_rows
    where query_rows.question_hash = embedding_rows.question_hash
  );
  get diagnostics affected_rows = row_count;
  deleted_embedding_cache_count := deleted_embedding_cache_count + affected_rows;

  with excess_embedding_rows as (
    select id
    from public.embedding_cache
    order by created_at desc nulls last, id desc
    offset effective_max_rows
  )
  delete from public.embedding_cache as embedding_rows
  using excess_embedding_rows
  where embedding_rows.id = excess_embedding_rows.id;
  get diagnostics affected_rows = row_count;
  deleted_embedding_cache_count := deleted_embedding_cache_count + affected_rows;

  select count(*) into remaining_query_cache_count
  from public.query_cache;

  select count(*) into remaining_embedding_cache_count
  from public.embedding_cache;

  return next;
end;
$$;

revoke all on function public.cleanup_rag_cache(integer, integer) from public;
grant execute on function public.cleanup_rag_cache(integer, integer) to service_role;

notify pgrst, 'reload schema';
