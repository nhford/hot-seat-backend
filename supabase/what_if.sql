-- Precomputed What If grid: one firing probability per coach, record, and playoff round.
-- Run once in the Supabase SQL editor before: python -m src.export
--
-- `record` is wins (0–17), the same index the What If select uses.
-- `round` is the playoff result (0 misses the playoffs … 5 wins the Super Bowl).
-- The anon key the site uses can SELECT. Upserts use the service role, which bypasses RLS.

create table if not exists public.what_if (
  id text not null,
  record integer not null,
  round integer not null,
  prob double precision not null,
  primary key (id, record, round)
);

alter table public.what_if enable row level security;

drop policy if exists what_if_anon_select on public.what_if;
create policy what_if_anon_select
  on public.what_if
  for select
  to anon
  using (true);

grant select on table public.what_if to anon;
