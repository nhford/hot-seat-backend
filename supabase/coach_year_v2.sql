-- New publish table for scored coach-seasons (avoids schema clashes with legacy coach_year).
-- Run in the Supabase SQL editor before: python -m src.export

create table if not exists public.coach_year_v2 (
  id text not null,
  year integer not null,
  fired integer not null default 0,
  prob double precision,
  tm text,
  age integer,
  round double precision,
  win_pct double precision,
  w_plyf double precision,
  exp integer,
  tenure integer,
  tenure_over_500 integer,
  tenure_w_plyf double precision,
  tenure_coy_share double precision,
  exp_coy_share double precision,
  srs double precision,
  ou double precision,
  gm integer,
  owner integer,
  coy_share double precision,
  coy_rank integer,
  poc integer,
  delta_1yr_win_pct double precision,
  delta_2yr_win_pct double precision,
  delta_3yr_win_pct double precision,
  delta_1yr_plyf integer,
  delta_2yr_plyf integer,
  delta_3yr_plyf integer,
  pred integer,
  name text,
  team text not null,
  win_pct_proj double precision,
  color1 text,
  color2 text,
  wins integer,
  losses integer,
  l_plyf integer,
  ou_line double precision,
  primary key (id, year, team)
);

create index if not exists coach_year_v2_year_idx on public.coach_year_v2 (year);
create index if not exists coach_year_v2_team_idx on public.coach_year_v2 (team);

alter table public.coach_year_v2 enable row level security;

-- Service role bypasses RLS; add anon/authenticated policies only if the frontend reads this table directly.
