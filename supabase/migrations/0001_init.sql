-- DeepShield initial schema
create extension if not exists "pgcrypto";

create table if not exists analyses (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete set null,
  image_path text not null,
  status text not null default 'pending' check (status in ('pending','done','failed')),
  dire_score numeric,
  final_score numeric,
  verdict text check (verdict in ('safe','caution','risk')),
  heatmap_path text,
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists analyses_user_created_idx
  on analyses(user_id, created_at desc);

create index if not exists analyses_status_idx
  on analyses(status);

-- Row-level security: each user sees only their own analyses
alter table analyses enable row level security;

create policy "analyses_owner_select"
  on analyses for select
  using (auth.uid() = user_id);

create policy "analyses_owner_insert"
  on analyses for insert
  with check (auth.uid() = user_id);

create policy "analyses_owner_update"
  on analyses for update
  using (auth.uid() = user_id);

-- Storage bucket (run once in Supabase dashboard or via supabase-js):
--   insert into storage.buckets (id, name, public) values ('deepshield','deepshield', false);
