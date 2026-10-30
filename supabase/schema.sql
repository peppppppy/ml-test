create table if not exists public.retail_sales (
    source_row bigint primary key,
    price double precision not null check (price >= 0),
    quantity integer not null check (quantity > 0),
    category_name text not null,
    payment_method text not null,
    age integer not null check (age > 0),
    review_score double precision check (review_score between 1 and 5),
    is_returned smallint check (is_returned in (0, 1))
);

alter table public.retail_sales
    add column if not exists is_returned smallint check (is_returned in (0, 1));

create table if not exists public.review_score_predictions (
    source_row bigint primary key,
    price double precision not null,
    quantity integer not null,
    category_name text not null,
    payment_method text not null,
    age integer not null,
    review_score double precision check (review_score between 1 and 5),
    prediction double precision not null check (prediction between 1 and 5),
    predicted_at timestamptz not null default now()
);

alter table public.retail_sales enable row level security;
alter table public.review_score_predictions enable row level security;
