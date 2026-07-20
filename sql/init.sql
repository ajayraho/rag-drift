CREATE TABLE IF NOT EXISTS issues (
    id BIGINT PRIMARY KEY,
    repo TEXT NOT NULL,
    number INT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    state TEXT NOT NULL,
    labels TEXT[],
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    author TEXT,
    comments_count INT DEFAULT 0,
    ingested_at TIMESTAMPTZ DEFAULT now(),
    embedded BOOLEAN DEFAULT false,
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quarantine (
    id SERIAL PRIMARY KEY,
    repo TEXT,
    raw_payload JSONB NOT NULL,
    error_reason TEXT NOT NULL,
    quarantined_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_started_at TIMESTAMPTZ DEFAULT now(),
    run_finished_at TIMESTAMPTZ,
    fetched_count INT DEFAULT 0,
    valid_count INT DEFAULT 0,
    quarantined_count INT DEFAULT 0,
    skipped_count INT DEFAULT 0,
    last_synced_updated_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS schema_snapshots (
    id SERIAL PRIMARY KEY,
    repo TEXT NOT NULL,
    fingerprint JSONB NOT NULL,
    captured_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS drift_alerts (
    id SERIAL PRIMARY KEY,
    repo TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT now()
);