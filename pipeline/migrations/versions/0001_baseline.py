"""Baseline: the schema that sql/init.sql used to create.

Every statement is IF NOT EXISTS on purpose. This one revision has to
work for two very different databases:

  * a brand-new, empty database  -> creates all tables
  * an EXISTING deployment whose tables were created by the old
    docker-entrypoint init.sql   -> no-op, just records revision 0001

That's what lets the already-running Azure VM adopt migrations with a
plain `python -m pipeline.migrate upgrade head`, no manual `stamp` step
and no risk to existing rows.

Raw SQL instead of op.create_table(): the schema was defined in SQL
from day one, and keeping it byte-for-byte reviewable against the old
init.sql matters more here than alembic autogenerate support (there is
no SQLAlchemy model layer to autogenerate from anyway).

Revision ID: 0001
Revises:
Create Date: 2026-11-08
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
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

        CREATE TABLE IF NOT EXISTS field_mapping_overrides (
            id SERIAL PRIMARY KEY,
            repo TEXT NOT NULL,
            original_field TEXT NOT NULL,
            mapped_field TEXT NOT NULL,
            confidence NUMERIC NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS field_mapping_overrides;
        DROP TABLE IF EXISTS drift_alerts;
        DROP TABLE IF EXISTS schema_snapshots;
        DROP TABLE IF EXISTS pipeline_runs;
        DROP TABLE IF EXISTS quarantine;
        DROP TABLE IF EXISTS issues;
        """
    )
