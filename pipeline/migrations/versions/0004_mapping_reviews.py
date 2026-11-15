"""Review queue for self-heal mappings without enough evidence to auto-apply.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-16
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS mapping_reviews (
            id SERIAL PRIMARY KEY,
            repo TEXT NOT NULL,
            original_field TEXT NOT NULL,
            candidate_field TEXT NOT NULL,
            evidence NUMERIC NOT NULL,
            matched INT NOT NULL,
            overlap INT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected
            created_at TIMESTAMPTZ DEFAULT now(),
            decided_at TIMESTAMPTZ
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_mapping_reviews_pending
            ON mapping_reviews (repo, original_field, candidate_field) WHERE status = 'pending';
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS mapping_reviews;")
