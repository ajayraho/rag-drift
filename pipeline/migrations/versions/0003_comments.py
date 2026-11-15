"""Issue comments, for chunked embedding of whole threads.

Most answers to an issue live in its comments, which were never ingested.
Also resets issues.embedded: the index moves to a new chunked collection
(see embed.py), so every issue is re-embedded once.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-15
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id BIGINT PRIMARY KEY,
            repo TEXT NOT NULL,
            issue_number INT NOT NULL,
            author TEXT,
            is_bot BOOLEAN NOT NULL DEFAULT false,
            body TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            ingested_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS ix_comments_repo_issue ON comments (repo, issue_number);
        CREATE INDEX IF NOT EXISTS ix_comments_repo_updated ON comments (repo, updated_at);
        UPDATE issues SET embedded = false;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS comments;")
