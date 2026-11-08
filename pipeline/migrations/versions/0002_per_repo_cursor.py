"""Per-repo ingest cursor + the indexes the hot queries need.

pipeline_runs had no repo column, so the incremental cursor
(last_synced_updated_at) was global: ingesting repo B right after repo A
used A's high-water mark as B's `since`, silently skipping every B issue
older than that -- and A's next hourly run could resume from B's cursor.

Existing rows keep repo = NULL. We deliberately do NOT backfill them with
GITHUB_REPO: those rows are exactly the ones that may belong to some
other repo (that's the bug). The cursor query filters on repo = %s, so
legacy rows are simply ignored and each repo's first run after this
migration starts from scratch -- safe, because upsert_issue is
idempotent and content_hash keeps unchanged issues from being re-embedded.

Indexes:
  * pipeline_runs (repo, run_started_at DESC) -- the cursor lookup
  * issues (repo)                             -- per-repo reads / counts
  * issues (updated_at) WHERE embedded = false -- the embedding work
    queue; partial, so it stays tiny no matter how big `issues` grows

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-08
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS repo TEXT;

        CREATE INDEX IF NOT EXISTS ix_pipeline_runs_repo_started
            ON pipeline_runs (repo, run_started_at DESC);

        CREATE INDEX IF NOT EXISTS ix_issues_repo ON issues (repo);

        CREATE INDEX IF NOT EXISTS ix_issues_unembedded
            ON issues (updated_at) WHERE embedded = false;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS ix_issues_unembedded;
        DROP INDEX IF EXISTS ix_issues_repo;
        DROP INDEX IF EXISTS ix_pipeline_runs_repo_started;
        ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS repo;
        """
    )
