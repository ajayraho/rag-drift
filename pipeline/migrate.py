"""
Schema migrations entrypoint -- a thin wrapper around alembic that
needs no alembic.ini on disk.

Why a module instead of just the `alembic` CLI: Airflow only mounts
./dags and ./pipeline into its containers, and the API image only
COPYs pipeline/. Keeping the migration scripts inside the pipeline
package (pipeline/migrations) and driving alembic from Python means
`python -m pipeline.migrate upgrade head` works identically in the
Airflow venv, a local shell, or CI -- with no extra files to mount.

Usage:
  python -m pipeline.migrate upgrade head      # apply everything (default)
  python -m pipeline.migrate current           # show applied revision
  python -m pipeline.migrate history
  python -m pipeline.migrate downgrade -1
  python -m pipeline.migrate upgrade head --sql   # print SQL, don't run it
  python -m pipeline.migrate revision -m "add foo"  # new empty revision

The repo-root alembic.ini points at the same directory, so the plain
`alembic ...` CLI also works for developers who prefer it.
"""

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def get_alembic_config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    return cfg


def upgrade(revision: str = "head", sql: bool = False) -> None:
    command.upgrade(get_alembic_config(), revision, sql=sql)


def main(argv: list[str]) -> int:
    cfg = get_alembic_config()
    if not argv:
        argv = ["upgrade", "head"]

    sql = "--sql" in argv
    args = [a for a in argv if a != "--sql"]
    cmd, rest = args[0], args[1:]

    if cmd == "upgrade":
        command.upgrade(cfg, rest[0] if rest else "head", sql=sql)
    elif cmd == "downgrade":
        if not rest:
            print("downgrade needs a target, e.g. -1 or base", file=sys.stderr)
            return 2
        command.downgrade(cfg, rest[0], sql=sql)
    elif cmd == "current":
        command.current(cfg, verbose=True)
    elif cmd == "history":
        command.history(cfg, verbose=True)
    elif cmd == "revision":
        message = rest[rest.index("-m") + 1] if "-m" in rest else None
        command.revision(cfg, message=message)
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
