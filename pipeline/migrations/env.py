"""
Alembic environment for the pipeline's own Postgres schema.

Two deliberate choices here:

1. The connection URL is built from pipeline.db.get_conn_params(), the
   same POSTGRES_* env vars every other pipeline entrypoint uses -- so
   there is exactly one place that decides where the database lives, and
   alembic.ini carries no credentials.

2. version_table is "rag_alembic_version", NOT alembic's default
   "alembic_version". Airflow keeps its metadata in this SAME database
   (see docker-compose: both point at rag_pipeline) and Airflow itself
   uses alembic with the default table name. Sharing that table would
   make each tool think the other's revision id is its own -- `airflow
   db init` / `airflow db migrate` would fail or, worse, skip
   migrations. A separate version table keeps the two histories apart
   until Airflow gets its own database (roadmap item 6.2).
"""

from alembic import context
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import URL

from pipeline.db import get_conn_params

VERSION_TABLE = "rag_alembic_version"


def _database_url() -> URL:
    p = get_conn_params()
    # URL.create (not an f-string) so passwords with @, /, : etc. are
    # escaped correctly instead of silently producing a wrong host.
    return URL.create(
        "postgresql+psycopg2",
        username=p["user"],
        password=p["password"],
        host=p["host"],
        port=int(p["port"]),
        database=p["dbname"],
    )


def run_migrations_offline() -> None:
    """`--sql` mode: emit the SQL instead of running it (for review / DBAs)."""
    context.configure(
        url=_database_url().render_as_string(hide_password=False),
        literal_binds=True,
        version_table=VERSION_TABLE,
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            version_table=VERSION_TABLE,
            # Each revision commits on its own: if 0003 fails, 0001-0002
            # stay applied and recorded, and a re-run resumes at 0003.
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
