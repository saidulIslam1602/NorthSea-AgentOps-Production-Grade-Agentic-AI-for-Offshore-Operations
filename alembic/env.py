"""Alembic environment configuration.

Reads DATABASE_URL from the application Settings so that the same
environment variables used by the API drive migrations — no duplication.

Run migrations:
    alembic upgrade head          # apply all
    alembic downgrade -1          # roll back one
    agentops migrate              # via CLI wrapper
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Alembic Config object (gives access to .ini values)
config = context.config

# Wire Python logging to alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Pull DATABASE_URL from the environment / pydantic Settings so we don't
# need to hard-code it in alembic.ini.
_env_url = os.environ.get("DATABASE_URL")
if _env_url:
    # Alembic needs a sync driver; strip the async +psycopg suffix.
    _sync_url = _env_url.replace("+psycopg", "")
    config.set_main_option("sqlalchemy.url", _sync_url)

target_metadata = None  # We use raw SQL in migration scripts, not ORM metadata.


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL script)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
