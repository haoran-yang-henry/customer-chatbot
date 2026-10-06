"""Alembic environment. Runs migrations synchronously with the psycopg 3 driver."""

from alembic import context
from sqlalchemy import create_engine, pool

from customer_agent.config import get_settings
from customer_agent.infrastructure.db.models import Base

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    # Tests pass an explicit URL; everything else uses the application settings.
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=_database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
