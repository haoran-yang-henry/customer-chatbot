"""``customer-agent-seed``: load ``data/seed`` into the mock order system (idempotent)."""

import asyncio
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from customer_agent.bootstrap import create_engine
from customer_agent.config import get_settings
from customer_agent.infrastructure.db.order_repository import seed_orders

SEED_DIR = Path("data/seed")


async def _run() -> tuple[int, int]:
    engine = create_engine(get_settings())
    try:
        customers = json.loads((SEED_DIR / "customers.json").read_text(encoding="utf-8"))
        orders = json.loads((SEED_DIR / "orders.json").read_text(encoding="utf-8"))
        return await seed_orders(async_sessionmaker(engine), customers, orders)
    finally:
        await engine.dispose()


def main() -> int:
    customers, orders = asyncio.run(_run())
    print(f"seeded {customers} customers and {orders} orders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
