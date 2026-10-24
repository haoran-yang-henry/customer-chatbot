"""The two tools (rebuild phase 2): who calls comes from the config, never from the arguments."""

import pytest
from pydantic import ValidationError

from customer_agent.application.tools import CreateTicket, GetOrderStatus, tool_config
from customer_agent.domain.intents import Intent
from customer_agent.infrastructure.memory import InMemoryAuditLog, InMemoryTicketService
from tests.fakes import SeedOrderRepository

OWN_ORDER = "NE-100231"  # belongs to C-1001
CONFIG = tool_config(customer_id="C-1001", tenant_id="nordelektro", session_id="s1")


def order_tool() -> tuple[GetOrderStatus, SeedOrderRepository, InMemoryAuditLog]:
    orders, audit = SeedOrderRepository(), InMemoryAuditLog()
    return GetOrderStatus(orders=orders, audit_log=audit), orders, audit


def test_the_arguments_a_model_could_fill_name_no_caller() -> None:
    tool, _, _ = order_tool()
    ticket = CreateTicket(tickets=InMemoryTicketService(), audit_log=InMemoryAuditLog())

    assert set(tool.args) == {"order_id"}
    assert set(ticket.args) == {"intent", "order_id", "request"}


async def test_the_lookup_is_for_the_customer_in_the_config_and_audited() -> None:
    tool, orders, audit = order_tool()

    order = await tool.ainvoke({"order_id": OWN_ORDER, "customer_id": "C-9999"}, config=CONFIG)

    assert order is not None and order.order_id == OWN_ORDER
    assert orders.lookups == [(OWN_ORDER, "C-1001")]  # the extra argument changed nothing
    [entry] = audit.entries
    assert (entry.tool, entry.customer_id, entry.session_id, entry.outcome) == (
        "get_order_status",
        "C-1001",
        "s1",
        "found",
    )


async def test_someone_elses_order_is_not_found() -> None:
    tool, _, audit = order_tool()
    other = tool_config(customer_id="C-1002", tenant_id="nordelektro", session_id="s1")

    assert await tool.ainvoke({"order_id": OWN_ORDER}, config=other) is None
    assert audit.entries[0].outcome == "not_found"


@pytest.mark.parametrize("order_id", ["100231", "NE-100231' OR '1'='1", "ne-100231"])
async def test_only_normalised_order_numbers_are_accepted(order_id: str) -> None:
    tool, orders, _ = order_tool()

    with pytest.raises(ValidationError):
        await tool.ainvoke({"order_id": order_id}, config=CONFIG)
    assert orders.lookups == []


async def test_an_anonymous_lookup_is_a_bug() -> None:
    tool, _, _ = order_tool()
    anonymous = tool_config(customer_id=None, tenant_id="nordelektro", session_id="s1")

    with pytest.raises(RuntimeError, match="authenticated"):
        await tool.ainvoke({"order_id": OWN_ORDER}, config=anonymous)


async def test_tickets_are_opened_for_the_caller_until_the_daily_limit() -> None:
    tickets, audit = InMemoryTicketService(), InMemoryAuditLog()
    tool = CreateTicket(tickets=tickets, audit_log=audit, max_per_day=1)
    request = {"intent": Intent.ORDER_CANCEL, "order_id": OWN_ORDER, "request": "Storno"}

    first = await tool.ainvoke(request, config=CONFIG)
    second = await tool.ainvoke(request, config=CONFIG)

    assert (first.ticket_id, first.created) == ("T-000001", True)
    assert (second.ticket_id, second.created) == ("T-000001", False)
    assert [t.customer_id for t in tickets.tickets] == ["C-1001"]
    assert tickets.tickets[0].summary == "ORDER_CANCEL: Storno"
    assert [e.outcome for e in audit.entries] == ["T-000001", "rate_limited:T-000001"]
