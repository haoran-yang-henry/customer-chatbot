"""The agent's two tools as LangChain tools (rebuild phase 2, invariants I1 and I2).

No model sees or chooses them: the state machine (``decide``) does, and the orchestrator calls
them. As tools, each call is a run with its arguments and outcome, and a later sub-agent could
be given them behind a confirmation gate. The arguments are only what is asked for; who asks
(customer, tenant, session) comes from the caller's ``RunnableConfig``, which no model can set.
Every call is audited; ``create_ticket`` is the only state-changing tool and is rate limited.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from customer_agent.application.ports import (
    AuditEntry,
    AuditLog,
    NewTicket,
    OrderRepository,
    TicketService,
)
from customer_agent.domain.intents import Intent
from customer_agent.domain.orders import Order
from customer_agent.observability import get_logger, request_id_var, step

log = get_logger("tools")


def tool_config(*, customer_id: str | None, tenant_id: str, session_id: str) -> RunnableConfig:
    return {
        "configurable": {
            "customer_id": customer_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
        }
    }


class _Caller(BaseModel):
    customer_id: str | None
    tenant_id: str
    session_id: str

    @classmethod
    def of(cls, config: RunnableConfig) -> "_Caller":
        return cls.model_validate(config.get("configurable", {}))

    def audit(self, tool: str, arguments: dict[str, str], outcome: str) -> AuditEntry:
        return AuditEntry(
            tool=tool,
            customer_id=self.customer_id,
            session_id=self.session_id,
            request_id=request_id_var.get(),
            arguments=arguments,
            outcome=outcome,
        )


class OrderLookup(BaseModel):
    order_id: str = Field(pattern=r"^NE-\d{6}$", description="Normalised order number")


class GetOrderStatus(BaseTool):
    """Read-only, scoped to the authenticated customer in SQL."""

    name: str = "get_order_status"
    description: str = "Status, dates and items of one order of the authenticated customer."
    args_schema: type[BaseModel] = OrderLookup
    orders: SkipValidation[OrderRepository]
    audit_log: SkipValidation[AuditLog]

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("get_order_status is async only")

    async def _arun(self, order_id: str, config: RunnableConfig) -> Order | None:
        caller = _Caller.of(config)
        if caller.customer_id is None:  # decide() never routes anonymous users here
            raise RuntimeError("order lookups require an authenticated customer")
        with step("tool.get_order_status"):
            order = await self.orders.get_for_customer(
                order_id, customer_id=caller.customer_id, tenant_id=caller.tenant_id
            )
        outcome = "found" if order else "not_found"
        await self.audit_log.record(caller.audit(self.name, {"order_id": order_id}, outcome))
        return order


class TicketRequest(BaseModel):
    intent: Intent | None
    order_id: str | None = Field(default=None, pattern=r"^NE-\d{6}$")
    request: str = Field(description="The confirmed request, not the confirmation itself")


class TicketOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticket_id: str
    created: bool  # False: the daily limit was reached; ``ticket_id`` is the newest open one


class CreateTicket(BaseTool):
    """Opens a ticket for a request the customer confirmed (ADR-0014)."""

    name: str = "create_ticket"
    description: str = "Open a support ticket for a request the customer has confirmed."
    args_schema: type[BaseModel] = TicketRequest
    tickets: SkipValidation[TicketService]
    audit_log: SkipValidation[AuditLog]
    max_per_day: int | None = 3  # None: no limit

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("create_ticket is async only")

    async def _arun(
        self, intent: Intent | None, order_id: str | None, request: str, config: RunnableConfig
    ) -> TicketOutcome:
        caller = _Caller.of(config)
        arguments = {"order_id": order_id or "-", "intent": str(intent)}
        if caller.customer_id is not None and self.max_per_day is not None:
            recent = await self.tickets.recent(
                customer_id=caller.customer_id, tenant_id=caller.tenant_id
            )
            if len(recent) >= self.max_per_day:
                # Rate limit on the only state-changing tool (T12): point to the open ticket.
                outcome = f"rate_limited:{recent[0]}"
                await self.audit_log.record(caller.audit(self.name, arguments, outcome))
                log.warning("ticket limit reached recent=%d", len(recent))
                return TicketOutcome(ticket_id=recent[0], created=False)
        with step("tool.create_ticket"):
            ticket_id = await self.tickets.create(
                NewTicket(
                    tenant_id=caller.tenant_id,
                    customer_id=caller.customer_id,
                    session_id=caller.session_id,
                    intent=intent.value if intent else None,
                    order_id=order_id,
                    summary=f"{intent or 'UNKNOWN'}: {request[:500]}",
                )
            )
        await self.audit_log.record(caller.audit(self.name, arguments, ticket_id))
        log.info("ticket=%s intent=%s order_id=%s", ticket_id, intent, order_id)
        return TicketOutcome(ticket_id=ticket_id, created=True)
