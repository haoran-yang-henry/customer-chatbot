"""In-memory ticket service and audit log: side-effect-free runs (evaluation).

Evaluation must not open real tickets or fill the audit log on every run; the Postgres
adapters are covered by integration tests instead.
"""

from customer_agent.application.ports import AuditEntry, NewTicket


class InMemoryTicketService:
    def __init__(self) -> None:
        self.tickets: list[NewTicket] = []

    async def create(self, ticket: NewTicket) -> str:
        self.tickets.append(ticket)
        return f"T-{len(self.tickets):06d}"

    async def recent(self, *, customer_id: str, tenant_id: str) -> list[str]:
        """Every ticket counts as recent: an evaluation run or a test is shorter than a day."""
        numbers = [
            f"T-{index:06d}"
            for index, ticket in enumerate(self.tickets, start=1)
            if ticket.customer_id == customer_id and ticket.tenant_id == tenant_id
        ]
        return numbers[::-1]


class InMemoryAuditLog:
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    async def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)
