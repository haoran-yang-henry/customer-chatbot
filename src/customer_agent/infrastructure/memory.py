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


class InMemoryAuditLog:
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    async def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)
