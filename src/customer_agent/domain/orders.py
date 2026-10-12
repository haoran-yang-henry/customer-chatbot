"""Orders and eligibility rules (FR-23). Pure functions with an explicit ``today``: deterministic,
fully unit-tested, and advisory only — the order system or a human makes the binding decision
(ADR-0006)."""

from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

RETURN_WINDOW_DAYS = 30  # policy-returns-v5
# In-ear headphones: returnable only with an intact hygiene seal (policy-return-exceptions).
HYGIENE_SEAL_SKUS = frozenset({"KW-BUDSS"})


class OrderStatus(StrEnum):
    PENDING_PAYMENT = "pending_payment"
    PAID = "paid"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURN_REQUESTED = "return_requested"
    RETURNED = "returned"


class OrderItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: str
    name: str
    quantity: int
    unit_price: Decimal


class Shipping(BaseModel):
    model_config = ConfigDict(frozen=True)

    carrier: str
    tracking_number: str | None = None
    shipped_at: date | None = None
    delivered_at: date | None = None
    expected_delivery: date | None = None


class ReturnInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    requested_at: date | None = None
    received_at: date | None = None
    refund_status: str | None = None
    refunded_at: date | None = None
    refund_amount: Decimal | None = None


class Order(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: str
    customer_id: str
    status: OrderStatus
    created_at: date
    items: list[OrderItem]
    total: Decimal
    payment_method: str
    payment_status: str
    shipping: Shipping
    return_info: ReturnInfo | None = None

    @property
    def needs_hygiene_seal(self) -> bool:
        return any(item.sku in HYGIENE_SEAL_SKUS for item in self.items)


class Eligibility(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str  # machine-readable code; wording lives in the application layer
    deadline: date | None = None


_CANCELLABLE = {OrderStatus.PENDING_PAYMENT, OrderStatus.PAID, OrderStatus.PROCESSING}
_ON_THE_WAY_OR_LATER = {
    OrderStatus.SHIPPED,
    OrderStatus.DELIVERED,
    OrderStatus.RETURN_REQUESTED,
    OrderStatus.RETURNED,
}


def can_cancel(order: Order) -> Eligibility:
    if order.status in _CANCELLABLE:
        return Eligibility(allowed=True, reason="not_shipped")
    if order.status is OrderStatus.CANCELLED:
        return Eligibility(allowed=False, reason="already_cancelled")
    if order.status in _ON_THE_WAY_OR_LATER:
        return Eligibility(allowed=False, reason="already_shipped")
    raise AssertionError(f"unhandled status {order.status}")  # pragma: no cover


def can_return(order: Order, today: date) -> Eligibility:
    match order.status:
        case OrderStatus.DELIVERED:
            delivered = order.shipping.delivered_at
            if delivered is None:  # inconsistent data: never guess in the customer's favour
                return Eligibility(allowed=False, reason="delivery_date_unknown")
            deadline = delivered + timedelta(days=RETURN_WINDOW_DAYS)
            if today <= deadline:
                return Eligibility(allowed=True, reason="within_window", deadline=deadline)
            return Eligibility(allowed=False, reason="window_expired", deadline=deadline)
        case OrderStatus.RETURN_REQUESTED:
            return Eligibility(allowed=False, reason="return_in_progress")
        case OrderStatus.RETURNED:
            return Eligibility(allowed=False, reason="already_returned")
        case OrderStatus.CANCELLED:
            return Eligibility(allowed=False, reason="cancelled")
        case _:
            return Eligibility(allowed=False, reason="not_delivered")
