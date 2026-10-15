"""Intent taxonomy. Keep in sync with docs/intent-taxonomy.md."""

from enum import StrEnum


class IntentGroup(StrEnum):
    KNOWLEDGE = "knowledge"
    TRANSACTIONAL = "transactional"
    SUPPORT = "support"
    CONTROL = "control"


class Intent(StrEnum):
    # knowledge
    RETURN_POLICY = "RETURN_POLICY"
    SHIPPING_INFO = "SHIPPING_INFO"
    PAYMENT_INFO = "PAYMENT_INFO"
    WARRANTY_INFO = "WARRANTY_INFO"
    PRODUCT_INFO = "PRODUCT_INFO"
    SERVICE_INFO = "SERVICE_INFO"
    # transactional
    ORDER_STATUS = "ORDER_STATUS"
    ORDER_CANCEL = "ORDER_CANCEL"
    ORDER_CHANGE = "ORDER_CHANGE"
    RETURN_REQUEST = "RETURN_REQUEST"
    REFUND_STATUS = "REFUND_STATUS"
    # support
    COMPLAINT = "COMPLAINT"
    HUMAN_AGENT = "HUMAN_AGENT"
    # control
    GREETING = "GREETING"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"

    @property
    def group(self) -> IntentGroup:
        return _INTENT_GROUPS[self]

    @property
    def requires_authentication(self) -> bool:
        return self.group is IntentGroup.TRANSACTIONAL


_INTENT_GROUPS: dict[Intent, IntentGroup] = {
    Intent.RETURN_POLICY: IntentGroup.KNOWLEDGE,
    Intent.SHIPPING_INFO: IntentGroup.KNOWLEDGE,
    Intent.PAYMENT_INFO: IntentGroup.KNOWLEDGE,
    Intent.WARRANTY_INFO: IntentGroup.KNOWLEDGE,
    Intent.PRODUCT_INFO: IntentGroup.KNOWLEDGE,
    Intent.SERVICE_INFO: IntentGroup.KNOWLEDGE,
    Intent.ORDER_STATUS: IntentGroup.TRANSACTIONAL,
    Intent.ORDER_CANCEL: IntentGroup.TRANSACTIONAL,
    Intent.ORDER_CHANGE: IntentGroup.TRANSACTIONAL,
    Intent.RETURN_REQUEST: IntentGroup.TRANSACTIONAL,
    Intent.REFUND_STATUS: IntentGroup.TRANSACTIONAL,
    Intent.COMPLAINT: IntentGroup.SUPPORT,
    Intent.HUMAN_AGENT: IntentGroup.SUPPORT,
    Intent.GREETING: IntentGroup.CONTROL,
    Intent.OUT_OF_SCOPE: IntentGroup.CONTROL,
}
