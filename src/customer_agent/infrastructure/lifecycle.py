"""Resource lifecycle helpers for adapters that hold connections (HTTP pools, DB engines)."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsAclose(Protocol):
    async def aclose(self) -> None: ...


async def aclose_all(*resources: object) -> None:
    """Close every resource that supports it; keeps going if one fails."""
    errors: list[BaseException] = []
    for resource in resources:
        if isinstance(resource, SupportsAclose):
            try:
                await resource.aclose()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise ExceptionGroup("closing resources failed", errors)  # type: ignore[type-var]
