"""Authentication dependency: anonymous requests are allowed unless ``CA_AUTH_REQUIRED``; bad
tokens never are."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from customer_agent.application.ports import AuthenticationError, Principal, TokenVerifier

_CHALLENGE = {"WWW-Authenticate": 'Bearer error="invalid_token"'}


async def current_principal(request: Request) -> Principal | None:
    """``None`` for anonymous requests. A presented token must be valid and for this tenant."""
    header = request.headers.get("authorization")
    if header is None:
        if request.app.state.auth_required:
            raise HTTPException(401, "please log in", {"WWW-Authenticate": "Bearer"})
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Authorization header must be 'Bearer <token>'", _CHALLENGE)
    verifier: TokenVerifier | None = request.app.state.verifier
    if verifier is None:
        raise HTTPException(401, "authentication is not configured", _CHALLENGE)
    try:
        principal = await verifier.verify(token)
    except AuthenticationError:
        raise HTTPException(401, "invalid or expired token", _CHALLENGE) from None
    if principal.tenant_id != request.app.state.tenant_id:
        raise HTTPException(403, "token is for another tenant")
    return principal


CurrentPrincipal = Annotated[Principal | None, Depends(current_principal)]
