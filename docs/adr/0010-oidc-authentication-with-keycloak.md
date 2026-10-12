# ADR-0010: OIDC authentication with Keycloak, JWT verification via JWKS

- **Status:** Accepted
- **Date:** 2026-10-07
- **Deciders:** Project owner
- **Related:** FR-20, FR-21, [threat model T06, T13](../threat-model.md), [ADR-0006](0006-no-order-mutations-by-the-agent.md)

## Context

From Day 5 the agent reads order data, so it must know *which* customer is asking — reliably,
and never from text the customer typed. The shop's customer account would normally be the
identity provider. For this project we need an issuer that behaves like a production identity
provider and can be run locally.

## Options considered

1. **Self-signed HS256 tokens** from a secret in `.env` and a small CLI — minimal setup, but a
   shared secret between issuer and API, and no realistic OIDC flows.
2. **Keycloak** (OIDC, RS256, JWKS) in Docker Compose with an imported realm — the most common
   open-source identity provider in German enterprises; asymmetric keys; standard discovery.
3. **Managed IdP** (Entra ID, Auth0) — realistic, but external accounts and not offline-capable.

## Decision

We choose **option 2**.

- `infra/keycloak/nordelektro-realm.json` defines realm `nordelektro`, public client
  `customer-chat` (PKCE; password grant enabled **for local testing only**), an audience mapper
  for `customer-agent-api`, and admin-managed user attributes `customer_id`, `tenant_id` mapped
  into the access token. Test users match `data/seed/customers.json`.
- The API verifies tokens itself (`JwtVerifier`): signature against the issuer's JWKS, RS256
  only (no `none`, no HMAC → no algorithm confusion), `iss`, `aud`, `exp`, `iat`, `sub` required,
  30 s leeway. The key fetch (`PyJWKClient`, synchronous) runs in a thread pool and is cached.
- Requests without a token are anonymous (knowledge questions remain possible); an invalid token
  is `401` with `WWW-Authenticate: Bearer`; a token for another tenant is `403`.
- `customer_id` reaches the use cases only via the verified `Principal`; sessions are bound to it.
- `CA_AUTH_ISSUER` unset disables authentication (all requests anonymous, tokens rejected).

## Consequences

**Positive**

- Production-like: replacing Keycloak with the shop's real IdP is configuration (issuer,
  audience, claim names), not code.
- Asymmetric keys: the API holds no signing secret.

**Negative / trade-offs**

- One more container (~500 MB RAM) for local development.
- The password grant is deprecated in OAuth 2.1; a real front end uses Authorization Code + PKCE.
- Claim names (`customer_id`, `tenant_id`) are a contract with the IdP configuration.
