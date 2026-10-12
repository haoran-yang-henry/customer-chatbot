# Keycloak (local development)

`docker compose up -d keycloak` imports `nordelektro-realm.json` on first start (ADR-0010).

| | |
|---|---|
| Admin console | http://localhost:8080/admin — `admin` / `admin` (development only) |
| Realm / issuer | `nordelektro` — `http://localhost:8080/realms/nordelektro` |
| Client | `customer-chat` (public, PKCE; password grant enabled for local testing only) |
| API audience | `customer-agent-api` (audience mapper) |
| Custom claims | `customer_id`, `tenant_id` (from admin-managed user attributes) |

Test users (password `nordelektro-dev`), matching `data/seed/customers.json`:

| Username | customer_id |
|---|---|
| anna.schneider | C-1001 |
| jonas.weber | C-1002 |
| mehmet.yilmaz | C-1003 |

Get a token: `make token USER=anna.schneider`. The realm file is the source of truth; changes in
the admin console are lost when the container's volume is removed.
