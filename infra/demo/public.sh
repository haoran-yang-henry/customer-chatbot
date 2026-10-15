#!/usr/bin/env bash
# Browser demo: the frontend on GitHub Pages, this machine as the backend, reachable through a
# Cloudflare quick tunnel for as long as it runs.
#
#   infra/demo/public.sh        start (or restart); prints the link and the demo password
#   infra/demo/public.sh down   stop the tunnel and restore the local setup
#
# Exposed through the tunnel: chat, health, and the login of the nordelektro realm (see
# Caddyfile). Chat requires a login; the demo users get a new password on every start
# (DEMO_PASSWORD in the environment or .env keeps a fixed one).
set -euo pipefail
cd "$(dirname "$0")/../.."

PAGES_URL=${PAGES_URL:-https://haoran-yang-henry.github.io/customer-chatbot/}
USERS=(anna.schneider jonas.weber mehmet.yilmaz)
compose() { docker compose -f docker-compose.yml -f infra/demo/compose.yml --profile app --profile demo "$@"; }

if [ "${1:-up}" = down ]; then
  compose rm --stop --force tunnel proxy app
  docker compose up -d --wait keycloak  # recreated with the local hostname and the repo passwords
  echo "Demo stopped."
  exit 0
fi

grep -q '^CA_LLM_API_KEY=..*' .env 2>/dev/null && grep -q '^CA_OPENAI_API_KEY=..*' .env ||
  { echo "Set CA_LLM_API_KEY and CA_OPENAI_API_KEY in .env first." >&2; exit 1; }
if [ -z "${DEMO_PASSWORD:-}" ]; then
  DEMO_PASSWORD=$(sed -n 's/^DEMO_PASSWORD=//p' .env)
fi
DEMO_PASSWORD=${DEMO_PASSWORD:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(9))')}

echo "Starting the tunnel …"
compose rm --stop --force tunnel > /dev/null 2>&1 || true
compose up -d tunnel
DEMO_URL=
for _ in $(seq 1 60); do
  DEMO_URL=$(compose logs tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1 || true)
  [ -n "$DEMO_URL" ] && break
  sleep 1
done
[ -n "$DEMO_URL" ] || { echo "The tunnel did not report its address:" >&2; compose logs tunnel >&2; exit 1; }
export DEMO_URL

echo "Preparing the database …"
docker compose up -d --wait postgres
uv run alembic upgrade head
uv run customer-agent-seed > /dev/null
uv run customer-agent-ingest > /dev/null

echo "Starting Keycloak, API and proxy for $DEMO_URL …"
compose up -d --build --wait keycloak app proxy

kcadm() { compose exec -T keycloak /opt/keycloak/bin/kcadm.sh "$@"; }
kcadm config credentials --server http://localhost:8080 --realm master --user admin --password admin > /dev/null 2>&1
for user in "${USERS[@]}"; do
  kcadm set-password -r nordelektro --username "$user" --new-password "$DEMO_PASSWORD"
done

echo "Waiting until the API answers through the tunnel …"
for _ in $(seq 1 60); do
  curl -fsS -o /dev/null "$DEMO_URL/health/ready" 2>/dev/null && break
  sleep 2
done
curl -fsS -o /dev/null "$DEMO_URL/health/ready" || { echo "Not reachable through the tunnel." >&2; exit 1; }

cat <<INFO

Demo is live while this machine and Docker are running.

  Link:      ${PAGES_URL}?backend=${DEMO_URL}
  Accounts:  ${USERS[*]}
  Password:  ${DEMO_PASSWORD}

Stop it with: make demo-public-down
INFO
