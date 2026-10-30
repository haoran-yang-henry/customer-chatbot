"""The nordelektro realm for the hosted backend, derived from the development realm.

    python realm.py nordelektro-realm.json > realm.json

The development realm (infra/keycloak) is the source; the hosted one differs only where it is
reachable from the internet: the demo users' password comes from the DEMO_PASSWORD environment
variable (Keycloak resolves ``${...}`` placeholders on import) instead of the published
development password, HTTPS is required, the password grant is off, and the login redirects only
to the GitHub Pages frontend.
"""

import json
import sys

PAGES_ORIGIN = "https://haoran-yang-henry.github.io"
CHAT_CLIENT = "customer-chat"


def hosted(realm: dict) -> dict:
    realm["sslRequired"] = "external"
    realm["registrationAllowed"] = False
    for user in realm.get("users", []):
        for credential in user.get("credentials", []):
            credential["value"] = "${DEMO_PASSWORD}"
            credential["temporary"] = False
    [client] = [c for c in realm["clients"] if c["clientId"] == CHAT_CLIENT]
    client["directAccessGrantsEnabled"] = False
    client["redirectUris"] = [f"{PAGES_ORIGIN}/customer-chatbot/*"]
    client["webOrigins"] = [PAGES_ORIGIN]  # logout redirects follow ("+")
    return realm


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as source:
        json.dump(hosted(json.load(source)), sys.stdout, ensure_ascii=False, indent=2)
