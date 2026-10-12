"""``customer-agent-token <username>``: get an access token from the local Keycloak.

Development only: uses the password grant of the public ``customer-chat`` client (ADR-0010).
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request

from customer_agent.config import get_settings

DEV_PASSWORD = "nordelektro-dev"  # noqa: S105 - documented local test credential


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", help="e.g. anna.schneider (see infra/keycloak/README.md)")
    parser.add_argument("--password", default=DEV_PASSWORD)
    parser.add_argument("--client-id", default="customer-chat")
    args = parser.parse_args(argv)

    issuer = get_settings().auth_issuer
    if issuer is None:
        print("CA_AUTH_ISSUER is not set", file=sys.stderr)
        return 1
    data = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": args.client_id,
            "username": args.username,
            "password": args.password,
        }
    ).encode()
    url = f"{issuer.rstrip('/')}/protocol/openid-connect/token"
    with urllib.request.urlopen(url, data=data, timeout=10) as response:  # noqa: S310 - trusted dev URL
        print(json.load(response)["access_token"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
