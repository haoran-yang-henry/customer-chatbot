"""The hosted realm (infra/gcp/keycloak/realm.py), derived from the development realm."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _realm_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("realm", ROOT / "infra/gcp/keycloak/realm.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hosted() -> dict[str, Any]:
    source = json.loads((ROOT / "infra/keycloak/nordelektro-realm.json").read_text())
    realm: dict[str, Any] = _realm_module().hosted(source)
    return realm


def test_no_published_password_reaches_the_hosted_realm() -> None:
    passwords = {
        user["username"]: [c["value"] for c in user["credentials"]] for user in hosted()["users"]
    }

    assert passwords == {
        "anna.schneider": ["${DEMO_PASSWORD}"],
        "jonas.weber": ["${DEMO_PASSWORD}"],
        "mehmet.yilmaz": ["${DEMO_PASSWORD}"],
        "owner": ["${OWNER_PASSWORD}"],  # its own password, not the shared one
    }


def test_the_owner_account_acts_on_annas_demo_data() -> None:
    users = {user["username"]: user for user in hosted()["users"]}

    assert users["owner"]["attributes"] == users["anna.schneider"]["attributes"]
    assert users["owner"]["email"] != users["anna.schneider"]["email"]  # e-mails are unique
    assert users["owner"]["enabled"] and users["owner"]["emailVerified"]


def test_the_hosted_realm_is_closed_and_redirects_only_to_pages() -> None:
    realm = hosted()
    [client] = [c for c in realm["clients"] if c["clientId"] == "customer-chat"]

    assert realm["registrationAllowed"] is False
    assert realm["sslRequired"] == "external"
    assert client["directAccessGrantsEnabled"] is False
    assert client["redirectUris"] == ["https://haoran-yang-henry.github.io/customer-chatbot/*"]
