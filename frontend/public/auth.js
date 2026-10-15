// OpenID Connect login against Keycloak: authorization code flow with PKCE (RFC 7636), as a
// public client. Tokens live in sessionStorage, so they end with the browser tab.

import { CLIENT_ID, REALM_PATH } from "./config.js";

const TOKENS = "nordelektro.tokens";
const PENDING = "nordelektro.login"; // state and verifier while the login page is shown
const REFRESH_MARGIN_MS = 30_000;

export function base64url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export async function challengeFor(verifier) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  return base64url(new Uint8Array(digest));
}

function randomString() {
  return base64url(crypto.getRandomValues(new Uint8Array(32)));
}

/** Claims of a JWT for display only; the API verifies the signature, the browser does not. */
export function claims(token) {
  try {
    const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const json = new TextDecoder().decode(Uint8Array.from(atob(payload), (c) => c.charCodeAt(0)));
    return JSON.parse(json);
  } catch {
    return {};
  }
}

export class Auth {
  constructor(backend, redirectUri, storage = sessionStorage) {
    this.backend = backend;
    this.redirectUri = redirectUri;
    this.storage = storage;
    this.base = `${backend}${REALM_PATH}/protocol/openid-connect`;
  }

  async login() {
    const state = randomString();
    const verifier = randomString();
    this.storage.setItem(PENDING, JSON.stringify({ state, verifier }));
    const params = new URLSearchParams({
      client_id: CLIENT_ID,
      response_type: "code",
      scope: "openid",
      redirect_uri: this.redirectUri,
      state,
      code_challenge: await challengeFor(verifier),
      code_challenge_method: "S256",
      ui_locales: "de",
    });
    location.assign(`${this.base}/auth?${params}`);
  }

  /** Finishes a login if the URL carries Keycloak's answer; returns false otherwise. */
  async completeLogin(search) {
    const params = new URLSearchParams(search);
    if (!params.has("code") && !params.has("error")) return false;
    const pending = JSON.parse(this.storage.getItem(PENDING) ?? "null");
    this.storage.removeItem(PENDING);
    if (params.has("error")) throw new Error(params.get("error_description") ?? params.get("error"));
    if (!pending || pending.state !== params.get("state")) {
      throw new Error("Die Anmeldung konnte nicht zugeordnet werden. Bitte erneut anmelden.");
    }
    await this.#token({
      grant_type: "authorization_code",
      code: params.get("code"),
      redirect_uri: this.redirectUri,
      code_verifier: pending.verifier,
    });
    return true;
  }

  /** A valid access token (refreshed when close to expiry), or null if a login is needed. */
  async accessToken() {
    const tokens = this.#tokens();
    if (!tokens) return null;
    if (Date.now() < tokens.expires_at - REFRESH_MARGIN_MS) return tokens.access_token;
    try {
      await this.#token({ grant_type: "refresh_token", refresh_token: tokens.refresh_token });
      return this.#tokens().access_token;
    } catch {
      this.storage.removeItem(TOKENS);
      return null;
    }
  }

  user() {
    const tokens = this.#tokens();
    if (!tokens) return null;
    const c = claims(tokens.access_token);
    return { name: c.given_name ?? c.preferred_username, username: c.preferred_username, customerId: c.customer_id };
  }

  logout() {
    const tokens = this.#tokens();
    this.storage.removeItem(TOKENS);
    const params = new URLSearchParams({ client_id: CLIENT_ID, post_logout_redirect_uri: this.redirectUri });
    if (tokens?.id_token) params.set("id_token_hint", tokens.id_token);
    location.assign(`${this.base}/logout?${params}`);
  }

  forget() {
    this.storage.removeItem(TOKENS);
  }

  #tokens() {
    return JSON.parse(this.storage.getItem(TOKENS) ?? "null");
  }

  async #token(fields) {
    const response = await fetch(`${this.base}/token`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ client_id: CLIENT_ID, ...fields }),
    });
    if (!response.ok) throw new Error(`token endpoint answered ${response.status}`);
    const body = await response.json();
    this.storage.setItem(
      TOKENS,
      JSON.stringify({
        access_token: body.access_token,
        refresh_token: body.refresh_token,
        id_token: body.id_token ?? this.#tokens()?.id_token,
        expires_at: Date.now() + body.expires_in * 1000,
      }),
    );
  }
}
