// Where the backend runs: the hosted one by default; a demo link may name another as ?backend=…,
// and the browser remembers it.

export const REALM_PATH = "/realms/nordelektro";
export const CLIENT_ID = "customer-chat";

// The Cloud Run service of infra/gcp (one origin for API and login).
export const DEFAULT_BACKEND = "https://customer-agent-865980992655.europe-west1.run.app";

// Only the hosted, tunnel and local addresses: a link must not be able to send people to a
// look-alike login.
const ALLOWED_BACKENDS = [
  new RegExp(`^${DEFAULT_BACKEND.replaceAll(".", "\\.")}$`),
  /^https:\/\/[a-z0-9-]+\.trycloudflare\.com$/,
  /^http:\/\/(localhost|127\.0\.0\.1)(:\d{1,5})?$/,
];

/** The backend's origin, or null if the value is not an allowed backend address. */
export function normaliseBackend(value) {
  let url;
  try {
    url = new URL(String(value).trim());
  } catch {
    return null;
  }
  return ALLOWED_BACKENDS.some((pattern) => pattern.test(url.origin)) ? url.origin : null;
}
