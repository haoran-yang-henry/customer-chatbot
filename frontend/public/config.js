// Where the backend runs. The demo link carries it as ?backend=…; the browser remembers it.

export const REALM_PATH = "/realms/nordelektro";
export const CLIENT_ID = "customer-chat";

// Only tunnel and local addresses: a link must not be able to send people to a look-alike login.
const ALLOWED_BACKENDS = [
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
