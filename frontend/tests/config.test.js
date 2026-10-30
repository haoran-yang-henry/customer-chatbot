import assert from "node:assert/strict";
import { test } from "node:test";

import { DEFAULT_BACKEND, normaliseBackend } from "../public/config.js";

test("the hosted, tunnel and local addresses are accepted as their origin", () => {
  assert.equal(normaliseBackend(`${DEFAULT_BACKEND}/v1/chat`), DEFAULT_BACKEND);
  assert.equal(normaliseBackend(" https://a-b-c.trycloudflare.com/path?x=1 "), "https://a-b-c.trycloudflare.com");
  assert.equal(normaliseBackend("http://localhost:8000"), "http://localhost:8000");
  assert.equal(normaliseBackend("http://127.0.0.1:3000/"), "http://127.0.0.1:3000");
});

test("any other address is refused, so a link cannot point to a look-alike login", () => {
  for (const value of [
    "https://evil.example",
    "http://abc.trycloudflare.com", // no TLS
    "https://trycloudflare.com.evil.example",
    "https://abc.trycloudflare.com.evil.example",
    "https://customer-agent-865980992655xeurope-west1.run.app", // the dots are literal
    "https://evil-865980992655.europe-west1.run.app", // another Cloud Run service
    "http://customer-agent-865980992655.europe-west1.run.app", // no TLS
    "https://user@evil.example#.trycloudflare.com",
    "javascript:alert(1)",
    "not a url",
  ]) {
    assert.equal(normaliseBackend(value), null, value);
  }
});
