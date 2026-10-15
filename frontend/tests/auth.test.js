import assert from "node:assert/strict";
import { test } from "node:test";

import { base64url, challengeFor, claims } from "../public/auth.js";

test("the PKCE challenge matches the RFC 7636 example", async () => {
  assert.equal(
    await challengeFor("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
    "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
  );
});

test("base64url has no padding and no URL-unsafe characters", () => {
  assert.equal(base64url(new Uint8Array([251, 255, 191])), "-_-_");
  assert.equal(base64url(new Uint8Array([1])), "AQ");
});

test("token claims are decoded for display, including non-ASCII names", () => {
  const payload = Buffer.from(JSON.stringify({ given_name: "Jürgen", customer_id: "C-1001" })).toString("base64url");

  assert.deepEqual(claims(`header.${payload}.signature`), { given_name: "Jürgen", customer_id: "C-1001" });
  assert.deepEqual(claims("not-a-jwt"), {});
});
