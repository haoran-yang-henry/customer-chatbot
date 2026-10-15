import assert from "node:assert/strict";
import { test } from "node:test";

import { tokens } from "../public/format.js";

test("bold text and links are recognised, everything else stays text", () => {
  assert.deepEqual(tokens("Sie haben **30 Tage** Zeit: https://www.nordelektro.de/ruecksendung."), [
    { type: "text", value: "Sie haben " },
    { type: "bold", value: "30 Tage" },
    { type: "text", value: " Zeit: " },
    { type: "link", value: "https://www.nordelektro.de/ruecksendung" },
    { type: "text", value: "." },
  ]);
});

test("markup in model output is never interpreted", () => {
  assert.deepEqual(tokens('<img src=x onerror="alert(1)"> javascript:alert(1)'), [
    { type: "text", value: '<img src=x onerror="alert(1)"> javascript:alert(1)' },
  ]);
  assert.deepEqual(tokens("http://insecure.example"), [{ type: "text", value: "http://insecure.example" }]);
});
