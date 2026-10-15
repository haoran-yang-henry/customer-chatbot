import assert from "node:assert/strict";
import { test } from "node:test";

import { parseEvent, readEvents } from "../public/api.js";

function streamOf(...chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

test("an event block becomes its name and JSON data", () => {
  assert.deepEqual(parseEvent('event: delta\ndata: {"text": "Hallo"}'), { event: "delta", data: { text: "Hallo" } });
  assert.equal(parseEvent(": keep-alive"), null);
});

test("events are reassembled whatever the network chunking", async () => {
  const stream = streamOf(
    'event: delta\ndata: {"text": "Die Rück',
    'sendung "}\r\n\r\nevent: delta\ndata: {"text": "ist kostenlos."}\n\nevent: fin',
    'al\ndata: {"answer": "Die Rücksendung ist kostenlos.", "behavior": "answer"}\n\n',
  );

  const events = [];
  for await (const event of readEvents(stream)) events.push(event);

  assert.deepEqual(
    events.map((e) => e.event),
    ["delta", "delta", "final"],
  );
  assert.equal(events.map((e) => e.data.text ?? "").join(""), "Die Rücksendung ist kostenlos.");
  assert.equal(events[2].data.behavior, "answer");
});
