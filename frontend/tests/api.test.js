import assert from "node:assert/strict";
import { test } from "node:test";

import { ApiError, parseEvent, readEvents, sendFeedback } from "../public/api.js";

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

test("feedback posts the rating of one answer", async (t) => {
  const requests = [];
  t.mock.method(globalThis, "fetch", async (url, init) => {
    requests.push({ url, init });
    return new Response(null, { status: 204 });
  });

  await sendFeedback("https://agent.example", "tok", { sessionId: "s1", requestId: "r1", helpful: false });

  assert.equal(requests[0].url, "https://agent.example/v1/feedback");
  assert.equal(requests[0].init.headers.authorization, "Bearer tok");
  assert.deepEqual(JSON.parse(requests[0].init.body), { session_id: "s1", request_id: "r1", rating: "down" });
});

test("rejected feedback is an ApiError with its status", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(null, { status: 404 }));

  await assert.rejects(sendFeedback("https://agent.example", "tok", { sessionId: "s1", requestId: "x", helpful: true }), (error) => error instanceof ApiError && error.status === 404);
});
