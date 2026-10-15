// The agent's HTTP API: POST /v1/chat/stream answers with Server-Sent Events — "delta" events
// with text, then one authoritative "final" event (or an "error" event).

export class ApiError extends Error {
  constructor(status, detail, retryAfter) {
    super(detail ?? `HTTP ${status}`);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

/** One SSE block ("event: …\ndata: …") as {event, data}, or null for comments and keep-alives. */
export function parseEvent(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  return data.length ? { event, data: JSON.parse(data.join("\n")) } : null;
}

/** Events of an SSE byte stream, whatever the chunking of the network. */
export async function* readEvents(stream) {
  const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const event = parseEvent(buffer.slice(0, end));
      buffer = buffer.slice(end + 2);
      if (event) yield event;
    }
  }
}

export async function* chatStream(backend, token, message, sessionId) {
  const response = await fetch(`${backend}/v1/chat/stream`, {
    method: "POST",
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify(sessionId ? { message, session_id: sessionId } : { message }),
  });
  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(response.status, detail, Number(response.headers.get("retry-after")) || null);
  }
  yield* readEvents(response.body);
}

export async function isReady(backend) {
  try {
    return (await fetch(`${backend}/health/ready`)).ok;
  } catch {
    return false;
  }
}
