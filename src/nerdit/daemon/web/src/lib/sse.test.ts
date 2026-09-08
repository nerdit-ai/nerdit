import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { backoffDelay, parseEventBlock, streamEvents, type StreamEvent } from "./sse";

function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => Array.from(values.keys())[index] ?? null,
    removeItem: (key) => values.delete(key),
    setItem: (key, value) => values.set(key, value)
  };
}

function streamFrom(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    }
  });
}

describe("backoffDelay", () => {
  it("grows exponentially from 1s and caps at 30s", () => {
    expect(backoffDelay(0)).toBe(1000);
    expect(backoffDelay(1)).toBe(2000);
    expect(backoffDelay(4)).toBe(16000);
    expect(backoffDelay(5)).toBe(30000);
    expect(backoffDelay(50)).toBe(30000);
  });
});

describe("parseEventBlock", () => {
  it("returns null for a comment-only (keep-alive/ping) block", () => {
    expect(parseEventBlock(": ping")).toBeNull();
    expect(parseEventBlock("")).toBeNull();
  });

  it("parses the event name and joins multiple data lines", () => {
    expect(parseEventBlock("event: audit.recorded\ndata: a\ndata: b")).toEqual({
      event: "audit.recorded",
      data: "a\nb"
    });
  });

  it("defaults the event name to 'message' when omitted", () => {
    expect(parseEventBlock("data: hello")).toEqual({ event: "message", data: "hello" });
  });
});

describe("streamEvents", () => {
  beforeEach(() => {
    vi.stubGlobal("sessionStorage", memoryStorage());
    vi.stubGlobal("localStorage", memoryStorage());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("splits a multi-event chunk and ignores ping events", async () => {
    const controller = new AbortController();
    const events: StreamEvent[] = [];
    const body = streamFrom([
      'event: job.status_changed\ndata: {"job_id":"a"}\n\n' +
        "event: ping\ndata: 1\n\n" +
        ": keep-alive\n\n" +
        'event: job.status_changed\ndata: {"job_id":"b"}\n\n'
    ]);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 200, body }));

    const outcome = await streamEvents("/events/stream", {
      signal: controller.signal,
      onEvent: (event) => {
        events.push(event);
        if (events.length === 2) controller.abort();
      }
    });

    expect(outcome).toBe("aborted");
    expect(events.map((e) => e.event)).toEqual(["job.status_changed", "job.status_changed"]);
    expect(JSON.parse(events[0].data).job_id).toBe("a");
    expect(JSON.parse(events[1].data).job_id).toBe("b");
  });

  it("stops permanently on 401 without retrying", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ status: 401, ok: false, body: null });
    vi.stubGlobal("fetch", fetchMock);

    const outcome = await streamEvents("/audit/stream", { onEvent: () => {} });

    expect(outcome).toBe("forbidden");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("stops permanently on 403 without retrying", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ status: 403, ok: false, body: null });
    vi.stubGlobal("fetch", fetchMock);

    const outcome = await streamEvents("/audit/stream", { onEvent: () => {} });

    expect(outcome).toBe("forbidden");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("propagates an onEvent handler exception instead of reconnecting", async () => {
    const body = streamFrom(['event: audit.recorded\ndata: {"id":1}\n\n']);
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, body });
    vi.stubGlobal("fetch", fetchMock);

    const boom = new Error("handler blew up");
    await expect(
      streamEvents("/audit/stream", {
        onEvent: () => {
          throw boom;
        }
      })
    ).rejects.toBe(boom);
    // No reconnect: fetch was called exactly once.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("cancels the response body on a forbidden response", async () => {
    const cancel = vi.fn().mockResolvedValue(undefined);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ status: 403, ok: false, body: { cancel } });
    vi.stubGlobal("fetch", fetchMock);

    const outcome = await streamEvents("/audit/stream", { onEvent: () => {} });

    expect(outcome).toBe("forbidden");
    expect(cancel).toHaveBeenCalledTimes(1);
  });

  it("attaches the stored bearer token to the stream request", async () => {
    sessionStorage.setItem("nerdit.token", "stream-token");
    const fetchMock = vi.fn().mockResolvedValue({ status: 403, ok: false, body: null });
    vi.stubGlobal("fetch", fetchMock);

    await streamEvents("/audit/stream", { onEvent: () => {} });

    const headers = new Headers(fetchMock.mock.calls[0]?.[1]?.headers);
    expect(headers.get("Authorization")).toBe("Bearer stream-token");
  });
});
