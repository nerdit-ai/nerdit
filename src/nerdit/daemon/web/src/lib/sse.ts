import { apiUrl } from "../api/client";
import { getStoredToken } from "./auth";

/**
 * Authenticated fetch-streaming for `text/event-stream` endpoints (P12 §1.3).
 *
 * A plain `EventSource` cannot send an `Authorization` header, so the admin
 * audit stream and the cluster event stream could never authenticate from the
 * dashboard (they 401-flooded instead). This helper reads the same bearer the
 * REST client uses, opens the stream with `fetch`, and parses the chunked
 * `text/event-stream` body via a `ReadableStream`.
 *
 * Failure model:
 * - 401/403 → **permanent stop** (`"forbidden"`). No retry loop, so a role
 *   gate or an expired token never floods the audit log.
 * - transport error / clean server close → reconnect with exponential backoff
 *   (1 s → 30 s cap).
 * - the caller's `AbortSignal` → stop (`"aborted"`).
 */

const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30_000;

/** One parsed Server-Sent Event (comment/ping lines are dropped upstream). */
export interface StreamEvent {
  /** Event name; "message" when the stream omits an `event:` field. */
  event: string;
  /** Concatenated `data:` payload (multiple data lines joined with "\n"). */
  data: string;
}

export interface StreamOptions {
  /** Called for every non-ping event. */
  onEvent: (event: StreamEvent) => void;
  /** Stops the loop when aborted (returns "aborted"). */
  signal?: AbortSignal;
  /** Called each time a connection opens (HTTP 200). */
  onOpen?: () => void;
  /** Called on a transport error, before backing off. */
  onError?: () => void;
}

/** Terminal reason the loop resolves with. */
export type StreamOutcome = "aborted" | "forbidden";

/**
 * Wraps an exception thrown by the caller's `onEvent` handler so `streamEvents`
 * can tell an application bug (which must propagate) apart from a transport
 * error (which triggers a reconnect). Without this, a throwing handler would be
 * swallowed by the retry loop and reconnect silently.
 */
class HandlerError extends Error {
  constructor(readonly reason: unknown) {
    super("onEvent handler threw");
    this.name = "HandlerError";
  }
}

/** Exponential backoff (ms) for the given zero-based reconnect attempt, capped. */
export function backoffDelay(attempt: number): number {
  return Math.min(INITIAL_BACKOFF_MS * 2 ** attempt, MAX_BACKOFF_MS);
}

/**
 * Parse one `text/event-stream` block (the text between blank-line separators)
 * into a `StreamEvent`, or null when it carries no data (a bare comment/ping).
 */
export function parseEventBlock(block: string): StreamEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    // Blank lines and comments (":" prefix, includes ": ping" keep-alives).
    if (line === "" || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

/** Sleep `ms`, resolving false if the signal aborts first. */
function sleep(ms: number, signal?: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve(false);
      return;
    }
    const timer = setTimeout(() => {
      cleanup();
      resolve(true);
    }, ms);
    const onAbort = () => {
      cleanup();
      resolve(false);
    };
    function cleanup() {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }
    signal?.addEventListener("abort", onAbort);
  });
}

/** Read a stream body, emitting events as complete blocks arrive. */
async function pump(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      // Normalize CRLF so the block/line split is uniform across servers.
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const parsed = parseEventBlock(block);
        if (parsed && parsed.event !== "ping") {
          try {
            onEvent(parsed);
          } catch (err) {
            // A handler bug is not a transport failure; tag it so the caller
            // sees it instead of the retry loop silently reconnecting.
            throw new HandlerError(err);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Open `path` as an authenticated event stream and drive `onEvent` until the
 * signal aborts (→ "aborted") or the server rejects auth (→ "forbidden").
 * Reconnects with capped exponential backoff on any other failure.
 */
export async function streamEvents(path: string, opts: StreamOptions): Promise<StreamOutcome> {
  const { onEvent, signal, onOpen, onError } = opts;
  let attempt = 0;
  for (;;) {
    if (signal?.aborted) return "aborted";
    try {
      const headers = new Headers({ Accept: "text/event-stream" });
      const token = getStoredToken();
      if (token) headers.set("Authorization", `Bearer ${token}`);
      const response = await fetch(apiUrl(path), { headers, signal });
      if (response.status === 401 || response.status === 403) {
        // Drain the unused body so the connection is released, not leaked.
        await response.body?.cancel();
        return "forbidden";
      }
      if (!response.ok || !response.body) {
        await response.body?.cancel();
        throw new Error(`stream failed: ${response.status}`);
      }
      onOpen?.();
      attempt = 0; // a good connection resets the backoff
      await pump(response.body, onEvent);
      // Clean server close: fall through to a short backoff, then reconnect.
    } catch (err) {
      // An onEvent handler bug must surface to the caller, not reconnect.
      if (err instanceof HandlerError) throw err.reason;
      if (signal?.aborted) return "aborted";
      onError?.();
    }
    const proceeded = await sleep(backoffDelay(attempt), signal);
    if (!proceeded) return "aborted";
    attempt += 1;
  }
}
