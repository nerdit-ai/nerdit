// @vitest-environment jsdom
import { createElement } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { appConfigWriteRequest, idempotencyHeaders, useRoutes, useSystemGc } from "./queries";
import Tokens from "../pages/Tokens";

const UUID_V4 =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

describe("idempotencyHeaders", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses crypto.randomUUID when available", () => {
    const randomUUID = vi.fn(() => "11111111-2222-4333-8444-555555555555");
    vi.stubGlobal("crypto", { ...globalThis.crypto, randomUUID });
    expect(idempotencyHeaders()).toEqual({
      "Idempotency-Key": "11111111-2222-4333-8444-555555555555"
    });
    expect(randomUUID).toHaveBeenCalledOnce();
  });

  it("falls back to a getRandomValues-based v4 UUID when randomUUID is undefined (plain-HTTP context)", () => {
    vi.stubGlobal("crypto", {
      randomUUID: undefined,
      getRandomValues: globalThis.crypto.getRandomValues.bind(globalThis.crypto)
    });
    const { "Idempotency-Key": key } = idempotencyHeaders();
    expect(key).toMatch(UUID_V4);
  });

  it("fallback yields fresh keys per call", () => {
    vi.stubGlobal("crypto", {
      randomUUID: undefined,
      getRandomValues: globalThis.crypto.getRandomValues.bind(globalThis.crypto)
    });
    expect(idempotencyHeaders()["Idempotency-Key"]).not.toEqual(
      idempotencyHeaders()["Idempotency-Key"]
    );
  });
});

describe("appConfigWriteRequest", () => {
  it("round-trips the ETag as If-Match and mints an Idempotency-Key on a real write", () => {
    const req = appConfigWriteRequest("my-app", {
      section: "deploy",
      values: { gpus: 1 },
      etag: "etag-abc"
    });
    expect(req.path).toBe("/config/apps/my-app/deploy");
    expect(req.headers["If-Match"]).toBe("etag-abc");
    expect(req.headers["Idempotency-Key"]).toMatch(UUID_V4);
    expect(JSON.parse(req.body)).toEqual({ gpus: 1 });
  });

  it("adds ?restart=true when restart is requested", () => {
    const req = appConfigWriteRequest("my-app", {
      section: "deploy",
      values: { start: "npm start" },
      etag: "etag-abc",
      restart: true
    });
    expect(req.path).toBe("/config/apps/my-app/deploy?restart=true");
  });

  it("dry run sends neither If-Match nor an Idempotency-Key", () => {
    const req = appConfigWriteRequest("my-app", {
      section: "ai",
      values: {},
      etag: "etag-abc",
      dryRun: true
    });
    expect(req.path).toBe("/config/apps/my-app/ai?dry_run=true");
    expect(req.headers["If-Match"]).toBeUndefined();
    expect(req.headers["Idempotency-Key"]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// D-P23-5 — the SPA never persists a minted token plaintext.
// ---------------------------------------------------------------------------

/**
 * A stand-in plaintext. It must never appear in a *query* cache entry, and the
 * mutation holding it must be gone from the MutationCache once the modal that
 * displayed it is closed — which is only true when the hook's `gcTime: 0` and
 * the close handler's `mutation.reset()` are BOTH in place.
 */
const MINTED_PLAINTEXT = "nrd_test_plaintext_do_not_persist";

const CREATED_TOKEN = {
  id: "tok0000000001",
  name: "ci-runner",
  role: "submitter",
  max_gpus: null,
  max_concurrent_jobs: null,
  created_at: "2026-08-07T10:00:00+00:00",
  last_used_at: null,
  revoked: false,
  token: MINTED_PLAINTEXT
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

/** Serialized contents of a Storage, so the pin can assert the plaintext is absent. */
function dumpStorage(storage: Storage): string {
  const entries: string[] = [];
  for (let i = 0; i < storage.length; i += 1) {
    const key = storage.key(i);
    if (key) entries.push(`${key}=${storage.getItem(key)}`);
  }
  return entries.join("|");
}

describe("useCreateToken (D-P23-5 plaintext handling)", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("mints an Idempotency-Key, keeps the plaintext out of the query cache, and drops the mutation on close", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (url.startsWith("/api/auth/check")) return jsonResponse({ ok: true, role: "admin" });
      if (url.startsWith("/api/tokens") && method === "POST") {
        return jsonResponse(CREATED_TOKEN, 201);
      }
      if (url.startsWith("/api/tokens") && method === "GET") return jsonResponse([]);
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    });
    render(
      createElement(QueryClientProvider, { client: queryClient }, createElement(Tokens))
    );

    // The create form only mounts once the admin probe + role check land.
    const nameInput = await screen.findByLabelText(/^name$/i);
    fireEvent.change(nameInput, { target: { value: "ci-runner" } });
    fireEvent.click(screen.getByRole("button", { name: /create token/i }));

    // The plaintext is displayed exactly once, from local component state.
    const shown = await screen.findByTestId("minted-token");
    expect(shown.textContent).toBe(MINTED_PLAINTEXT);

    // The write carries a freshly minted Idempotency-Key.
    const post = fetchMock.mock.calls.find(
      ([, init]) => (init?.method ?? "GET").toUpperCase() === "POST"
    );
    expect(post).toBeDefined();
    expect(new Headers(post![1]!.headers).get("Idempotency-Key")).toMatch(UUID_V4);

    // (i) No *query* cache entry carries a `token` field — or the value itself.
    const cachedQueries = queryClient
      .getQueryCache()
      .getAll()
      .map((query) => JSON.stringify(query.state.data ?? null));
    expect(cachedQueries.some((data) => data.includes('"token"'))).toBe(false);
    expect(cachedQueries.some((data) => data.includes(MINTED_PLAINTEXT))).toBe(false);

    // Sanity: while the modal is open the mutation legitimately holds the value.
    expect(queryClient.getMutationCache().getAll()).toHaveLength(1);

    // (ii) After the close handler (clear local state + `reset()`), plus one
    // flushed macrotask for the zero-delay GC timer, the mutation is ABSENT
    // from the cache — not merely reset to `state.data === undefined`. This is
    // the half that fails if either `gcTime: 0` or `reset()` is ever dropped.
    fireEvent.click(screen.getByRole("button", { name: /done/i }));
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(screen.queryByTestId("minted-token")).toBeNull();
    expect(queryClient.getMutationCache().getAll()).toHaveLength(0);

    // And it never reached browser storage either.
    expect(dumpStorage(window.localStorage)).not.toContain(MINTED_PLAINTEXT);
    expect(dumpStorage(window.sessionStorage)).not.toContain(MINTED_PLAINTEXT);
  });
});

// ---------------------------------------------------------------------------
// A gc dry run must not claim an Idempotency-Key (the daemon's
// middleware bypasses the claim on a dry run, so a key sent there would be
// wasted at best and would poison the real run that follows at worst).
// ---------------------------------------------------------------------------

function gcResult(dryRun: boolean) {
  return {
    dry_run: dryRun,
    images: { removed: ["nerdit-app/old"], skipped: [], reclaimed_bytes_estimate: 1024 },
    orphan_data: { enabled: false, removed: [], skipped: [] },
    reports: {
      weights: { ollama: 0, huggingface: 0 },
      build_cache_bytes: 0,
      backups_over_keep: 0
    },
    warnings: []
  };
}

describe("useSystemGc (dry run claims no Idempotency-Key)", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  async function callGc(dryRun: boolean) {
    const fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(
      async () => jsonResponse(gcResult(dryRun))
    );
    vi.stubGlobal("fetch", fetchMock);

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    });
    const { result } = renderHook(() => useSystemGc(), {
      wrapper: ({ children }) =>
        createElement(QueryClientProvider, { client: queryClient }, children)
    });

    await act(async () => {
      await result.current.mutateAsync({ includeOrphanData: true, dryRun });
    });

    const [url, init] = fetchMock.mock.calls[0];
    return { url: String(url), init: init!, headers: new Headers(init?.headers) };
  }

  it("sends no Idempotency-Key on a dry run", async () => {
    const { url, init, headers } = await callGc(true);
    expect(url).toBe("/api/system/gc?dry_run=true");
    expect(headers.get("Idempotency-Key")).toBeNull();
    expect(JSON.parse(String(init.body))).toEqual({ include_orphan_data: true });
  });

  it("mints an Idempotency-Key on a real run", async () => {
    const { url, headers } = await callGc(false);
    expect(url).toBe("/api/system/gc");
    expect(headers.get("Idempotency-Key")).toMatch(UUID_V4);
  });
});

// ---------------------------------------------------------------------------
// `useRoutes` builds its URL correctly. The cursor is opaque server text
// (it can carry `+`, `/`, `=`), so it MUST be percent-encoded: a raw `+` in a
// query string decodes to a space server-side and silently returns the wrong
// page. The first page must also send no `cursor` param at all, not `cursor=`.
// ---------------------------------------------------------------------------

describe("useRoutes (cursor URL build)", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  async function callRoutes(cursor?: string) {
    const fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(
      async () => jsonResponse({ items: [], next_cursor: null, live_table: "disabled" })
    );
    vi.stubGlobal("fetch", fetchMock);

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderHook(() => useRoutes(cursor), {
      wrapper: ({ children }) =>
        createElement(QueryClientProvider, { client: queryClient }, children)
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    return {
      url: String(fetchMock.mock.calls[0][0]),
      keys: queryClient
        .getQueryCache()
        .getAll()
        .map((query) => query.queryKey)
    };
  }

  it("omits the cursor entirely on the first page and keys it as null", async () => {
    const { url, keys } = await callRoutes();
    expect(url).toBe("/api/routes?limit=50");
    expect(keys).toEqual([["routes", null]]);
  });

  it("percent-encodes an opaque cursor and keys the page by it", async () => {
    const { url, keys } = await callRoutes("a+b/c=");
    expect(url).toBe("/api/routes?limit=50&cursor=a%2Bb%2Fc%3D");
    expect(keys).toEqual([["routes", "a+b/c="]]);
  });
});
