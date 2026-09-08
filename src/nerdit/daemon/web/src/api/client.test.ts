import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, apiRaw } from "./client";

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

function okResponse(): Response {
  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
}

function requestHeaders(fetchMock: ReturnType<typeof vi.fn>): Headers {
  const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
  return new Headers(init?.headers);
}

describe("api request headers", () => {
  beforeEach(() => {
    vi.stubGlobal("sessionStorage", memoryStorage());
    vi.stubGlobal("localStorage", memoryStorage());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("retains stored auth, idempotency, and JSON content headers", async () => {
    sessionStorage.setItem("nerdit.token", "stored-token");
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal("fetch", fetchMock);

    await api("/models", {
      method: "POST",
      body: JSON.stringify({ model: "llama3.1:8b" }),
      headers: { "Idempotency-Key": "idem-json" }
    });

    const headers = requestHeaders(fetchMock);
    expect(headers.get("Authorization")).toBe("Bearer stored-token");
    expect(headers.get("Idempotency-Key")).toBe("idem-json");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("retains auth and idempotency for FormData without setting content type", async () => {
    localStorage.setItem("nerdit.token", "persistent-token");
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal("fetch", fetchMock);
    const form = new FormData();
    form.append("name", "demo");

    await api("/deploy", {
      method: "POST",
      body: form,
      headers: { "Idempotency-Key": "idem-form" }
    });

    const headers = requestHeaders(fetchMock);
    expect(headers.get("Authorization")).toBe("Bearer persistent-token");
    expect(headers.get("Idempotency-Key")).toBe("idem-form");
    expect(headers.has("Content-Type")).toBe(false);
  });

  it("preserves an explicit authorization header over the stored token", async () => {
    sessionStorage.setItem("nerdit.token", "stale-token");
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal("fetch", fetchMock);

    await api("/auth/check", {
      headers: { Authorization: "Bearer candidate-token" }
    });

    expect(requestHeaders(fetchMock).get("Authorization")).toBe("Bearer candidate-token");
  });

  it("preserves default auth and JSON headers without caller headers", async () => {
    sessionStorage.setItem("nerdit.token", "stored-token");
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal("fetch", fetchMock);

    await api("/services", { method: "POST", body: JSON.stringify({}) });

    const headers = requestHeaders(fetchMock);
    expect(headers.get("Authorization")).toBe("Bearer stored-token");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("clears stored credentials and redirects on 401", async () => {
    sessionStorage.setItem("nerdit.token", "session-token");
    localStorage.setItem("nerdit.token", "persistent-token");
    localStorage.setItem("nerdit.token.persist", "1");
    const assign = vi.fn();
    vi.stubGlobal("window", { location: { assign } });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Invalid token" }), {
          status: 401,
          headers: { "Content-Type": "application/json" }
        })
      )
    );

    await expect(api("/services")).rejects.toThrow("Invalid token");

    expect(sessionStorage.getItem("nerdit.token")).toBeNull();
    expect(localStorage.getItem("nerdit.token")).toBeNull();
    expect(localStorage.getItem("nerdit.token.persist")).toBeNull();
    expect(assign).toHaveBeenCalledWith("/login");
  });

  it("surfaces the P1 error envelope code and hint on ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            code: "config.stale",
            message: "The app config changed since you last read it.",
            hint: "Re-read it (GET) and retry with the new ETag."
          }),
          { status: 409, headers: { "Content-Type": "application/json" } }
        )
      )
    );

    const err = (await api("/config/apps/x/deploy", { method: "PUT" }).catch(
      (e) => e
    )) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.code).toBe("config.stale");
    expect(err.message).toBe("The app config changed since you last read it.");
    expect(err.hint).toBe("Re-read it (GET) and retry with the new ETag.");
  });

  it("falls back to legacy `detail` when there is no envelope code", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Not found" }), {
          status: 404,
          headers: { "Content-Type": "application/json" }
        })
      )
    );
    const err = (await api("/services/nope").catch((e) => e)) as ApiError;
    expect(err.message).toBe("Not found");
    expect(err.code).toBeUndefined();
  });

  it("apiRaw exposes the ETag header for optimistic concurrency", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ service_name: "my-app", etag: "etag-1" }), {
          status: 200,
          headers: { "Content-Type": "application/json", ETag: "etag-1" }
        })
      )
    );
    const { data, etag } = await apiRaw<{ service_name: string }>("/config/apps/my-app");
    expect(data.service_name).toBe("my-app");
    expect(etag).toBe("etag-1");
  });
});
