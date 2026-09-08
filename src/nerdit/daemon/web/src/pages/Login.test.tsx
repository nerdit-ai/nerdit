// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Login } from "./Login";
import { clearStoredToken, getStoredToken, storeToken } from "../lib/auth";

// The token check is the only network call Login makes.
vi.mock("../api/client", () => ({
  api: vi.fn(async () => ({ ok: true }))
}));

/**
 * Regression cover for the `react-hooks/rules-of-hooks` violation that shipped
 * undetected until WP0.4 turned the rule on.
 *
 * `Login` declared `useNavigate`/`useRef`/four `useState`, then early-returned
 * `<Navigate/>` when a token was already stored, and only *then* declared the
 * magic-link `useEffect`. The hook count therefore depended on whether a token
 * was stored *at that render*, and `Login` is the one component that writes
 * that token while it is on screen (`storeToken` in `submitToken`, and the
 * magic-link effect that calls it). Any re-render that lands between the write
 * and the router unmounting `Login` runs one hook fewer than the previous one,
 * and React aborts the tree with "Rendered fewer hooks than expected".
 */
describe("Login", () => {
  beforeEach(() => {
    clearStoredToken();
    window.history.replaceState(null, "", "/login");
  });

  afterEach(() => {
    cleanup();
    clearStoredToken();
    vi.restoreAllMocks();
  });

  it("keeps a stable hook count when the stored token appears mid-mount", () => {
    // `Login` is rendered unconditionally here so it stays mounted across the
    // token flip — the render pair the guard used to straddle. Before the fix
    // React throws "Rendered fewer hooks than expected" on the second render.
    // A fresh element every time — React bails out of a re-render when handed
    // the identical element object, which would skip the case under test.
    const tree = () => (
      <MemoryRouter initialEntries={["/login"]}>
        <Login />
      </MemoryRouter>
    );
    const { rerender } = render(tree());
    expect(screen.getByLabelText(/paste your nerdit token/i)).toBeTruthy();

    storeToken("appeared-mid-mount", false);
    expect(() => rerender(tree())).not.toThrow();
  });

  it("consumes a #token= magic link, stores it and navigates", async () => {
    window.history.replaceState(null, "", "/login#token=magic-link-token");
    render(
      <MemoryRouter initialEntries={["/login"]}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<div>apps page</div>} />
        </Routes>
      </MemoryRouter>
    );

    // The effect only runs at all when it is declared above the guard.
    await waitFor(() => expect(getStoredToken()).toBe("magic-link-token"));
    await screen.findByText("apps page");
    // The effect strips the fragment once the link has been consumed.
    expect(window.location.hash).toBe("");
  });

  it("redirects straight to the apps list when a token is already stored", async () => {
    storeToken("already-signed-in", false);
    render(
      <MemoryRouter initialEntries={["/login"]}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<div>apps page</div>} />
        </Routes>
      </MemoryRouter>
    );
    await screen.findByText("apps page");
  });
});
