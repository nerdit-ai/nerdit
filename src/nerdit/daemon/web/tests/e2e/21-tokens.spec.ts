import { expect, test, type Page, type Request } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

// Tokens page (P23 WP4). The security-critical behaviour under test is D-P23-5:
// the minted plaintext is shown exactly once, in a modal, and is gone from the
// page — and from browser storage — as soon as that modal is closed. The
// in-memory half of that pin (the mutation is evicted from the MutationCache)
// lives in the vitest suite, since the SPA deliberately exposes no query client
// to the page.

const MINTED_PLAINTEXT = "nrd_e2e_plaintext_shown_once";

const ACTIVE_TOKEN = {
  id: "tok0000000001",
  name: "ci-runner",
  role: "submitter",
  max_gpus: 1,
  max_concurrent_jobs: 2,
  created_at: "2026-08-05T10:00:00+00:00",
  last_used_at: "2026-08-07T09:00:00+00:00",
  revoked: false
};

const REVOKED_TOKEN = {
  id: "tok0000000002",
  name: "retired-laptop",
  role: "readonly",
  max_gpus: null,
  max_concurrent_jobs: null,
  created_at: "2026-06-01T10:00:00+00:00",
  last_used_at: null,
  revoked: true
};

/**
 * `/api/tokens` stubs, registered after `mockApi` (Playwright route precedence
 * is last-registered-wins). `revoked` is mutated by the DELETE handler so the
 * re-list after a revoke is a real state change, not a static fixture.
 */
async function mockTokens(page: Page, options: { forbidden?: boolean } = {}): Promise<void> {
  const state = { revoked: false };

  await page.route(/\/api\/tokens(\?.*)?$/, (route) => {
    if (options.forbidden) {
      return route.fulfill({
        status: 403,
        contentType: "application/json",
        body: JSON.stringify({
          code: "auth.forbidden",
          message: "Admin role required",
          detail: "Admin role required"
        })
      });
    }
    const method = route.request().method();
    if (method === "POST") {
      const body = route.request().postDataJSON() as { name?: string; role?: string };
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          id: "tok0000000003",
          name: body?.name ?? "new-token",
          role: body?.role ?? "submitter",
          max_gpus: null,
          max_concurrent_jobs: null,
          created_at: "2026-08-07T12:00:00+00:00",
          last_used_at: null,
          revoked: false,
          token: MINTED_PLAINTEXT
        })
      });
    }
    const url = new URL(route.request().url());
    const includeRevoked = url.searchParams.get("include_revoked") === "true";
    const active = state.revoked ? { ...ACTIVE_TOKEN, revoked: true } : ACTIVE_TOKEN;
    const items = includeRevoked
      ? [active, REVOKED_TOKEN]
      : [active, REVOKED_TOKEN].filter((t) => !t.revoked);
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(items)
    });
  });

  await page.route(/\/api\/tokens\/[^/?]+$/, (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    state.revoked = true;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ id: "tok0000000001", revoked: true })
    });
  });
}

test("lists tokens and reveals revoked ones behind the toggle", async ({ page }) => {
  await mockApi(page);
  await mockTokens(page);
  await loginAsToken(page);

  await page.getByRole("link", { name: "Tokens", exact: true }).click();
  await expect(page).toHaveURL("/tokens");

  await expect(page.getByRole("heading", { name: "Tokens", level: 1 })).toBeVisible();
  await expect(page.getByText("ci-runner")).toBeVisible();
  // Quotas render from the TokenView fields only (∞ for an uncapped one).
  await expect(page.getByText("1 gpu · 2 jobs")).toBeVisible();
  await expect(page.getByText("retired-laptop")).toHaveCount(0);

  await page.getByRole("switch", { name: /include revoked/i }).click();
  await expect(page.getByText("retired-laptop")).toBeVisible();
});

test("create shows the plaintext exactly once and drops it on close", async ({ page }) => {
  await mockApi(page);
  await mockTokens(page);
  await loginAsToken(page);

  await page.goto("/tokens");

  await page.getByLabel("Name", { exact: true }).fill("deploy-bot");
  await page.getByRole("button", { name: /create token/i }).click();

  const modal = page.getByRole("dialog", { name: "New token" });
  await expect(modal).toBeVisible();
  await expect(modal.getByText(/only time you will see this value/i)).toBeVisible();
  await expect(page.getByTestId("minted-token")).toContainText(MINTED_PLAINTEXT);

  await modal.getByRole("button", { name: /^done$/i }).click();

  // Gone from the DOM...
  await expect(modal).toHaveCount(0);
  await expect(page.getByTestId("minted-token")).toHaveCount(0);
  await expect(page.getByText(MINTED_PLAINTEXT)).toHaveCount(0);

  // ...and never persisted to browser storage (D-P23-5).
  const stored = await page.evaluate(() =>
    JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } })
  );
  expect(stored).not.toContain(MINTED_PLAINTEXT);

  // Re-opening the page does not resurrect it.
  await page.reload();
  await expect(page.getByText("ci-runner")).toBeVisible();
  await expect(page.getByText(MINTED_PLAINTEXT)).toHaveCount(0);
});

/** Every `POST /api/tokens`, so the spec can assert on what was (not) minted. */
function tokenPosts(page: Page): Request[] {
  const posts: Request[] = [];
  page.on("request", (req) => {
    if (req.method() === "POST" && /\/api\/tokens(\?|$)/.test(req.url())) posts.push(req);
  });
  return posts;
}

test("an unparseable quota blocks the create; only blank means uncapped", async ({ page }) => {
  await mockApi(page);
  await mockTokens(page);
  const posts = tokenPosts(page);
  await loginAsToken(page);

  await page.goto("/tokens");

  const create = page.getByRole("button", { name: /create token/i });
  const gpus = page.getByLabel(/Max GPUs/);
  const nameField = page.getByLabel("Name", { exact: true });
  await nameField.fill("typo-bot");

  // A typo is a form error, never a silent `null`: coercing it to "uncapped"
  // would mint an UNLIMITED token from a mistyped cap.
  for (const bad of ["abc", "-1", "1.5"]) {
    await gpus.fill(bad);
    await expect(page.getByTestId("max-gpus-error")).toBeVisible();
    await expect(create).toBeDisabled();
    await create.click({ force: true });
    expect(posts).toHaveLength(0);
  }

  // A real cap still goes through as an integer.
  await gpus.fill("2");
  await expect(page.getByTestId("max-gpus-error")).toHaveCount(0);
  await create.click();
  await expect(page.getByRole("dialog", { name: "New token" })).toBeVisible();
  expect(posts).toHaveLength(1);
  expect(posts[0].postDataJSON()).toMatchObject({ max_gpus: 2 });
  await page
    .getByRole("dialog", { name: "New token" })
    .getByRole("button", { name: /^done$/i })
    .click();

  // Blank is the ONE thing that means uncapped, and it still submits `null`.
  await nameField.fill("uncapped-bot");
  await expect(gpus).toHaveValue("");
  await expect(create).toBeEnabled();
  await create.click();
  await expect(page.getByRole("dialog", { name: "New token" })).toBeVisible();
  expect(posts).toHaveLength(2);
  expect(posts[1].postDataJSON()).toMatchObject({ max_gpus: null, max_concurrent_jobs: null });
});

test("revoke goes through the confirm dialog and re-lists", async ({ page }) => {
  await mockApi(page);
  await mockTokens(page);
  await loginAsToken(page);

  await page.goto("/tokens");
  await expect(page.getByText("ci-runner")).toBeVisible();

  await page.getByRole("button", { name: /^revoke$/i }).first().click();

  const dialog = page.getByRole("dialog");
  await expect(dialog.getByText("Revoke token")).toBeVisible();
  await dialog.getByRole("button", { name: /^revoke$/i }).click();

  await expect(page.getByText("Revoked ci-runner")).toBeVisible();
  // The invalidated list refetches; the now-revoked token drops out of the
  // default (active-only) view.
  await expect(page.getByText("ci-runner")).toHaveCount(0);
});

/** The bearer as the browser still holds it (either storage half). */
function storedToken(page: Page): Promise<string | null> {
  return page.evaluate(
    () => sessionStorage.getItem("nerdit.token") ?? localStorage.getItem("nerdit.token")
  );
}

test("a non-admin token gets the empty state, not an error loop", async ({ page }) => {
  await mockApi(page, { authRole: "submitter" });
  await mockTokens(page, { forbidden: true });
  await loginAsToken(page);

  await page.goto("/tokens");

  await expect(page.getByText("Admin token required")).toBeVisible();
  // No create form for a non-admin.
  await expect(page.getByRole("button", { name: /create token/i })).toHaveCount(0);

  // A role refusal is NOT a dead bearer: the token this submitter is using
  // everywhere else must survive the visit, and the page must stay put.
  await expect(page).toHaveURL(/\/tokens$/);
  expect(await storedToken(page)).toBeTruthy();
});

test("a revoked bearer (403 invalid_token) clears the token and bounces to login", async ({
  page
}) => {
  await mockApi(page);
  await mockTokens(page);
  await loginAsToken(page);

  // The token goes dead only now — as it would if an admin revoked the very
  // token driving this tab. `ScopedTokenAuthMiddleware` answers an
  // unresolvable bearer with 403 `invalid_token`, NOT 401, so a status-only
  // reading files this under "not an admin" and wedges the user: stale token
  // kept, empty state rendered, /login bounced back by the truthy token.
  await page.route(/\/api\/tokens(\?.*)?$/, (route) =>
    route.fulfill({
      status: 403,
      contentType: "application/json",
      body: JSON.stringify({
        code: "invalid_token",
        message: "Invalid token. Check your authentication token.",
        detail: "Invalid token. Check your authentication token."
      })
    })
  );

  await page.goto("/tokens");

  await expect(page).toHaveURL(/\/login$/);
  expect(await storedToken(page)).toBeNull();
  await expect(page.getByText("Admin token required")).toHaveCount(0);
});
