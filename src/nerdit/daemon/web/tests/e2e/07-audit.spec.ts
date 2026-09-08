import { expect, test } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

test("activity table renders entries and paginates via the cursor", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/activity");
  await expect(page.getByRole("heading", { name: "Activity", level: 1 })).toBeVisible();

  // First page: actions, principal and result badge all render.
  await expect(page.getByText("deploy.create")).toBeVisible();
  await expect(page.getByText("secrets.set")).toBeVisible();
  await expect(page.getByText("model.serve")).toBeVisible();
  await expect(page.getByText("tok-admin-01").first()).toBeVisible();
  await expect(page.getByText("ok", { exact: true }).first()).toBeVisible();

  // next_cursor set → "Load more" appears; clicking it appends page 2.
  const loadMore = page.getByRole("button", { name: /load more/i });
  await expect(loadMore).toBeVisible();
  await loadMore.click();

  await expect(page.getByText("service.stop")).toBeVisible();
  await expect(page.getByText("auth.denied")).toBeVisible();
  await expect(page.getByText("denied", { exact: true })).toBeVisible();
  // Page 2 exhausts the log → the button disappears.
  await expect(page.getByRole("button", { name: /load more/i })).toHaveCount(0);
});

test("non-admin token gets the friendly admin-required state, not an error", async ({
  page
}) => {
  await mockApi(page, { auditForbidden: true });
  await loginAsToken(page);

  await page.goto("/activity");
  await expect(page.getByText(/admin token required/i)).toBeVisible();
  await expect(
    page.getByText(/sign in with an admin token/i)
  ).toBeVisible();
  // The 403 is surfaced as data, never as an error banner or a retry loop.
  await expect(page.getByText(/could not reach/i)).toHaveCount(0);
});

test("live stream prepends a new row without a refetch", async ({ page }) => {
  await mockApi(page);
  // Emit one audit event on the stream; this action is NOT in the polled pages,
  // so seeing it proves the row arrived live.
  await page.unroute("**/api/audit/stream");
  await page.route("**/api/audit/stream", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "cache-control": "no-cache" },
      body:
        "event: audit.secrets.rotate_key\n" +
        'data: {"type":"audit.secrets.rotate_key","ts":"2026-07-12T10:00:00+00:00",' +
        '"action":"secrets.rotate_key","result":"ok","status_code":200,' +
        '"principal_id":"tok-live-01","principal_role":"admin","target_type":"secret",' +
        '"target_id":"shared","params":null,"request_id":"req-live-1"}\n\n'
    })
  );
  await loginAsToken(page);

  await page.goto("/activity");

  await expect(page.getByText("secrets.rotate_key")).toBeVisible();
  await expect(page.getByText("tok-live-01").first()).toBeVisible();
  await expect(page.getByText("req-live-1")).toBeVisible();
});

test("polling still renders when the live stream is down", async ({ page }) => {
  await mockApi(page);
  await page.unroute("**/api/audit/stream");
  await page.route("**/api/audit/stream", (route) =>
    route.fulfill({ status: 500, contentType: "application/json", body: "{}" })
  );
  await loginAsToken(page);

  await page.goto("/activity");
  // The polled first page still renders despite the dead stream.
  await expect(page.getByText("deploy.create")).toBeVisible();
  await expect(page.getByText("secrets.set")).toBeVisible();
});

test("activity copy uses no em dashes", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/activity");
  await expect(page.getByText("deploy.create")).toBeVisible();

  const body = await page.locator("body").innerText();
  expect(body).not.toContain("—");
});
