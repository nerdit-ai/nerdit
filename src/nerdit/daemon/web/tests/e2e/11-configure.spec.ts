import { expect, test } from "@playwright/test";
import { SAMPLE_SERVICES, loginAsToken, mockApi } from "./fixtures";

// Configure is no longer a header action: it opens from the app page's Manage
// tab (W4 — the header verb is always Deploy). The diagnose assertions stay on
// Overview, where the panel still auto-expands for an unhealthy app.

test("configure panel exposes read-only name/port with a reason", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  await page.getByRole("button", { name: /^configure$/i }).click();

  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("heading", { name: /configure my-app/i })).toBeVisible();

  // name/port render as read-only with a one-line reason, not as inputs.
  await expect(dialog.getByText("Fixed at deploy. Redeploy to rename.")).toBeVisible();
  await expect(dialog.getByText("Fixed at deploy time.")).toBeVisible();

  // The editable GPUs field can be typed into.
  const gpus = dialog.getByPlaceholder("0");
  await gpus.fill("2");
  await expect(gpus).toHaveValue("2");
});

test("preview runs a dry-run write only, apply round-trips the ETag", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  await page.getByRole("button", { name: /^configure$/i }).click();
  const dialog = page.getByRole("dialog");

  // Make the deploy section dirty so Preview/Apply enable.
  await dialog.getByPlaceholder("0").fill("2");

  // Preview → a dry_run PUT with no If-Match header, and a preview note.
  const dryReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/deploy") &&
      r.url().includes("dry_run=true")
  );
  await dialog.getByRole("button", { name: /^preview$/i }).first().click();
  const dry = await dryReq;
  expect(dry.headers()["if-match"]).toBeUndefined();
  // The note names the pending edit (W5: the changed keys render as rows) and
  // still says plainly that nothing was written.
  const preview = dialog.getByTestId("config-preview");
  await expect(preview).toBeVisible();
  await expect(preview).toContainText(/preview only, nothing saved/i);
  await expect(preview).toContainText("gpus");

  // Apply → a real PUT that carries the ETag from the GET as If-Match.
  const realReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/deploy") &&
      !r.url().includes("dry_run")
  );
  await dialog.getByRole("button", { name: /^apply$/i }).first().click();
  const real = await realReq;
  expect(real.headers()["if-match"]).toBe("app-config-etag-1");
  await expect(page.getByText(/deploy config saved/i)).toBeVisible();
});

test("a stale 409 surfaces a review-and-retry message", async ({ page }) => {
  await mockApi(page);
  // Override the section PUT to reject with the structured stale envelope.
  await page.route(/\/api\/config\/apps\/[^/?]+\/[^/?]+(\?.*)?$/, (route) => {
    if (route.request().method() !== "PUT") return route.fallback();
    const url = new URL(route.request().url());
    if (url.searchParams.get("dry_run") === "true") return route.fallback();
    return route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({
        code: "config.stale",
        message: "Config changed since the ETag was read",
        detail: "Config changed since the ETag was read"
      })
    });
  });
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  await page.getByRole("button", { name: /^configure$/i }).click();
  const dialog = page.getByRole("dialog");

  await dialog.getByPlaceholder("0").fill("2");
  await dialog.getByRole("button", { name: /^apply$/i }).first().click();

  await expect(dialog.getByText(/settings changed elsewhere\. review and retry/i)).toBeVisible();
});

test("diagnose panel shows remediation for a failed service and never leaks a value", async ({
  page
}) => {
  const failed = {
    items: [
      {
        ...SAMPLE_SERVICES.items[0],
        status: "failed",
        last_deploy: { version: 4, action: "redeploy", phase: "failed", reason: "container exited (1)" }
      }
    ],
    next_cursor: null
  };
  await mockApi(page, {
    services: failed,
    diagnose: {
      service_name: "my-app",
      kind: "service",
      status: "failed",
      desired_state: "running",
      last_deploy: { version: 4, action: "redeploy", phase: "failed", reason: "container exited (1)" },
      error: { class: "USER_ERROR", message: "start command not found: npm" },
      forensics: { last_exit_code: 1, oom_killed: false, last_crash_at: "2026-07-04T11:30:00+00:00" },
      restarts: {
        policy: "always",
        count: 3,
        max_restarts: 3,
        window_seconds: 300,
        window_start: "2026-07-04T11:25:00+00:00",
        last_exit_at: "2026-07-04T11:30:00+00:00",
        backoff_s: 30,
        next_retry_in_s: null
      },
      health: { spec: { path: "/healthz" }, probe: null },
      bindings: { waiting: false, messages: [] },
      build: { version: 4, last_result: "failed", reason: "container exited (1)" },
      injected_env_keys: ["LOG_LEVEL", "OPENAI_API_KEY"],
      injected_env_keys_source: "launch",
      pending_env_keys: ["OPENAI_API_KEY"],
      logs: [{ stream: "stderr", line: "sh: npm: not found", ts: "2026-07-04T11:30:00+00:00" }],
      remediation: { code: "fix_start_command", detail: "Fix the start command in [deploy].start." },
      // Trap: a secret value must never be rendered by the panel.
      secret_value_trap: "sk-live-DONOTLEAK"
    }
  });
  await loginAsToken(page);

  await page.goto("/projects/my-app");

  // Auto-expanded for a failed service: the remediation headline is visible.
  await expect(page.getByText("Fix the start command", { exact: true })).toBeVisible();
  await expect(page.getByText(/USER_ERROR/)).toBeVisible();
  // Env key NAMES render; the trap secret value never appears anywhere.
  await expect(page.getByTestId("diagnose-panel").getByText("OPENAI_API_KEY")).toBeVisible();
  await expect(page.getByText("sk-live-DONOTLEAK")).toHaveCount(0);
});
