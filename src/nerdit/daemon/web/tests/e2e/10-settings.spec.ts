import { expect, test, type Page, type Request } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

// Settings is the machine page (W3 P4): appearance + analytics disclosure,
// the admin hostname editor (compare-and-set round-trip, dry-run preview,
// Idempotency-Key on real writes), the GPU card folded out of the retired
// Hardware page, and the doctor / proxy readouts.

async function gotoSettings(page: Page) {
  await loginAsToken(page);
  // The apps list is the landing page now; "/projects" only redirects to it.
  await expect(page).toHaveURL("/");
  await page.goto("/settings");
}

function proxyConfigPuts(page: Page): Request[] {
  const puts: Request[] = [];
  page.on("request", (req) => {
    if (req.method() === "PUT" && /\/api\/config\/daemon\/proxy/.test(req.url())) {
      puts.push(req);
    }
  });
  return puts;
}

test("doctor card renders the worst-of status and every check", async ({ page }) => {
  await mockApi(page);
  await gotoSettings(page);

  // Worst-of status badge is "warn" (config_restart_pending is warn).
  await expect(page.getByTestId("doctor-status")).toHaveText("warn");
  await expect(page.getByTestId("doctor-check")).toHaveCount(9);
  await expect(page.getByText("config_restart_pending")).toBeVisible();
  await expect(page.getByText("pending restart: proxy.hostname_override")).toBeVisible();
  // A skipped check renders without being flagged as a problem.
  await expect(page.getByText("mDNS advertising disabled")).toBeVisible();
});

test("proxy status card lists the live facts", async ({ page }) => {
  await mockApi(page);
  await gotoSettings(page);

  const proxy = page.getByTestId("proxy-card");
  await expect(proxy.getByText("SHA256:ab12cd34ef56")).toBeVisible();
  await expect(proxy.getByText("test-host.local")).toBeVisible();
  await expect(proxy.getByText("TLS synced")).toBeVisible();
});

test("the GPU card lists each card with its memory and utilization", async ({ page }) => {
  await mockApi(page);
  await gotoSettings(page);

  const rows = page.getByTestId("gpu-row");
  await expect(rows).toHaveCount(2);
  await expect(rows.first()).toContainText("NVIDIA A100 80GB");
  await expect(rows.first()).toContainText("12% busy");
  // Memory as a meter with the exact figures beside it (4096 / 81920 MB).
  await expect(rows.first()).toContainText("4.0 GB / 80.0 GB");
  // The GPU-lab framing is retired: no vendor badge, no temperature.
  await expect(rows.first()).not.toContainText("42");
  await expect(page.getByText("nvidia", { exact: true })).toHaveCount(0);
});

test("no GPUs reads as a plain CPU statement, not an error", async ({ page }) => {
  await mockApi(page, { gpus: [] });
  await gotoSettings(page);

  await expect(page.getByTestId("gpu-empty")).toHaveText("No GPUs detected. Apps run on CPU.");
});

test("the theme control stamps the document and remembers the choice", async ({ page }) => {
  await mockApi(page);
  await gotoSettings(page);

  const toggle = page.getByTestId("theme-toggle");
  await expect(toggle).toBeVisible();
  // Default is "system": nothing stamped on <html>.
  await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.*/);
  await expect(page.getByTestId("theme-system")).toHaveAttribute("aria-pressed", "true");

  await page.getByTestId("theme-dark").click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByTestId("theme-dark")).toHaveAttribute("aria-pressed", "true");

  await page.getByTestId("theme-light").click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");

  // The choice survives a reload (it is stored per browser).
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.getByTestId("theme-light")).toHaveAttribute("aria-pressed", "true");

  // Back to system: the stamp is removed, the OS decides again.
  await page.getByTestId("theme-system").click();
  await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.*/);
});

test("analytics disclosure reads off when the daemon configured no key", async ({ page }) => {
  await mockApi(page);
  await gotoSettings(page);

  await expect(page.getByTestId("analytics-disclosure")).toHaveText(
    "Off. Enable via [posthog] in the daemon config."
  );
});

test("analytics disclosure reads active when the daemon configured one", async ({ page }) => {
  await mockApi(page);
  // Re-stub /cluster/info AFTER mockApi (last registered wins) — the fixture's
  // sample carries no posthog fields, and fixtures.ts is not this wave's file.
  await page.route("**/api/cluster/info", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        hostname: "test-host",
        version: "0.3.0",
        uptime_seconds: 120,
        posthog_key: "phc_test",
        posthog_host: "https://posthog.invalid"
      })
    })
  );
  // The SPA initialises the tracker from those fields; keep its traffic off-box.
  await page.route("https://posthog.invalid/**", (route) => route.abort());
  await gotoSettings(page);

  await expect(page.getByTestId("analytics-disclosure")).toContainText("Active. Page views only");
  await expect(page.getByTestId("analytics-disclosure")).toContainText("[posthog]");
});

test("dry-run preview shows the diff and restart notice with no real write", async ({ page }) => {
  await mockApi(page);
  const puts = proxyConfigPuts(page);
  await gotoSettings(page);

  await page.getByTestId("hostname-input").fill("my-box.local");
  await page.getByTestId("hostname-preview").click();

  await expect(page.getByTestId("hostname-preview-diff")).toBeVisible();
  await expect(page.getByText("Applies after restart.")).toBeVisible();
  await expect(page.getByText(/hostname_override/).first()).toBeVisible();

  // Exactly one PUT so far, and it was the dry run (no key/If-Match).
  await expect.poll(() => puts.length).toBe(1);
  expect(puts[0].url()).toContain("dry_run=true");
  expect(puts[0].headers()["idempotency-key"]).toBeUndefined();
  expect(puts[0].headers()["if-match"]).toBeUndefined();
});

test("apply sends If-Match + Idempotency-Key", async ({ page }) => {
  await mockApi(page);
  const puts = proxyConfigPuts(page);
  await gotoSettings(page);

  await page.getByTestId("hostname-input").fill("my-box.local");
  await page.getByTestId("hostname-apply").click();

  await expect.poll(() => puts.some((p) => !p.url().includes("dry_run=true"))).toBe(true);
  const real = puts.find((p) => !p.url().includes("dry_run=true"))!;
  expect(real.headers()["if-match"]).toBe("proxy-config-etag-1");
  expect(real.headers()["idempotency-key"]).toBeTruthy();
});

test("hostname writes stay disabled until the config loads", async ({ page }) => {
  await mockApi(page);
  // Hold the proxy-config GET so the card renders before its value + version.
  let releaseConfig!: () => void;
  const gate = new Promise<void>((resolve) => (releaseConfig = resolve));
  await page.route(/\/api\/config\/daemon\/proxy$/, async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    await gate;
    await route.fallback();
  });
  await gotoSettings(page);

  const preview = page.getByTestId("hostname-preview");
  const apply = page.getByTestId("hostname-apply");
  await expect(preview).toBeDisabled();
  await expect(apply).toBeDisabled();

  releaseConfig();
  await expect(preview).toBeEnabled();
  // Still pristine: Apply needs a dirty value.
  await expect(apply).toBeDisabled();
  await page.getByTestId("hostname-input").fill("my-box.local");
  await expect(apply).toBeEnabled();
});

test("non-admin sees no hostname card and no restart button", async ({ page }) => {
  await mockApi(page, { authRole: "readonly" });
  await gotoSettings(page);

  // The page still renders its read-only halves.
  await expect(page.getByTestId("doctor-status")).toBeVisible();
  // The hostname editor (input + Preview/Apply + restart) is the admin surface.
  await expect(page.getByTestId("hostname-card")).toHaveCount(0);
  await expect(page.getByTestId("hostname-input")).toHaveCount(0);
  await expect(page.getByTestId("daemon-restart")).toHaveCount(0);
});
