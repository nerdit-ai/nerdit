import { expect, test, type Page, type Request } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

// Settings Disk & GC card (P23 WP5) + Routes card (P23 WP6).
//
// Disk & GC — two things are load-bearing here:
//   * a `docker: null` report renders "unavailable", never a row of zeros, and
//     a timed-out walk (null bytes + a `scan_timeout` warning) renders
//     "unknown", never `0`;
//   * gc is plan-first (R4) — no real (non-dry-run) `POST /system/gc` may fire
//     until the destructive confirm dialog is accepted.
//
// Routes (D-P23-7) — the route tri-state must never be rendered by truthiness,
// a null `live` must read as unknown (not "missing"), and the bare route count
// deleted from `ProxyStatusCard` must stay deleted (a revert would restate the
// same fact in two places).

const DISK_REPORT = {
  docker: {
    images_bytes: 5_368_709_120,
    containers_bytes: 1_048_576,
    volumes_bytes: 104_857_600,
    build_cache_bytes: 2_147_483_648
  },
  images: {
    total_bytes: 5_368_709_120,
    by_repo: { "nerdit-app/my-app": 4_294_967_296, "nerdit-app/stale": 1_073_741_824 }
  },
  data_dir: {
    services: [{ name: "my-app", bytes: 12_582_912 }],
    services_total_bytes: 12_582_912,
    models: { ollama: 4_294_967_296, huggingface: 0 },
    archive_bytes: 0,
    backups: { bytes: 2_097_152, count: 2 },
    volume_backups: { bytes: 1_048_576, count: 1 },
    dumps: { bytes: 3_145_728, count: 3 },
    dump_staging_bytes: 0
  },
  orphan_images: ["nerdit-app/stale"],
  orphan_data_dirs: ["gone-app"],
  warnings: []
};

// docker df unreadable + a bounded walk that overran its budget: every one of
// these values must render as words, not as zeros.
const DISK_REPORT_DEGRADED = {
  ...DISK_REPORT,
  docker: null,
  images: { total_bytes: null, by_repo: {} },
  data_dir: {
    services: [],
    services_total_bytes: null,
    models: { ollama: null, huggingface: null },
    archive_bytes: null,
    backups: { bytes: 0, count: 0 },
    volume_backups: { bytes: 0, count: 0 },
    dumps: { bytes: 0, count: 0 },
    dump_staging_bytes: null
  },
  warnings: ["scan_timeout"]
};

function gcResult(dryRun: boolean, includeOrphanData: boolean) {
  return {
    dry_run: dryRun,
    images: {
      removed: ["nerdit-app/stale"],
      skipped: [{ repo: "nerdit-app/busy", reason: "in_use_or_error" }],
      reclaimed_bytes_estimate: 1_073_741_824
    },
    orphan_data: {
      enabled: includeOrphanData,
      removed: includeOrphanData ? ["gone-app"] : [],
      skipped: []
    },
    reports: {
      weights: { ollama: 0, huggingface: 0 },
      build_cache_bytes: 0,
      backups_over_keep: 0
    },
    warnings: []
  };
}

// A routed service, a subdomain-shaped route (`""` — a real route, not a
// missing one), and an unrouted model (`route: null`, by design). `live` is
// null on every row because this page's `live_table` is "unreadable".
const ROUTES_PAGE = {
  items: [
    {
      service_name: "my-app",
      kind: "service",
      status: "running",
      host_port: 31001,
      container_port: 8000,
      protocol: "tcp",
      route: "/my-app",
      public_url: "https://test-host.local/my-app/",
      live: null
    },
    {
      service_name: "sub-app",
      kind: "service",
      status: "running",
      host_port: 31002,
      container_port: 8000,
      protocol: "tcp",
      route: "",
      public_url: "https://sub-app.test-host.local/",
      live: null
    },
    {
      service_name: "ollama-llama3",
      kind: "model",
      status: "running",
      host_port: 31100,
      container_port: 11434,
      protocol: "tcp",
      route: null,
      public_url: null,
      live: null
    }
  ],
  next_cursor: null,
  live_table: "unreadable"
};

/** `GET /api/routes` stub, registered after `mockApi` (last-registered wins). */
async function mockRoutes(page: Page, body: unknown = ROUTES_PAGE): Promise<void> {
  await page.route(/\/api\/routes(\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body)
    })
  );
}

/**
 * A manually released gate. Used to hold a real (non-dry-run) gc in flight for
 * as long as the assertions need, without a wall-clock delay that could race a
 * slow CI box either way.
 */
function gate(): { wait: Promise<void>; release: () => void } {
  let release = () => {};
  const wait = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { wait, release: () => release() };
}

/**
 * `/api/system/*` stubs, registered after `mockApi` (last-registered wins).
 * `holdRealRun`, when given, parks every non-dry-run POST until it resolves.
 */
async function mockSystem(
  page: Page,
  report: unknown = DISK_REPORT,
  holdRealRun?: Promise<void>
): Promise<void> {
  await page.route(/\/api\/system\/disk(\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(report)
    })
  );
  await page.route(/\/api\/system\/gc(\?.*)?$/, async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const url = new URL(route.request().url());
    const body = route.request().postDataJSON() as { include_orphan_data?: boolean } | null;
    const dryRun = url.searchParams.get("dry_run") === "true";
    if (!dryRun && holdRealRun) await holdRealRun;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(gcResult(dryRun, Boolean(body?.include_orphan_data)))
    });
  });
}

/** Every `POST /api/system/gc`, so the spec can assert on the dry_run split. */
function gcPosts(page: Page): Request[] {
  const posts: Request[] = [];
  page.on("request", (req) => {
    if (req.method() === "POST" && /\/api\/system\/gc/.test(req.url())) posts.push(req);
  });
  return posts;
}

async function gotoSettings(page: Page) {
  await loginAsToken(page);
  await page.goto("/settings");
}

/** The rows of the routes table. `TableRow` carries no testid of its own. */
function routeRows(page: Page) {
  return page.getByTestId("routes-card").locator("tbody tr");
}

/** The data-dir opt-in is a pressed-state toggle button, not a checkbox. */
function orphanDataToggle(page: Page) {
  return page.getByTestId("gc-orphan-data");
}

test("disk card renders the report", async ({ page }) => {
  await mockApi(page);
  await mockSystem(page);
  await gotoSettings(page);

  await expect(page.getByTestId("disk-card")).toBeVisible();
  // docker aggregate (same unit ladder as `nerdit disk`).
  await expect(page.getByText("5.0 GiB").first()).toBeVisible();
  await expect(page.getByText("2.0 GiB").first()).toBeVisible();
  // data-dir trees, incl. the volume-backup bucket beside the control-plane one.
  await expect(page.getByText("services/my-app")).toBeVisible();
  await expect(page.getByText("volume backups")).toBeVisible();
  // P37 dump tars + the staging tree beside them.
  await expect(page.getByText("3.0 MiB").first()).toBeVisible();
  await expect(page.getByText("dump staging")).toBeVisible();
  await expect(page.getByText("1.0 MiB (1 files)")).toBeVisible();
  // orphans are named — they are the actionable half of the report.
  await expect(page.getByText("nerdit-app/stale").first()).toBeVisible();
  await expect(page.getByText("gone-app").first()).toBeVisible();
});

test("a docker-less report says unavailable and timed-out walks say unknown", async ({ page }) => {
  await mockApi(page);
  await mockSystem(page, DISK_REPORT_DEGRADED);
  await gotoSettings(page);

  await expect(page.getByTestId("disk-docker-unavailable")).toBeVisible();
  await expect(page.getByTestId("disk-warning")).toHaveText("warning: scan_timeout");
  // Null walk values render as words; the card must show no bare "0 B".
  await expect(page.getByText("unknown").first()).toBeVisible();
  await expect(page.getByText("0 B", { exact: true })).toHaveCount(0);
});

test("gc is plan-first: no real run fires before the dialog is confirmed", async ({ page }) => {
  await mockApi(page);
  await mockSystem(page);
  const posts = gcPosts(page);
  await gotoSettings(page);

  // The real-run button is inert until a plan exists.
  const run = page.getByTestId("gc-run");
  await expect(run).toBeDisabled();

  await page.getByTestId("gc-dry-run").click();

  await expect(page.getByTestId("gc-report")).toBeVisible();
  await expect(page.getByTestId("gc-report").getByText("Preview")).toBeVisible();
  await expect(page.getByText(/images would remove/)).toBeVisible();
  await expect(page.getByText(/image skipped nerdit-app\/busy \(in_use_or_error\)/)).toBeVisible();
  await expect(page.getByText("1.0 GiB")).toBeVisible();

  // Exactly one POST so far and it was the dry run — with no key claimed.
  await expect.poll(() => posts.length).toBe(1);
  expect(posts[0].url()).toContain("dry_run=true");
  expect(posts[0].headers()["idempotency-key"]).toBeUndefined();

  // Opening the dialog is still not a run.
  await run.click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByText("Run garbage collection")).toBeVisible();
  expect(posts.filter((p) => !p.url().includes("dry_run=true"))).toHaveLength(0);

  // Cancelling is not a run either.
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toHaveCount(0);
  expect(posts.filter((p) => !p.url().includes("dry_run=true"))).toHaveLength(0);

  // Only the confirm fires the real run — with a minted Idempotency-Key.
  await run.click();
  await page.getByRole("dialog").getByRole("button", { name: "Collect" }).click();

  await expect.poll(() => posts.some((p) => !p.url().includes("dry_run=true"))).toBe(true);
  const real = posts.find((p) => !p.url().includes("dry_run=true"))!;
  expect(real.headers()["idempotency-key"]).toBeTruthy();
  await expect(page.getByText("Result")).toBeVisible();
  await expect(page.getByText(/images removed/)).toBeVisible();
});

test("the IRREVERSIBLE data-dir opt-in is carried into the request and the dialog", async ({
  page
}) => {
  await mockApi(page);
  await mockSystem(page);
  const posts = gcPosts(page);
  await gotoSettings(page);

  await orphanDataToggle(page).click();
  await expect(orphanDataToggle(page)).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("gc-dry-run").click();

  await expect(page.getByText(/data dirs would remove/)).toBeVisible();
  expect(posts[0].postDataJSON()).toEqual({ include_orphan_data: true });

  await page.getByTestId("gc-run").click();
  await expect(page.getByRole("dialog").getByText(/This cannot be undone/)).toBeVisible();
});

test("widening the scope after a preview invalidates the plan (R4)", async ({ page }) => {
  await mockApi(page);
  await mockSystem(page);
  const posts = gcPosts(page);
  await gotoSettings(page);

  const run = page.getByTestId("gc-run");
  const orphanData = orphanDataToggle(page);

  // 1. Preview with the box UNCHECKED: an images-only plan, no data dirs.
  await page.getByTestId("gc-dry-run").click();
  await expect(page.getByTestId("gc-report")).toBeVisible();
  await expect(page.getByText(/data dirs would remove/)).toHaveCount(0);
  await expect(run).toBeEnabled();

  // 2. Widen the scope AFTER the preview. The plan you see is the run you
  //    confirm, so the rendered plan must go and the button must re-disable —
  //    otherwise this click would arm an IRREVERSIBLE data-dir deletion that
  //    no plan ever showed.
  await orphanData.click();
  await expect(page.getByTestId("gc-report")).toHaveCount(0);
  await expect(run).toBeDisabled();

  // The disabled button cannot even open the dialog, so no real run can fire.
  await run.click({ force: true });
  await expect(page.getByRole("dialog")).toHaveCount(0);
  expect(posts.filter((p) => !p.url().includes("dry_run=true"))).toHaveLength(0);

  // 3. A FRESH dry run with the new scope re-enables the button...
  await page.getByTestId("gc-dry-run").click();
  await expect(page.getByText(/data dirs would remove/)).toBeVisible();
  await expect(run).toBeEnabled();
  await expect.poll(() => posts.length).toBe(2);
  expect(posts[1].postDataJSON()).toEqual({ include_orphan_data: true });

  // ...and the real run carries the PREVIEWED scope.
  await run.click();
  await page.getByRole("dialog").getByRole("button", { name: "Collect" }).click();
  await expect.poll(() => posts.some((p) => !p.url().includes("dry_run=true"))).toBe(true);
  const real = posts.find((p) => !p.url().includes("dry_run=true"))!;
  expect(real.postDataJSON()).toEqual({ include_orphan_data: true });
});

test("narrowing the scope after a preview also invalidates the plan (R4)", async ({ page }) => {
  await mockApi(page);
  await mockSystem(page);
  const posts = gcPosts(page);
  await gotoSettings(page);

  const run = page.getByTestId("gc-run");
  const orphanData = orphanDataToggle(page);

  await orphanData.click();
  await page.getByTestId("gc-dry-run").click();
  await expect(page.getByText(/data dirs would remove/)).toBeVisible();

  // Unchecking is a scope change too: same reset, same re-disable.
  await orphanData.click();
  await expect(page.getByTestId("gc-report")).toHaveCount(0);
  await expect(run).toBeDisabled();

  await page.getByTestId("gc-dry-run").click();
  await expect(page.getByTestId("gc-report")).toBeVisible();
  await expect(page.getByText(/data dirs would remove/)).toHaveCount(0);

  await run.click();
  // The dialog describes the PREVIEWED scope, so it must not threaten data dirs.
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByText(/Data dirs are left untouched/)).toBeVisible();
  await dialog.getByRole("button", { name: "Collect" }).click();

  await expect.poll(() => posts.some((p) => !p.url().includes("dry_run=true"))).toBe(true);
  const real = posts.find((p) => !p.url().includes("dry_run=true"))!;
  expect(real.postDataJSON()).toEqual({ include_orphan_data: false });
});

test("a scope toggle cannot detach the UI from an in-flight real run", async ({ page }) => {
  const held = gate();
  await mockApi(page);
  await mockSystem(page, DISK_REPORT, held.wait);
  const posts = gcPosts(page);
  await gotoSettings(page);

  // Locate by testid, not by name: the run button's label becomes "Collecting…"
  // while it is pending, so a name-based locator would stop matching exactly
  // when this test needs it.
  const run = page.getByTestId("gc-run");
  const dryRun = page.getByTestId("gc-dry-run");
  const orphanData = orphanDataToggle(page);

  await dryRun.click();
  await expect(page.getByTestId("gc-report")).toBeVisible();

  await run.click();
  await page.getByRole("dialog").getByRole("button", { name: "Collect" }).click();

  // The destructive run is now in flight (and stays there until we release it).
  await expect(run).toHaveText("Collecting…");
  // The two controls that would reset `runGc` — and so orphan a deletion that
  // is still running server-side — are inert for the duration.
  await expect(orphanData).toBeDisabled();
  await expect(dryRun).toBeDisabled();
  await orphanData.click({ force: true });
  await expect(orphanData).toHaveAttribute("aria-pressed", "false");
  await expect(run).toHaveText("Collecting…");

  // The pending run renders ITS OWN outcome when it settles, and it was the
  // only real run — no second one was armed behind it.
  held.release();
  await expect(page.getByText("Result")).toBeVisible();
  await expect(page.getByText(/images removed/)).toBeVisible();
  await expect(orphanData).toBeEnabled();
  expect(posts.filter((p) => !p.url().includes("dry_run=true"))).toHaveLength(1);
});

test("a non-admin sees the report but no garbage-collection controls", async ({ page }) => {
  await mockApi(page, { authRole: "readonly" });
  await mockSystem(page);
  await gotoSettings(page);

  await expect(page.getByText("services/my-app")).toBeVisible();
  await expect(page.getByTestId("gc-dry-run")).toHaveCount(0);
  await expect(page.getByTestId("gc-run")).toHaveCount(0);
});

// ---------------------------------------------------------------------------
// Routes card (WP6, D-P23-7)
// ---------------------------------------------------------------------------

test("the routes card renders the tri-state: path, subdomain, unrouted model", async ({ page }) => {
  await mockApi(page);
  await mockSystem(page);
  await mockRoutes(page);
  await gotoSettings(page);

  await expect(page.getByTestId("routes-card")).toBeVisible();
  const rows = routeRows(page);
  await expect(rows).toHaveCount(3);

  // A path route renders literally, with its public_url as a real anchor.
  const app = rows.filter({ hasText: "my-app" }).first();
  await expect(app.getByText("/my-app", { exact: true })).toBeVisible();
  await expect(app.getByRole("link", { name: "https://test-host.local/my-app/" })).toHaveAttribute(
    "href",
    "https://test-host.local/my-app/"
  );

  // `route === ""` is a REAL route (subdomain shape), never "unrouted" — this
  // is the assertion a truthiness check would fail.
  const sub = rows.filter({ hasText: "sub-app" });
  await expect(sub.getByText("(subdomain)")).toBeVisible();
  await expect(sub.getByText("unrouted")).toHaveCount(0);

  // A model is unrouted by design, with no public URL.
  const model = rows.filter({ hasText: "ollama-llama3" });
  await expect(model.getByText("unrouted")).toBeVisible();
  await expect(model.getByRole("link")).toHaveCount(0);
});

test("an unreadable live table is explained by the page-level caption", async ({ page }) => {
  await mockApi(page);
  await mockSystem(page);
  await mockRoutes(page);
  await gotoSettings(page);

  await expect(page.getByTestId("routes-live-table")).toHaveText(/unreadable/);
  // Every `live: null` row reads as unknown, never as "missing".
  await expect(routeRows(page).first().getByText("missing")).toHaveCount(0);
});

test("the proxy card no longer restates the route count (D-P23-7 deletion)", async ({ page }) => {
  await mockApi(page);
  await mockSystem(page);
  await mockRoutes(page);
  await gotoSettings(page);

  // The card is loaded (a sibling row proves it) but carries no "Routes" row:
  // the count lives in exactly one place, the routes table itself.
  const proxyCard = page.getByTestId("proxy-card");
  await expect(proxyCard.getByText("TLS synced")).toBeVisible();
  await expect(proxyCard.getByText("Routes", { exact: true })).toHaveCount(0);
});
