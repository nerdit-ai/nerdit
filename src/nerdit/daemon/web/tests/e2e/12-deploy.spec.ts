import { expect, test } from "@playwright/test";
import type { Route } from "@playwright/test";
import { SAMPLE_SERVICES, loginAsToken, makeWaitResponse, mockApi } from "./fixtures";

function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function openDeploy(page: import("@playwright/test").Page) {
  // The deploy dialog opens from the apps list's "New app" menu (dashboard
  // redesign W3); the dialog's own internals are unchanged.
  await page.goto("/");
  await page.getByRole("button", { name: /new app/i }).click();
  await page.getByRole("menuitem", { name: /deploy a folder/i }).click();
  await expect(page.getByTestId("deploy-dialog")).toBeVisible();
}

test("git tab posts to /deploy/git with the entered body", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);
  await openDeploy(page);

  await page.getByRole("button", { name: "Git", exact: true }).click();
  await page.getByPlaceholder("https://github.com/owner/repo").fill(
    "https://github.com/nerdit-ai/demo"
  );
  await page.getByPlaceholder("main").fill("v1.0.0");
  await page.getByPlaceholder("my-app").fill("demo");

  const [request] = await Promise.all([
    page.waitForRequest(
      (req) =>
        req.url().includes("/api/deploy/git") &&
        req.method() === "POST" &&
        !req.url().includes("dry_run")
    ),
    page.getByRole("dialog").getByRole("button", { name: "Deploy", exact: true }).click()
  ]);
  const body = request.postDataJSON();
  expect(body.repo_url).toBe("https://github.com/nerdit-ai/demo");
  expect(body.name).toBe("demo");
  expect(body.ref).toBe("v1.0.0");
  expect(request.headers()["idempotency-key"]).toBeTruthy();

  // Deploy succeeds → the dialog swaps to the progress view.
  await expect(page.getByText(/deploying/i)).toBeVisible();
});

test("the overrides live behind a collapsed Advanced fold", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);
  await openDeploy(page);

  // Name and the source chooser are the visible form (design guidelines §4);
  // nerdit.toml supplies the rest, so the overrides start folded away.
  await expect(page.getByTestId("deploy-name")).toBeVisible();
  await expect(page.getByTestId("deploy-source-tabs")).toBeVisible();
  const advanced = page.getByTestId("deploy-advanced");
  await expect(advanced).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByTestId("deploy-port")).toBeHidden();
  await expect(page.getByTestId("deploy-env-add")).toBeHidden();

  await advanced.click();
  await expect(advanced).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByTestId("deploy-port")).toBeVisible();
  await expect(page.getByTestId("deploy-gpus")).toBeVisible();
  await expect(page.getByTestId("deploy-start")).toBeVisible();
  await expect(page.getByTestId("deploy-health")).toBeVisible();
  await expect(page.getByTestId("deploy-env-add")).toBeVisible();

  // The overrides still reach the request when the operator sets them.
  await page.getByTestId("deploy-port").fill("8080");
  await page.getByTestId("deploy-gpus").fill("1");
  await page.getByTestId("deploy-start").fill("npm run serve");
  await page
    .locator('input[type="file"]')
    .setInputFiles({ name: "app.zip", mimeType: "application/zip", buffer: Buffer.from("PK") });
  await page.getByTestId("deploy-name").fill("my-app");

  const [request] = await Promise.all([
    page.waitForRequest((req) => req.url().endsWith("/api/deploy") && req.method() === "POST"),
    page.getByTestId("deploy-submit").click()
  ]);
  const form = request.postData() ?? "";
  expect(form).toContain("8080");
  expect(form).toContain("npm run serve");
});

test("preview renders the dry-run plan without deploying", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);
  await openDeploy(page);

  await page.getByRole("button", { name: "Git", exact: true }).click();
  await page.getByPlaceholder("https://github.com/owner/repo").fill(
    "https://github.com/nerdit-ai/demo"
  );
  await page.getByPlaceholder("my-app").fill("demo");

  let realDeploy = false;
  page.on("request", (req) => {
    if (req.url().includes("/api/deploy/git") && !req.url().includes("dry_run")) realDeploy = true;
  });

  await page.getByRole("button", { name: "Preview", exact: true }).click();

  // The plan summary renders (fixture dry-run = create via node).
  await expect(page.getByText(/plan:\s*create via node/i)).toBeVisible();
  // No real deploy fired, and the form is still shown (no progress view).
  await expect(page.getByText(/deploying/i)).toHaveCount(0);
  expect(realDeploy).toBe(false);
});

test("progress walks building to healthy on the mocked wait", async ({ page }) => {
  const service = {
    ...SAMPLE_SERVICES.items[0],
    name: "demo",
    last_deploy: { version: 7, action: "create", phase: "queued" }
  };
  await mockApi(page, { deployGitResponse: { status: 201, body: service } });
  await loginAsToken(page);

  // Sequence the wait: building first (keeps polling), then converged healthy.
  let calls = 0;
  await page.route(/\/api\/services\/[^/]+\/wait(\?.*)?$/, (route) => {
    calls += 1;
    if (calls === 1) {
      return json(
        route,
        makeWaitResponse({
          outcome: "timeout",
          service_name: "demo",
          version: 7,
          phase: "building",
          status: "building",
          public_url: null,
          waited_s: 10
        })
      );
    }
    return json(
      route,
      makeWaitResponse({ service_name: "demo", version: 7, phase: "healthy" })
    );
  });

  await openDeploy(page);
  await page.getByRole("button", { name: "Git", exact: true }).click();
  await page.getByPlaceholder("https://github.com/owner/repo").fill("https://github.com/x/y");
  await page.getByPlaceholder("my-app").fill("demo");
  await page.getByRole("dialog").getByRole("button", { name: "Deploy", exact: true }).click();

  const dialog = page.getByRole("dialog");
  await expect(dialog.getByText("Building", { exact: true })).toBeVisible();
  await expect(dialog.getByText(/app is live/i)).toBeVisible();
});

test("a failed outcome links to the service page", async ({ page }) => {
  await mockApi(page, {
    waitResponse: makeWaitResponse({
      outcome: "failed",
      phase: "failed",
      status: "running",
      public_url: null,
      reason: "container exited (1)"
    })
  });
  await loginAsToken(page);
  await openDeploy(page);

  // ZIP path: attach an archive, then deploy.
  await page
    .locator('input[type="file"]')
    .setInputFiles({ name: "app.zip", mimeType: "application/zip", buffer: Buffer.from("PK") });
  await page.getByPlaceholder("my-app").fill("my-app");
  await page.getByRole("dialog").getByRole("button", { name: "Deploy", exact: true }).click();

  await expect(page.getByText(/deploy failed/i)).toBeVisible();
  await expect(page.getByText(/previous version still serving/i)).toBeVisible();
  await page.getByRole("link", { name: /open diagnose/i }).click();
  await expect(page).toHaveURL("/projects/my-app");
});

test("zip upload path deploys and shows progress", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);
  await openDeploy(page);

  await page
    .locator('input[type="file"]')
    .setInputFiles({ name: "app.zip", mimeType: "application/zip", buffer: Buffer.from("PK") });
  await page.getByPlaceholder("my-app").fill("my-app");

  const [request] = await Promise.all([
    page.waitForRequest(
      (req) =>
        req.url().endsWith("/api/deploy") && req.method() === "POST"
    ),
    page.getByRole("dialog").getByRole("button", { name: "Deploy", exact: true }).click()
  ]);
  expect(request.method()).toBe("POST");

  await expect(page.getByText(/deploying/i)).toBeVisible();
  await expect(page.getByText(/app is live/i)).toBeVisible();
});
