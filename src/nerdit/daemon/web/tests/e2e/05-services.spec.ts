import { expect, test } from "@playwright/test";
import { SAMPLE_SERVICES, loginAsToken, mockApi } from "./fixtures";

// The /services list page was retired (dashboard refonte PR2). Its public-URL /
// proxy-off list coverage moved to the apps table (17-projects.spec.ts); the
// service detail moved to the app page /projects/:name (/services/:ident is now
// a kind-aware redirect), so its rollback / build-version / share assertions
// live on here against the app page. W4 moved rollback into the `…` menu and
// the identity facts behind the Manage tab's Details disclosure.

test("roll back is offered in the menu only when a previous version exists", async ({ page }) => {
  await mockApi(page, {
    rollbackResponse: {
      status: 409,
      body: {
        code: "deploy.no_previous_version",
        message: "No previous version recorded for 'my-app'",
        detail: "No previous version recorded for 'my-app'"
      }
    }
  });
  await loginAsToken(page);

  // my-app has rollback_available=true → the menu offers Roll back.
  await page.goto("/projects/my-app");
  await page.getByRole("button", { name: "More actions" }).click();
  const rollback = page.getByRole("menuitem", { name: /roll back/i });
  await expect(rollback).toBeVisible();

  // The structured 409 envelope lands as an error toast; the page stays up.
  await rollback.click();
  await expect(page.getByText(/rollback failed/i)).toBeVisible();
  await expect(page.getByText(/no previous version/i)).toBeVisible();
  await expect(page.getByRole("heading", { name: "my-app" })).toBeVisible();

  // worker-api has rollback_available=false → no Roll back item at all.
  await page.goto("/projects/worker-api");
  // `exact` matters: worker-api carries an in-flight deploy, so the progress
  // walk contributes a second heading ("Deploying worker-api") on Overview.
  await expect(page.getByRole("heading", { name: "worker-api", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "More actions" }).click();
  await expect(page.getByRole("menuitem", { name: /roll back/i })).toHaveCount(0);
});

test("the Details disclosure carries the build version and endpoint facts", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const svc = SAMPLE_SERVICES.items[0];

  const details = page.getByTestId("app-details");
  await details.getByRole("button", { name: "Show" }).click();

  await expect(details.getByText(`v${svc.build_version}`)).toBeVisible();
  await expect(details.getByText(/previous version available/i)).toBeVisible();
  await expect(details.getByText(svc.image)).toBeVisible();
  await expect(
    details.getByText(
      `${svc.endpoint.container_port} → ${svc.endpoint.host_port} (${svc.endpoint.protocol})`
    )
  ).toBeVisible();
});

test("a routed app shows one shareable address with a copy affordance", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app");
  const url = SAMPLE_SERVICES.items[0].endpoint.public_url;

  // The address renders once, in the header block.
  const address = page.getByTestId("app-address");
  await expect(address.getByText(url)).toBeVisible();
  await expect(address.getByText(/nerdit trust/i)).toBeVisible();
  await expect(address.getByRole("button", { name: /copy/i }).first()).toBeVisible();
});
