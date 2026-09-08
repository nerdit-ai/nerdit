import { expect, test } from "@playwright/test";
import { loginAsToken, mockApi, SAMPLE_SERVICES } from "./fixtures";

// Command palette: ⌘K jumps to any app and carries per-app quick actions (logs,
// redeploy), on top of the static nav rows. Apps come from
// GET /services?limit=200 (kind=service only). The app detail route is still
// /projects/:name (W4 moves it); the nav word is "Apps".

test("⌘K jumps to an app page", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);
  await expect(page).toHaveURL("/");

  await page.keyboard.press("Control+k");
  await page.getByPlaceholder("Jump to…").fill("my-app");
  // The plain jump item is the only one carrying the /projects/my-app suffix
  // (the quick actions show "logs" / "redeploy" instead).
  await page.getByRole("option").filter({ hasText: "/projects/my-app" }).click();
  await expect(page).toHaveURL("/projects/my-app");
  await expect(page.getByRole("heading", { name: "my-app" })).toBeVisible();
});

test("⌘K logs quick action lands on the app logs tab", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.keyboard.press("Control+k");
  await page.getByPlaceholder("Jump to…").fill("my-app logs");
  await page.getByRole("option", { name: /logs/ }).first().click();
  await expect(page).toHaveURL("/projects/my-app/logs");
});

test("⌘K redeploy quick action takes the app's own deploy path, cleared on reload", async ({
  page
}) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.keyboard.press("Control+k");
  await page.getByPlaceholder("Jump to…").fill("my-app redeploy");
  await page.getByRole("option", { name: /redeploy/ }).first().click();

  await expect(page).toHaveURL("/projects/my-app");
  // The handoff routes through the SAME source-aware handler as the header
  // Deploy button, so a git-sourced app (my-app, per fixtures) gets the
  // "Deploy the latest source" confirm — not the ZIP upload dialog.
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByTestId("app-deploy-confirm")).toBeVisible();

  // Dismiss it, then reload: the redeploy history state was cleared, so the
  // confirm must not re-open.
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await page.reload();
  await expect(page.getByRole("heading", { name: "my-app" })).toBeVisible();
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("⌘K still reaches static nav entries", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.keyboard.press("Control+k");
  await page.getByPlaceholder("Jump to…").fill("Activity");
  await page.getByRole("option", { name: /Activity/ }).first().click();
  await expect(page).toHaveURL("/activity");
});

test("⌘K Apps group excludes non-service kinds", async ({ page }) => {
  // A model-kind row present in the /services page must never surface as an
  // app (the palette filters kind === "service", mirroring the list).
  await mockApi(page, {
    services: {
      items: [
        ...SAMPLE_SERVICES.items,
        { ...SAMPLE_SERVICES.items[0], id: "mdlrow0001", name: "sneaky-model", kind: "model" }
      ],
      next_cursor: null
    }
  });
  await loginAsToken(page);

  await page.keyboard.press("Control+k");
  // First prove the Apps group renders at all in this very session (a broken
  // fetch would also yield zero options below, vacuously).
  await page.getByPlaceholder("Jump to…").fill("my-app");
  await expect(page.getByRole("option", { name: /\/projects\/my-app/ })).toBeVisible();
  await page.getByPlaceholder("Jump to…").fill("sneaky-model");
  await expect(page.getByRole("option")).toHaveCount(0);
});

test("D9: a model row names the app bound to it, linking to its page", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.getByRole("link", { name: "Models", exact: true }).click();
  await expect(page).toHaveURL("/models");

  // SAMPLE_APP_CONFIG binds [ai.default] to llama3.1:8b, which is the ready
  // model row's backing ref — so it carries a "Used by" chip to its app page.
  // Anchored on copy rather than on the row's container class, which the
  // Models page redesign is free to change.
  await expect(page.getByText("Used by").first()).toBeVisible();
  await page.getByRole("link", { name: "my-app", exact: true }).first().click();
  await expect(page).toHaveURL("/projects/my-app");
});
