import { expect, test } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

// Navigation after the redesign (W3): one flat six-row nav, the apps list AT
// "/", and every retired URL still resolving rather than 404ing.

test("sidebar shows one flat nav of six destinations", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  const nav = page.getByRole("navigation", { name: "Dashboard" });

  for (const label of ["Apps", "Models", "Databases", "Tokens", "Activity", "Settings"]) {
    await expect(nav.getByRole("link", { name: label, exact: true })).toBeVisible();
  }

  // Retired rows: Services (PR3), Store and Hardware (both folded elsewhere),
  // and the old words for two that survive.
  for (const gone of ["Services", "Store", "Hardware", "Projects", "Audit"]) {
    await expect(nav.getByRole("link", { name: gone, exact: true })).toHaveCount(0);
  }
});

test("the apps list is the home page, not a redirect", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/");
  await expect(page).toHaveURL("/");
  await expect(page.getByRole("navigation", { name: "Dashboard" })).toBeVisible();
});

test("retired URLs still resolve", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  // "/bindings/*" has no row of its own: it is the catch-all, which is the
  // point — a retired address must never reach the router's error screen.
  for (const from of ["/projects", "/services", "/store", "/bindings/ai", "/bindings/db"]) {
    await page.goto(from);
    await expect(page).toHaveURL("/");
  }

  await page.goto("/audit");
  await expect(page).toHaveURL("/activity");

  await page.goto("/hardware");
  await expect(page).toHaveURL("/settings");
});

test("/services/:ident redirects a service to its app page (kind-aware, PR3)", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/services/my-app");
  await expect(page).toHaveURL("/projects/my-app");
  await expect(page.getByRole("heading", { name: "my-app" })).toBeVisible();
});
