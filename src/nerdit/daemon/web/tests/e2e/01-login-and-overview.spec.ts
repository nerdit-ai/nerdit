import { expect, test } from "@playwright/test";
import { loginAsToken, mockApi, TOKEN } from "./fixtures";

// The apps list is the post-login home and now renders AT "/" (dashboard
// redesign W3). The old Hardware page is folded into Settings, so its
// responsive-telemetry coverage moved to the Settings spec with it.

for (const status of [401, 403]) {
  for (const entry of ["form", "magic link"]) {
    test(`invalid ${entry} token (${status}) keeps its error and allows retry`, async ({ page }) => {
      await mockApi(page);
      await page.route("**/api/auth/check", (route) => {
        const auth = route.request().headers().authorization;
        if (auth === "Bearer rejected-candidate") {
          return route.fulfill({ status, contentType: "application/json", body: JSON.stringify({
            code: "invalid_token", message: "Invalid token"
          }) });
        }
        return route.fallback();
      });
      const documents: string[] = [];
      page.on("request", (request) => {
        if (request.resourceType() === "document") documents.push(request.url());
      });
      await page.goto(entry === "magic link" ? "/login#token=rejected-candidate" : "/login");
      if (entry === "form") {
        await page.getByLabel(/paste your nerdit token/i).fill("rejected-candidate");
        await page.getByRole("button", { name: "Sign in", exact: true }).click();
      }
      await expect(page.getByText("Invalid token. Please try again.")).toBeVisible();
      await expect(page.getByLabel(/paste your nerdit token/i)).toBeFocused();
      await expect(page).toHaveURL("/login");
      expect(documents).toHaveLength(1);
      expect(await page.evaluate(() => sessionStorage.getItem("nerdit.token"))).toBeNull();
      expect(await page.evaluate(() => localStorage.getItem("nerdit.token"))).toBeNull();
      await page.getByLabel(/paste your nerdit token/i).fill(TOKEN);
      await page.getByRole("button", { name: "Sign in", exact: true }).click();
      await expect(page.getByRole("heading", { name: "Apps", level: 1 })).toBeVisible();
      expect(documents).toHaveLength(1);
    });
  }
}

test("login lands on the apps list with mocked daemon", async ({ page }) => {
  await mockApi(page);

  await loginAsToken(page);
  await expect(page).toHaveURL("/");

  // The list header.
  await expect(page.getByRole("heading", { name: "Apps", level: 1 })).toBeVisible();

  // Both fixture services render as rows in the apps table.
  await expect(page.getByTestId("apps-table")).toBeVisible();
  await expect(page.getByTestId("app-row-my-app")).toBeVisible();
  await expect(page.getByTestId("app-row-worker-api")).toBeVisible();

  // The one primary action of the page.
  await expect(page.getByRole("button", { name: /new app/i })).toBeVisible();

  // Daemon-status banner must stay hidden when /api/health returns 200.
  await expect(page.getByText(/daemon offline/i)).toHaveCount(0);
  await expect(page.getByText(/reconnecting to nerditd/i)).toHaveCount(0);
});

test("Cloud session opens without a token and never offers a fake sign out", async ({ page }) => {
  await mockApi(page, { authMode: "tunnel", authRole: "submitter" });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Apps", level: 1 })).toBeVisible();
  expect(await page.evaluate(() => sessionStorage.getItem("nerdit.token"))).toBeNull();
  expect(await page.evaluate(() => localStorage.getItem("nerdit.token"))).toBeNull();
  await expect(page.getByRole("button", { name: "Sign out", exact: true })).toHaveCount(0);
  await page.goto("/settings");
  await expect(page.getByText("Connected through Nerdit Cloud.")).toBeVisible();
  await expect(page.getByTestId("sign-out")).toHaveCount(0);
  await expect(page.getByText("Token storage", { exact: true })).toHaveCount(0);
});

test("expired Cloud session has recovery instead of daemon token entry", async ({ page }) => {
  await mockApi(page, { authMode: "tunnel", authRole: "submitter" });
  await page.route("**/api/auth/check", (route) => route.fulfill({
    status: 401, contentType: "application/json", body: JSON.stringify({ error: { code: "proxy_session_expired", message: "The session expired", status: 401 } })
  }));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Session ended" })).toBeVisible();
  await expect(page.getByText("Reopen this machine from Nerdit Cloud.")).toBeVisible();
  await expect(page.getByLabel(/paste your nerdit token/i)).toHaveCount(0);
});

test("a Cloud session ending mid-page keeps recovery in the loaded shell", async ({ page }) => {
  await mockApi(page, { authMode: "tunnel", authRole: "submitter" });
  const documents: string[] = [];
  page.on("request", (request) => {
    if (request.resourceType() === "document") documents.push(request.url());
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Apps", level: 1 })).toBeVisible();
  await page.route("**/api/tokens", (route) => route.fulfill({
    status: 401, contentType: "application/json", body: JSON.stringify({ error: { code: "proxy_session_expired", message: "The session expired", status: 401 } })
  }));
  await page.getByTestId("nav-link-tokens").click();
  await expect(page.getByRole("heading", { name: "Session ended" })).toBeVisible();
  expect(documents).toHaveLength(1);
  await expect(page.getByLabel(/paste your nerdit token/i)).toHaveCount(0);
});
