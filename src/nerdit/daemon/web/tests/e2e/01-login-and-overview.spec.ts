import { expect, test } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

// The apps list is the post-login home and now renders AT "/" (dashboard
// redesign W3). The old Hardware page is folded into Settings, so its
// responsive-telemetry coverage moved to the Settings spec with it.

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
