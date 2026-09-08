import { expect, test } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

// The shared-secrets card moved from the retired /services list to Settings
// (dashboard refonte PR2, D10).
test("shared secrets card lists existing keys and adds a new one", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.getByRole("link", { name: "Settings", exact: true }).click();
  await expect(page).toHaveURL("/settings");

  await expect(page.getByText("Shared secrets")).toBeVisible();
  // ${secrets.shared.KEY} usage hint is present in the card copy.
  await expect(page.getByText(/secrets\.shared\.KEY/)).toBeVisible();

  // Fixture GET /api/secrets/shared always returns one existing key. The row is
  // anchored by testid (W5: the key and the "••••••" mask are separate nodes
  // now); the mask assertion is what proves the write-only contract on a read.
  const row = page.getByTestId("secret-row-API_KEY");
  await expect(row).toBeVisible();
  await expect(row).toContainText("API_KEY");
  await expect(row).toContainText("••••••");

  // Happy path: add a new key/value pair. The success toast is the signal;
  // the fixture's GET handler is stateless (always returns ["API_KEY"]) so
  // the post-refetch list isn't asserted here — that round trip is covered
  // by the backend's own test suite.
  await page.getByPlaceholder("KEY").fill("NEW_TOKEN");
  await page.getByPlaceholder("value").fill("super-secret-value");
  await page.getByRole("button", { name: /^add$/i }).click();

  await expect(page.getByText("Secret NEW_TOKEN set")).toBeVisible();

  // Write-only contract: the value never appears anywhere on the page.
  await expect(page.getByText("super-secret-value")).toHaveCount(0);
});
