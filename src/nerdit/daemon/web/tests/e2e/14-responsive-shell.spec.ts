import { expect, test } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

// The shell after the redesign (W3): one flat nav of six destinations, no
// groups, no status pill. The drawer's focus contract is unchanged.

test("mobile navigation exposes every dashboard route and restores focus", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await mockApi(page);
  await loginAsToken(page);

  const trigger = page.getByRole("button", { name: "Open navigation" });
  await trigger.click();

  const drawer = page.getByRole("dialog", { name: "Dashboard navigation" });
  await expect(drawer).toBeVisible();

  for (const label of ["Apps", "Models", "Databases", "Tokens", "Activity", "Settings"]) {
    await expect(drawer.getByRole("link", { name: label, exact: true })).toBeVisible();
  }

  await page.keyboard.press("Escape");
  await expect(drawer).toBeHidden();
  await expect(trigger).toBeFocused();

  await trigger.click();
  await drawer.getByRole("link", { name: "Models", exact: true }).click();
  await expect(page).toHaveURL("/models");
  await expect(drawer).toBeHidden();
});
