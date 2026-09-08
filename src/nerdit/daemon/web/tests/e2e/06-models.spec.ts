import { expect, test } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

test("models list renders with pull state, GPU util and loopback endpoint only", async ({
  page
}) => {
  await mockApi(page);
  await loginAsToken(page);

  // exact: the projects grid's Infrastructure hint links "N models serving",
  // whose accessible name would also match a non-exact "Models" query.
  await page.getByRole("link", { name: "Models", exact: true }).click();
  await expect(page).toHaveURL("/models");

  // Both fixture models render with their backing model reference.
  await expect(page.getByText("ollama-llama3-1-8b")).toBeVisible();
  await expect(page.getByText("llama3.1:8b")).toBeVisible();
  await expect(page.getByText("ollama-qwen2-5-3b")).toBeVisible();

  // Run state comes from the shared status map (one Badge per row).
  const readyRow = page.getByRole("row").filter({ hasText: "ollama-llama3-1-8b" });
  await expect(readyRow.getByTestId("model-status")).toHaveText("Running");

  // Weights-pull readiness: one rendering per row (a muted badge), never a dot
  // and a pill. The ready model is "Pulled", the in-flight one is still
  // pulling — a distinct expected phase, not an error.
  await expect(readyRow.getByTestId("model-readiness")).toHaveText("Pulled");
  const pullingRow = page.getByRole("row").filter({ hasText: "ollama-qwen2-5-3b" });
  await expect(pullingRow.getByTestId("model-readiness")).toHaveText("Pulling weights");

  // GPU id + live utilization for the GPU-backed model; CPU for the other.
  await expect(page.getByText("GPU-A100-001")).toBeVisible();
  await expect(page.getByText("42%")).toBeVisible();
  await expect(page.getByText("CPU", { exact: true })).toBeVisible();

  // Loopback OpenAI-compatible endpoint is shown...
  await expect(page.getByText("http://127.0.0.1:38100/v1")).toBeVisible();
  // ...pending for the model whose port is not published yet.
  await expect(page.getByText(/endpoint pending/i)).toBeVisible();
});

test("serve form posts the chosen backend from capabilities", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  // Capture the POST /models body. Registered after mockApi so it wins.
  let posted: Record<string, unknown> | null = null;
  await page.route(/\/api\/models(\?.*)?$/, async (route) => {
    if (route.request().method() === "POST") {
      posted = route.request().postDataJSON();
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({ id: "m1", name: "vllm-x", model: "x", status: "pending" })
      });
    }
    return route.fallback();
  });

  await page.goto("/models");

  // The page's one primary action reveals the serve form (calm by default).
  await page.getByTestId("open-serve-form").click();
  await expect(page.getByTestId("serve-panel")).toBeVisible();

  // Backend select is fed by GET /capabilities (ollama default, vllm offered).
  const backend = page.getByLabel("Backend");
  await expect(backend).toBeVisible();
  await backend.selectOption("vllm");
  // "vLLM needs at least 1 GPU" one-liner shows while GPUs is 0.
  await expect(page.getByText(/vLLM needs at least 1 GPU/i)).toBeVisible();

  await page.getByLabel("Model reference").fill("mistral-7b");
  await page.getByRole("button", { name: "Serve", exact: true }).click();

  await expect.poll(() => posted).not.toBeNull();
  expect(posted).toMatchObject({ model: "mistral-7b", backend: "vllm" });
});

test("model row actions live in the menu, and delete confirms first", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  // Capture the DELETE so the test proves the confirm gates the call.
  let deleted: string | null = null;
  await page.route(/\/api\/services\/[^/]+(\?.*)?$/, async (route) => {
    if (route.request().method() === "DELETE") {
      deleted = new URL(route.request().url()).pathname;
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ id: "mdl1cccc0003", name: "ollama-llama3-1-8b", deleted: true })
      });
    }
    return route.fallback();
  });

  await page.goto("/models");
  const row = page.getByRole("row").filter({ hasText: "ollama-llama3-1-8b" });
  await row.getByRole("button", { name: "More actions" }).click();

  const menu = page.getByRole("menu");
  await expect(menu.getByRole("menuitem", { name: "Restart" })).toBeVisible();
  await expect(menu.getByRole("menuitem", { name: "Stop" })).toBeVisible();
  await menu.getByRole("menuitem", { name: "Delete…" }).click();

  // Nothing is destroyed until the confirm is accepted.
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  expect(deleted).toBeNull();

  await dialog.getByRole("button", { name: "Delete", exact: true }).click();
  await expect.poll(() => deleted).toBe("/api/services/ollama-llama3-1-8b");
});

test("models never render a public URL (loopback-only by design)", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);
  await page.goto("/models");
  await expect(page.getByText("ollama-llama3-1-8b")).toBeVisible();

  // No external anchor anywhere on the page: router <Link>s are relative, so
  // any http(s) href would be a leaked public URL for a model.
  await expect(page.locator('a[href^="http"]')).toHaveCount(0);
  // And no https:// text is rendered for any model row.
  await expect(page.getByText(/https:\/\//)).toHaveCount(0);
});
