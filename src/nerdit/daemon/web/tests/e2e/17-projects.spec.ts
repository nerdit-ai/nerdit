import { expect, test } from "@playwright/test";
import { loginAsToken, mockApi } from "./fixtures";

// The apps list (dashboard redesign W3): a table, one row per kind=service row,
// with the run-state badge from lib/status, the address, and the last-deploy
// column. The aggregated project status (lib/projects.ts) still reaches the
// operator as the badge's tooltip. Also carries the public-URL / proxy-off
// coverage relocated from the retired /services list (05-services.spec.ts).

test("the table renders a row per app with status, address and last deploy", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);
  await expect(page).toHaveURL("/");

  await expect(page.getByRole("heading", { name: "Apps", level: 1 })).toBeVisible();

  // Both fixture services render as rows.
  await expect(page.getByTestId("app-row-my-app")).toBeVisible();
  await expect(page.getByTestId("app-row-worker-api")).toBeVisible();

  // Aggregated status reaches the operator as the badge tooltip: my-app is
  // healthy and bound to a ready model → running; worker-api has a redeploy
  // building → deploying.
  await expect(page.getByTitle("App running.")).toBeVisible();
  await expect(page.getByTitle("Building the new image.")).toBeVisible();
  // worker-api's live deploy phase renders in the last-deploy column. W4
  // retired the phase chip: the phase is now a word beside the timestamp in
  // one span, so the anchor is the cell's testid, not an exact text node.
  await expect(page.getByTestId("app-last-deploy-worker-api")).toContainText("Building");

  // Routed service: the public URL is a real clickable anchor (new tab).
  const publicLink = page.getByRole("link", {
    name: "https://test-host.nerdit.internal/my-app",
    exact: true
  });
  await expect(publicLink).toBeVisible();
  await expect(publicLink).toHaveAttribute("href", "https://test-host.nerdit.internal/my-app");
  await expect(publicLink).toHaveAttribute("target", "_blank");

  // public_url null while an endpoint exists → expected "proxy off" chip plus
  // the still-usable loopback URL, never an error state.
  await expect(page.getByText("proxy off")).toBeVisible();
  await expect(page.getByText("http://127.0.0.1:38001")).toBeVisible();
  await expect(page.getByText(/error/i)).toHaveCount(0);
});

test("a row opens the app detail page", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.getByTestId("app-row-my-app").click();
  await expect(page).toHaveURL("/projects/my-app");
  await expect(page.getByRole("heading", { name: "my-app" })).toBeVisible();
});

test("empty list sells the north-star gesture", async ({ page }) => {
  await mockApi(page, { services: { items: [], next_cursor: null } });
  await loginAsToken(page);
  await expect(page).toHaveURL("/");

  const empty = page.getByTestId("apps-empty");
  await expect(empty).toBeVisible();
  await expect(empty.getByText("No apps yet.")).toBeVisible();
  await expect(empty.getByText("nerdit deploy .")).toBeVisible();
  // The empty state carries its own action (distinct from the header menu
  // button of the same name).
  await expect(page.getByTestId("apps-empty-new")).toBeVisible();
});

test("the New app menu offers the three sources", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.getByRole("button", { name: /new app/i }).click();
  await expect(page.getByRole("menuitem", { name: /deploy a folder/i })).toBeVisible();
  await expect(page.getByRole("menuitem", { name: /from a git repository/i })).toBeVisible();
  await expect(page.getByRole("menuitem", { name: /from a template/i })).toBeVisible();
});

test("the template flow deploys from the catalog without leaving the list", async ({ page }) => {
  await mockApi(page);
  // The catalog is not part of the shared fixtures; this spec is its only user.
  await page.route(/\/api\/app-templates$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "fastapi-ai-chat",
          name: "FastAPI AI chat",
          description: "A chat app wired to a local model.",
          icon: "message-square",
          category: "ai",
          repo_url: "https://github.com/nerdit-ai/nerdit-templates",
          ref: "v1.0.1",
          subdir: "fastapi-ai-chat",
          deploy_defaults: { port: 8000, gpus: 0, start: null, health: "/" },
          env_schema: [
            {
              name: "OPENAI_API_KEY",
              description: "Key for the external provider.",
              required: true,
              secret: true
            }
          ],
          ai_hint: "Uses [ai.default]"
        }
      ])
    })
  );
  await page.route(/\/api\/app-templates\/[^/]+\/deploy(\?.*)?$/, (route) =>
    route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({ id: "svc9zzzz0009", name: "chat", status: "building" })
    })
  );
  await loginAsToken(page);

  await page.getByRole("button", { name: /new app/i }).click();
  await page.getByRole("menuitem", { name: /from a template/i }).click();

  const dialog = page.getByTestId("template-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAttribute("role", "dialog");

  // Step 1: pick a template. Step 2: name it and fill the required secret.
  await dialog.getByTestId("template-row-fastapi-ai-chat").click();
  await expect(dialog.getByTestId("template-submit")).toBeDisabled();

  await dialog.getByTestId("template-name").fill("chat");
  await dialog.getByTestId("template-env-OPENAI_API_KEY").fill("sk-test");

  const [request] = await Promise.all([
    page.waitForRequest(
      (req) => req.url().includes("/api/app-templates/fastapi-ai-chat/deploy") && req.method() === "POST"
    ),
    dialog.getByTestId("template-submit").click()
  ]);
  const body = request.postDataJSON();
  expect(body.name).toBe("chat");
  expect(body.secrets.OPENAI_API_KEY).toBe("sk-test");
  expect(request.headers()["idempotency-key"]).toBeTruthy();

  // The dialog closes back onto the apps list; no Store page is involved.
  await expect(page.getByTestId("template-dialog")).toHaveCount(0);
  await expect(page).toHaveURL("/");
});

test("the template flow sends the advanced gpus/start overrides", async ({ page }) => {
  // Relocated from 06-models.spec.ts with the retired Store page: the override
  // fields now live behind the dialog's "Advanced" disclosure.
  await mockApi(page);
  await page.route(/\/api\/app-templates(\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "fastapi-ai-chat",
          name: "FastAPI AI chat",
          description: "A minimal chat app.",
          icon: "message-square",
          category: "ai",
          repo_url: "https://github.com/nerdit-ai/nerdit-templates",
          ref: "v1.0.1",
          subdir: "fastapi-ai-chat",
          deploy_defaults: { port: 8000, gpus: 0, start: null, health: "/" },
          env_schema: [
            { name: "MODEL", description: "Model ref", required: true, secret: false }
          ],
          ai_hint: "Binds to a local model."
        }
      ])
    })
  );

  let posted: Record<string, unknown> | null = null;
  await page.route(/\/api\/app-templates\/[^/]+\/deploy(\?.*)?$/, (route) => {
    posted = route.request().postDataJSON();
    return route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({ id: "svc1", name: "my-chat", status: "building" })
    });
  });
  await loginAsToken(page);

  await page.getByRole("button", { name: /new app/i }).click();
  await page.getByRole("menuitem", { name: /from a template/i }).click();

  const dialog = page.getByTestId("template-dialog");
  await dialog.getByTestId("template-row-fastapi-ai-chat").click();
  await dialog.getByTestId("template-name").fill("my-chat");
  await dialog.getByTestId("template-env-MODEL").fill("llama3.1:8b");

  await dialog.getByTestId("template-advanced").click();
  await dialog.getByTestId("template-gpus").fill("1");
  await dialog.getByTestId("template-start").fill("uvicorn app:app");

  await dialog.getByTestId("template-submit").click();

  await expect.poll(() => posted).not.toBeNull();
  expect(posted).toMatchObject({ gpus: 1, start: "uvicorn app:app", env: { MODEL: "llama3.1:8b" } });
});
