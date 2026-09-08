import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { SAMPLE_APP_CONFIG, SAMPLE_APP_CONFIG_API, loginAsToken, mockApi } from "./fixtures";

// The panel lives in the "AI resources" panel of the app page's Manage tab (no
// dialog). Scope banners to that region so page-wide text never satisfies an
// in-panel assertion.
function bindingsRegion(page: Page) {
  return page.locator("section").filter({ has: page.getByRole("heading", { name: "AI resources" }) });
}

test("binding card shows provider, readiness and injected env key names", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");

  const card = page.getByTestId("binding-card-default");
  await expect(card.getByText("ollama")).toBeVisible();
  await expect(card.getByText("llama3.1:8b")).toBeVisible();
  await expect(card.getByText("Model running, weights pulled.")).toBeVisible();
  await expect(card.getByText("OPENAI_BASE_URL")).toBeVisible();
  await expect(card.getByText("NERDIT_AI_DEFAULT_URL")).toBeVisible();
});

test("swap local to api with a pasted key writes the secret then the ref", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();

  await card.getByRole("button", { name: "External API" }).click();
  // A non-preset base URL keeps the derived-name paste flow (the OpenAI/OpenRouter
  // presets now suggest their own secret name — covered by the preset tests below).
  await card.getByPlaceholder("https://api.openai.com/v1").fill("https://api.example.com/v1");
  await card.getByPlaceholder("gpt-4o-mini").fill("gpt-4o-mini");

  await card.getByLabel("Paste a new key").check();
  // The derived secret name pre-fills beside the value input.
  await expect(card.getByTestId("keyflow-paste-name")).toHaveValue("AI_DEFAULT_API_KEY");
  await card.getByPlaceholder("sk-...").fill("sk-test-123");

  const secretsReq = page.waitForRequest(
    (r) => r.method() === "POST" && r.url().includes("/api/secrets/my-app")
  );
  const putReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/ai") &&
      r.url().includes("restart=true") &&
      !r.url().includes("dry_run")
  );

  await card.getByRole("button", { name: /^save$/i }).click();

  const secrets = await secretsReq;
  expect(secrets.postDataJSON()).toEqual({ values: { AI_DEFAULT_API_KEY: "sk-test-123" } });

  const put = await putReq;
  const body = put.postDataJSON();
  expect(body.default.api_key).toBe("${secrets.AI_DEFAULT_API_KEY}");
  expect(body.default.provider).toBe("api");
  expect(put.headers()["if-match"]).toBe("app-config-etag-1");
  expect(put.headers()["idempotency-key"]).toBeTruthy();
  // The raw key is never a literal in the ai section (D3).
  expect(put.postData() ?? "").not.toContain("sk-test-123");

  await expect(page.getByText(/resource default saved/i)).toBeVisible();
});

test("an incomplete api binding cannot save and never posts a secret", async ({ page }) => {
  // Regression (Codex P2): a pasted key alone must NOT enable Save — the config
  // PUT would 422 on the missing model/base_url and orphan the just-written
  // secret. The validity gate keeps Save disabled and the secret write out of
  // reach until the mandatory provider fields are filled.
  await mockApi(page);
  await loginAsToken(page);

  let secretPosted = false;
  page.on("request", (r) => {
    if (r.method() === "POST" && r.url().includes("/api/secrets/my-app")) secretPosted = true;
  });

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();

  // Flip to External API but fill ONLY the pasted key — leave model + base_url empty.
  await card.getByRole("button", { name: "External API" }).click();
  await card.getByLabel("Paste a new key").check();
  await card.getByPlaceholder("sk-...").fill("sk-orphan-999");

  const save = card.getByRole("button", { name: /^save$/i });
  await expect(save).toBeDisabled();
  // Attempting Save on the disabled control does nothing; no secret is posted.
  await save.click({ force: true }).catch(() => undefined);
  expect(secretPosted).toBe(false);

  // Fill the mandatory fields; Save becomes reachable.
  await card.getByPlaceholder("https://api.openai.com/v1").fill("https://api.openai.com/v1");
  await card.getByPlaceholder("gpt-4o-mini").fill("gpt-4o-mini");
  await expect(save).toBeEnabled();
});

test("swap api to local picks from the served-model catalog", async ({ page }) => {
  await mockApi(page, { appConfig: SAMPLE_APP_CONFIG_API });
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();

  await card.getByRole("button", { name: "Local model" }).click();

  const select = card.locator("select");
  await expect(select.getByRole("option", { name: "llama3.1:8b" })).toHaveCount(1);
  await expect(select.getByRole("option", { name: "qwen2.5:3b (pulling)" })).toHaveCount(1);
  await select.selectOption("llama3.1:8b");

  const putReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/ai") &&
      r.url().includes("restart=true") &&
      !r.url().includes("dry_run")
  );
  await card.getByRole("button", { name: /^save$/i }).click();

  const put = await putReq;
  expect(put.postDataJSON()).toEqual({ default: { provider: "ollama", model: "llama3.1:8b" } });
  expect(put.headers()["if-match"]).toBe("app-config-etag-api-1");
});

test("a stale config 409 says someone else changed it", async ({ page }) => {
  await mockApi(page);
  // Override the section PUT to reject with the structured stale envelope.
  await page.route(/\/api\/config\/apps\/[^/?]+\/[^/?]+(\?.*)?$/, (route) => {
    if (route.request().method() !== "PUT") return route.fallback();
    return route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({
        code: "config.stale",
        message: "Config changed since the ETag was read",
        detail: "Config changed since the ETag was read"
      })
    });
  });
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();

  // Edit the model via the free-text "Other model name" path.
  await card.locator("select").selectOption({ label: "Other model name" });
  await card.getByPlaceholder("llama3.1:8b").fill("mistral:7b");
  await card.getByRole("button", { name: /^save$/i }).click();

  await expect(
    bindingsRegion(page).getByText(/someone else changed this config/i)
  ).toBeVisible();
});

test("abandoned paste does not leak into a local apply", async ({ page }) => {
  // Start from an api binding so the flip to local is a real (dirty) change; a
  // default ollama llama3.1:8b binding would round-trip to itself and never dirty.
  await mockApi(page, { appConfig: SAMPLE_APP_CONFIG_API });
  await loginAsToken(page);

  // Track that no secret POST fires when the paste is abandoned via the ollama flip.
  let secretPosted = false;
  page.on("request", (r) => {
    if (r.method() === "POST" && r.url().includes("/api/secrets/my-app")) secretPosted = true;
  });

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();

  // Arm a paste on the api path, then abandon it by switching to local.
  await card.getByRole("button", { name: "External API" }).click();
  await card.getByLabel("Paste a new key").check();
  await card.getByPlaceholder("sk-...").fill("sk-abandoned-1");

  await card.getByRole("button", { name: "Local model" }).click();
  await card.locator("select").selectOption("llama3.1:8b");

  const putReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/ai") &&
      r.url().includes("restart=true") &&
      !r.url().includes("dry_run")
  );
  await card.getByRole("button", { name: /^save$/i }).click();

  const put = await putReq;
  expect(put.postDataJSON()).toEqual({ default: { provider: "ollama", model: "llama3.1:8b" } });
  expect(put.postData() ?? "").not.toContain("sk-abandoned-1");
  expect(secretPosted).toBe(false);
});

test("editing an api binding re-supplies the redacted key before apply", async ({ page }) => {
  // GET redacts api_key to "***"; the stored ${secrets.*} ref is unknown to the
  // client, and _merge_ai replaces the whole spec, so a plain model edit must
  // re-pick a key source (echoing "***" back would 422 config.invalid).
  await mockApi(page, {
    appConfig: SAMPLE_APP_CONFIG_API,
    secretNames: { "my-app": ["AI_DEFAULT_API_KEY"] }
  });
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();

  // Change only the model; the key source is still "keep current".
  await card.getByPlaceholder("gpt-4o-mini").fill("gpt-4o");

  // Apply is blocked while the redacted key has no fresh source, and the hidden
  // key is surfaced honestly (never the literal "***" as a "reference").
  await expect(card.getByText(/stored key is hidden/i)).toBeVisible();
  await expect(card.getByRole("button", { name: /^save$/i })).toBeDisabled();

  // Re-pick the existing secret; the ref is reconstructed from its name.
  await card.getByLabel("Use an existing secret").check();

  const putReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/ai") &&
      r.url().includes("restart=true") &&
      !r.url().includes("dry_run")
  );
  await card.getByRole("button", { name: /^save$/i }).click();

  const put = await putReq;
  const body = put.postDataJSON();
  expect(body.default.provider).toBe("api");
  expect(body.default.model).toBe("gpt-4o");
  expect(body.default.api_key).toBe("${secrets.AI_DEFAULT_API_KEY}");
  // The redaction sentinel is never written back to the server.
  expect(put.postData() ?? "").not.toContain("***");
});

test("readiness is honest for a pulling and an unserved model", async ({ page }) => {
  await mockApi(page, {
    appConfig: {
      ...SAMPLE_APP_CONFIG,
      ai: {
        default: { provider: "ollama", model: "qwen2.5:3b", base_url: null, api_key: null },
        extra: { provider: "ollama", model: "missing:1b", base_url: null, api_key: null }
      }
    }
  });
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");

  const def = page.getByTestId("binding-card-default");
  const extra = page.getByTestId("binding-card-extra");
  await expect(def.getByText("Model still pulling. App waits at launch.")).toBeVisible();
  await expect(extra.getByText("Model not served. App waits at launch.")).toBeVisible();

  await expect(extra.getByText("NERDIT_AI_EXTRA_URL")).toBeVisible();
  await expect(extra.getByText("OPENAI_BASE_URL")).toHaveCount(0);
});

// --- OpenRouter friction-reduction (provider presets) -----------------------

test("external API path shows a provider preset select; OpenRouter fills the base URL", async ({
  page
}) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();
  await card.getByRole("button", { name: "External API" }).click();

  const baseUrl = card.getByPlaceholder("https://api.openai.com/v1");
  await expect(baseUrl).toHaveValue("");

  // The preset select is the first select on the api path (before the base URL).
  // Only OpenRouter + Custom ship (owner decision — OpenAI was dropped).
  const presetSelect = card.locator("select").first();
  await expect(presetSelect.getByRole("option", { name: "OpenRouter" })).toHaveCount(1);
  await expect(presetSelect.getByRole("option", { name: "OpenAI" })).toHaveCount(0);
  await expect(presetSelect.getByRole("option", { name: "Custom" })).toHaveCount(1);

  await presetSelect.selectOption({ label: "OpenRouter" });
  await expect(baseUrl).toHaveValue("https://openrouter.ai/api/v1");
  // The derived preset value now reflects OpenRouter (no separate draft field).
  await expect(presetSelect).toHaveValue("openrouter");
});

test("OpenRouter base URL turns the model field into a curated free-model dropdown", async ({
  page
}) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();
  await card.getByRole("button", { name: "External API" }).click();
  await card.locator("select").first().selectOption({ label: "OpenRouter" });

  // The model field is now a dropdown of curated free models.
  const modelSelect = card.locator("select").nth(1);
  await expect(
    modelSelect.getByRole("option", { name: "meta-llama/llama-3.3-70b-instruct:free" })
  ).toHaveCount(1);
  await modelSelect.selectOption("meta-llama/llama-3.3-70b-instruct:free");
  // The helper line names the active provider (label-driven, not hardcoded).
  await expect(card.getByText("Free tier, rate-limited by OpenRouter.")).toBeVisible();

  // No shared OPENROUTER_API_KEY on this node (default fixture): the preset key
  // flow is a single paste input wired to the conventional name — no radios.
  await expect(card.getByText("OpenRouter API key")).toBeVisible();
  await expect(card.getByLabel("Paste a new key")).toHaveCount(0);
  await card.getByPlaceholder("sk-...").fill("sk-or-abc");

  const secretsReq = page.waitForRequest(
    (r) => r.method() === "POST" && r.url().includes("/api/secrets/my-app")
  );
  const putReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/ai") &&
      r.url().includes("restart=true") &&
      !r.url().includes("dry_run")
  );
  await card.getByRole("button", { name: /^save$/i }).click();

  // The pasted value is written once under the conventional per-app name…
  const secrets = await secretsReq;
  expect(secrets.postDataJSON()).toEqual({ values: { OPENROUTER_API_KEY: "sk-or-abc" } });
  // …and the ai section carries only the reference.
  const put = await putReq;
  expect(put.postDataJSON().default).toEqual({
    provider: "api",
    model: "meta-llama/llama-3.3-70b-instruct:free",
    base_url: "https://openrouter.ai/api/v1",
    api_key: "${secrets.OPENROUTER_API_KEY}"
  });
});

test("Other model name on the OpenRouter path reveals a free-text model input", async ({
  page
}) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();
  await card.getByRole("button", { name: "External API" }).click();
  await card.locator("select").first().selectOption({ label: "OpenRouter" });

  const modelSelect = card.locator("select").nth(1);
  await modelSelect.selectOption("meta-llama/llama-3.3-70b-instruct:free");
  // Flip to the escape hatch — the free-text input appears (placeholder = the
  // preset's example model) and its typed value lands in the PUT body.
  await modelSelect.selectOption({ label: "Other model name" });
  const customInput = card.getByPlaceholder("meta-llama/llama-3.3-70b-instruct:free");
  await expect(customInput).toBeVisible();
  await customInput.fill("acme/custom-model:free");

  await card.getByPlaceholder("sk-...").fill("sk-or-xyz");

  const putReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/ai") &&
      r.url().includes("restart=true") &&
      !r.url().includes("dry_run")
  );
  await card.getByRole("button", { name: /^save$/i }).click();

  const put = await putReq;
  expect(put.postDataJSON().default.model).toBe("acme/custom-model:free");
  expect(put.postDataJSON().default.base_url).toBe("https://openrouter.ai/api/v1");
});

test("OpenRouter asks for no key when the shared OPENROUTER_API_KEY exists", async ({
  page
}) => {
  // The whole point of the preset: a node where the shared convention secret is
  // already set never asks the user for a key — the binding references it.
  await mockApi(page, {
    secretNames: { shared: ["API_KEY", "OPENROUTER_API_KEY"] }
  });
  await loginAsToken(page);

  let secretPosted = false;
  page.on("request", (r) => {
    if (r.method() === "POST" && r.url().includes("/api/secrets/")) secretPosted = true;
  });

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();
  await card.getByRole("button", { name: "External API" }).click();
  await card.locator("select").first().selectOption({ label: "OpenRouter" });

  // No paste input, no radio picker — just the honest wiring note.
  await expect(card.getByText(/Uses the shared/)).toBeVisible();
  await expect(card.getByText("OPENROUTER_API_KEY")).toBeVisible();
  await expect(card.getByPlaceholder("sk-...")).toHaveCount(0);
  await expect(card.getByLabel("Paste a new key")).toHaveCount(0);

  await card.locator("select").nth(1).selectOption("meta-llama/llama-3.3-70b-instruct:free");

  const putReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/ai") &&
      r.url().includes("restart=true") &&
      !r.url().includes("dry_run")
  );
  await card.getByRole("button", { name: /^save$/i }).click();

  const put = await putReq;
  expect(put.postDataJSON().default.api_key).toBe("${secrets.shared.OPENROUTER_API_KEY}");
  // Referenced, never copied: no secret write of any kind fired.
  expect(secretPosted).toBe(false);
});

test("a custom URL keeps the full key-source picker with the derived name", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();
  await card.getByRole("button", { name: "External API" }).click();

  // Custom (non-preset) base URL → the full KeyFlow radios, derived paste name.
  await card.getByPlaceholder("https://api.openai.com/v1").fill("https://api.example.com/v1");
  await card.getByLabel("Paste a new key").check();
  await expect(card.getByTestId("keyflow-paste-name")).toHaveValue("AI_DEFAULT_API_KEY");
});

test("flipping the provider preset never clears an already-typed model", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();
  await card.getByRole("button", { name: "External API" }).click();

  // Type a model on the custom path, then flip to OpenRouter: the model must
  // survive, riding the "Other model name" free-text path (it is not curated).
  await card.getByPlaceholder("https://api.openai.com/v1").fill("https://api.example.com/v1");
  await card.getByPlaceholder("gpt-4o-mini").fill("gpt-4o");
  await card.locator("select").first().selectOption({ label: "OpenRouter" });

  await expect(card.getByPlaceholder("https://api.openai.com/v1")).toHaveValue(
    "https://openrouter.ai/api/v1"
  );
  await expect(card.getByPlaceholder("meta-llama/llama-3.3-70b-instruct:free")).toHaveValue(
    "gpt-4o"
  );
});

test("Custom stays selectable while a preset URL is active", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();
  await card.getByRole("button", { name: "External API" }).click();

  const presetSelect = card.locator("select").first();
  await presetSelect.selectOption({ label: "OpenRouter" });

  // Choosing Custom from a preset must stick (not silently snap back): the URL
  // is kept for editing and the model field reverts to the plain text input.
  await presetSelect.selectOption({ label: "Custom" });
  await expect(presetSelect).toHaveValue("custom");
  await expect(card.getByPlaceholder("https://api.openai.com/v1")).toHaveValue(
    "https://openrouter.ai/api/v1"
  );
  await expect(card.getByPlaceholder("gpt-4o-mini")).toBeVisible();

  // Re-picking the preset restores the preset view.
  await presetSelect.selectOption({ label: "OpenRouter" });
  await expect(presetSelect).toHaveValue("openrouter");
});

test("a custom base URL keeps the plain model text input", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");
  const card = page.getByTestId("binding-card-default");
  await card.getByRole("button", { name: /edit/i }).click();
  await card.getByRole("button", { name: "External API" }).click();

  // Custom (empty) base URL: plain model input, no curated free-model dropdown.
  await expect(card.getByPlaceholder("gpt-4o-mini")).toBeVisible();
  await expect(
    card.getByRole("option", { name: "meta-llama/llama-3.3-70b-instruct:free" })
  ).toHaveCount(0);

  // A filled non-preset URL keeps the same plain input.
  await card.getByPlaceholder("https://api.openai.com/v1").fill("https://api.example.com/v1");
  await expect(card.getByPlaceholder("gpt-4o-mini")).toBeVisible();
  await expect(
    card.getByRole("option", { name: "meta-llama/llama-3.3-70b-instruct:free" })
  ).toHaveCount(0);
});
