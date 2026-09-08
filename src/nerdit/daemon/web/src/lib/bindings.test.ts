import { describe, expect, it } from "vitest";
import type { Database, Model } from "../api/types";
import {
  bindingEnvKeys,
  bindingSpec,
  bindingValidity,
  dbBindingEnvKeys,
  dbBindingSpec,
  dbBindingValidity,
  dbDraftFromView,
  dbExternalReadiness,
  dbManagedReadiness,
  dbSpecFromView,
  deriveDbSecretName,
  deriveSecretName,
  describeDbBinding,
  draftFromView,
  isDbDirty,
  isDirty,
  modelReadiness,
  REDACTED_KEY,
  resolveKeyRef,
  secretRef,
  specFromView,
  type AiBindingDraft,
  type DbBindingDraft,
  type KeySource
} from "./bindings";

describe("deriveSecretName", () => {
  it("uppercases a binding name into AI_<NAME>_API_KEY", () => {
    expect(deriveSecretName("default")).toBe("AI_DEFAULT_API_KEY");
  });

  it("replaces every non-alphanumeric run char with an underscore", () => {
    expect(deriveSecretName("cheap-eu")).toBe("AI_CHEAP_EU_API_KEY");
  });

  it("passes an already-uppercase name straight through", () => {
    expect(deriveSecretName("PROD")).toBe("AI_PROD_API_KEY");
  });
});

describe("secretRef", () => {
  it("builds a plain per-service reference when not shared", () => {
    expect(secretRef("MY_KEY", false)).toBe("${secrets.MY_KEY}");
  });

  it("builds a shared-scope reference when shared", () => {
    expect(secretRef("MY_KEY", true)).toBe("${secrets.shared.MY_KEY}");
  });
});

// These env-key names mirror core/models/binding.py inject_env exactly: every
// binding gets NERDIT_AI_<NAME>_URL/_KEY/_MODEL (suffix _URL, NOT _BASE_URL),
// and only `default` additionally gets the plain OPENAI_* triplet.
describe("bindingEnvKeys", () => {
  it("gives `default` both the OPENAI_* triplet and the NERDIT_AI_DEFAULT_* triplet", () => {
    const keys = bindingEnvKeys("default");
    expect(keys).toContain("OPENAI_BASE_URL");
    expect(keys).toContain("OPENAI_API_KEY");
    expect(keys).toContain("OPENAI_MODEL");
    expect(keys).toContain("NERDIT_AI_DEFAULT_URL");
    expect(keys).toContain("NERDIT_AI_DEFAULT_KEY");
    expect(keys).toContain("NERDIT_AI_DEFAULT_MODEL");
  });

  it("gives a non-default binding only its NERDIT_AI_<NAME>_* triplet, no OPENAI_* keys", () => {
    const keys = bindingEnvKeys("cheap");
    expect(keys).toEqual(["NERDIT_AI_CHEAP_URL", "NERDIT_AI_CHEAP_KEY", "NERDIT_AI_CHEAP_MODEL"]);
    expect(keys.some((k) => k.startsWith("OPENAI_"))).toBe(false);
  });
});

// Two rows lifted from tests/e2e/fixtures.ts SAMPLE_MODELS shapes: a
// running+pulled model and a running+not-yet-pulled model.
const MODELS: Model[] = [
  {
    id: "mdl1cccc0003",
    name: "ollama-llama3-1-8b",
    model: "llama3.1:8b",
    backend: "ollama",
    status: "running",
    desired_state: "running",
    model_pulled: true,
    gpu_count: 1,
    gpu_ids: ["GPU-A100-001"],
    gpu_utilization: { "GPU-A100-001": 42 },
    endpoint: "http://127.0.0.1:38100/v1",
    created_at: "2026-07-04T08:00:00+00:00"
  },
  {
    id: "mdl2dddd0004",
    name: "ollama-qwen2-5-3b",
    model: "qwen2.5:3b",
    backend: "ollama",
    status: "running",
    desired_state: "running",
    model_pulled: false,
    gpu_count: 0,
    gpu_ids: [],
    gpu_utilization: {},
    endpoint: null,
    created_at: "2026-07-04T08:30:00+00:00"
  }
];

describe("modelReadiness", () => {
  it("is ready when the model is running and weights are pulled", () => {
    expect(modelReadiness("llama3.1:8b", MODELS)).toEqual({
      state: "ready",
      detail: "Model running, weights pulled."
    });
  });

  it("waits while the model is running but still pulling weights", () => {
    expect(modelReadiness("qwen2.5:3b", MODELS)).toEqual({
      state: "waiting",
      detail: "Model still pulling. App waits at launch."
    });
  });

  it("waits when the referenced model is not served at all", () => {
    expect(modelReadiness("missing:model", MODELS)).toEqual({
      state: "waiting",
      detail: "Model not served. App waits at launch."
    });
  });

  it("waits when no model is set (null ref)", () => {
    expect(modelReadiness(null, MODELS)).toEqual({
      state: "waiting",
      detail: "No model set. App waits at launch."
    });
  });
});

// Mirrors the ai.default entry of SAMPLE_APP_CONFIG (fixtures.ts): an ollama
// binding with null base_url + api_key.
const VIEW_ENTRY: Record<string, unknown> = {
  provider: "ollama",
  model: "llama3.1:8b",
  base_url: null,
  api_key: null
};

describe("bindingSpec / specFromView / isDirty", () => {
  it("omits empty keys — only provider + model, no base_url/api_key members", () => {
    const spec = bindingSpec(draftFromView("default", VIEW_ENTRY));
    expect(spec).toEqual({ provider: "ollama", model: "llama3.1:8b" });
    expect("base_url" in spec).toBe(false);
    expect("api_key" in spec).toBe(false);
  });

  it("normalizes a stored view entry to the same only-present-keys shape", () => {
    expect(specFromView(VIEW_ENTRY)).toEqual({ provider: "ollama", model: "llama3.1:8b" });
  });

  it("is not dirty for a round-trip draft built from the same view entry", () => {
    const draft = draftFromView("default", VIEW_ENTRY);
    expect(isDirty(draft, VIEW_ENTRY)).toBe(false);
  });

  it("is dirty after the provider changes", () => {
    const draft = { ...draftFromView("default", VIEW_ENTRY), provider: "api" };
    expect(isDirty(draft, VIEW_ENTRY)).toBe(true);
  });

  it("is dirty for a new binding (no original entry)", () => {
    const draft = draftFromView("default", VIEW_ENTRY);
    expect(isDirty(draft, undefined)).toBe(true);
  });
});

describe("bindingValidity", () => {
  const ollama: AiBindingDraft = {
    name: "default",
    provider: "ollama",
    model: "llama3.1:8b",
    baseUrl: "",
    apiKeyRef: ""
  };
  const api: AiBindingDraft = {
    name: "default",
    provider: "api",
    model: "gpt-4o-mini",
    baseUrl: "https://api.openai.com/v1",
    apiKeyRef: ""
  };
  const none: KeySource = { kind: "none" };

  it("accepts an ollama binding with a model set", () => {
    expect(bindingValidity(ollama, none, false)).toEqual({ valid: true, hint: null });
  });

  it("rejects an ollama binding with no model, naming the missing field", () => {
    expect(bindingValidity({ ...ollama, model: "  " }, none, false)).toEqual({
      valid: false,
      hint: "Model is required."
    });
  });

  it("rejects an api binding missing base_url", () => {
    expect(bindingValidity({ ...api, baseUrl: "" }, none, false)).toEqual({
      valid: false,
      hint: "Base URL is required."
    });
  });

  it("rejects an api binding whose only input is a pasted key (model/base_url empty)", () => {
    const draft: AiBindingDraft = { ...api, model: "", baseUrl: "" };
    const source: KeySource = { kind: "paste", secretName: "AI_DEFAULT_API_KEY", value: "sk-live" };
    // Model is checked before base_url, so the hint names model first.
    expect(bindingValidity(draft, source, false)).toEqual({
      valid: false,
      hint: "Model is required."
    });
  });

  it("accepts an api binding with model + base_url + a valid pasted secret name", () => {
    const source: KeySource = { kind: "paste", secretName: "AI_DEFAULT_API_KEY", value: "sk-live" };
    expect(bindingValidity(api, source, false)).toEqual({ valid: true, hint: null });
  });

  it("rejects a paste with an empty value", () => {
    const source: KeySource = { kind: "paste", secretName: "AI_DEFAULT_API_KEY", value: "  " };
    expect(bindingValidity(api, source, false)).toEqual({
      valid: false,
      hint: "Paste an API key."
    });
  });

  it("rejects a paste whose secret name violates the server key grammar", () => {
    const source: KeySource = { kind: "paste", secretName: "lower-case", value: "sk-live" };
    expect(bindingValidity(api, source, false)).toEqual({
      valid: false,
      hint: "Secret name must be uppercase."
    });
  });

  it("rejects a paste whose secret name has trailing whitespace (gate === write input)", () => {
    const source: KeySource = { kind: "paste", secretName: "MY_KEY ", value: "sk-live" };
    expect(bindingValidity(api, source, false)).toEqual({
      valid: false,
      hint: "Secret name must be uppercase."
    });
  });

  it("rejects an existing/shared source with no key picked", () => {
    expect(bindingValidity(api, { kind: "existing", key: "" }, false).valid).toBe(false);
    expect(bindingValidity(api, { kind: "shared", key: "" }, false).valid).toBe(false);
  });

  it("accepts an existing source with a key picked", () => {
    expect(bindingValidity(api, { kind: "existing", key: "AI_DEFAULT_API_KEY" }, false)).toEqual({
      valid: true,
      hint: null
    });
  });

  it("treats a redacted stored key as no key: an api edit must re-pick a source", () => {
    expect(bindingValidity({ ...api, apiKeyRef: REDACTED_KEY }, none, false)).toEqual({
      valid: false,
      hint: "Re-pick the API key."
    });
  });

  it("keeps an untouched valid stored ref (source none)", () => {
    expect(
      bindingValidity({ ...api, apiKeyRef: "${secrets.OPENAI_KEY}" }, none, false)
    ).toEqual({ valid: true, hint: null });
  });

  it("rejects a source none whose stored ref is not a secret reference", () => {
    expect(bindingValidity({ ...api, apiKeyRef: "" }, none, false)).toEqual({
      valid: false,
      hint: "API key is required."
    });
  });

  it("rejects a NEW binding whose name breaks the [ai] name rule", () => {
    expect(bindingValidity({ ...ollama, name: "Bad-Name" }, none, true)).toEqual({
      valid: false,
      hint: "Name must start with a lowercase letter."
    });
  });

  it("accepts a NEW binding with a valid lowercase name", () => {
    expect(bindingValidity({ ...ollama, name: "cheap_eu" }, none, true)).toEqual({
      valid: true,
      hint: null
    });
  });
});

describe("resolveKeyRef", () => {
  it("paste → derived ref + a secret write, and NEVER leaks the raw value into the ref", () => {
    const value = "sk-live-DONOTLEAK";
    const { ref, secretWrite } = resolveKeyRef(
      { kind: "paste", secretName: "AI_DEFAULT_API_KEY", value },
      ""
    );
    expect(ref).toBe("${secrets.AI_DEFAULT_API_KEY}");
    expect(secretWrite).toEqual({ name: "AI_DEFAULT_API_KEY", value });
    // The never-a-literal-key invariant (D3): the raw value appears nowhere in
    // the reference the ai section transports.
    expect(ref.includes(value)).toBe(false);
  });

  it("existing → a plain reference and no secret write", () => {
    expect(resolveKeyRef({ kind: "existing", key: "MY_KEY" }, "")).toEqual({
      ref: "${secrets.MY_KEY}",
      secretWrite: null
    });
  });

  it("shared → a shared-scope reference and no secret write", () => {
    expect(resolveKeyRef({ kind: "shared", key: "SHARED_KEY" }, "")).toEqual({
      ref: "${secrets.shared.SHARED_KEY}",
      secretWrite: null
    });
  });

  it("none → passes the current ref through unchanged and writes no secret", () => {
    expect(resolveKeyRef({ kind: "none" }, "${secrets.EXISTING}")).toEqual({
      ref: "${secrets.EXISTING}",
      secretWrite: null
    });
  });
});

// P15 [db.*] bindings — the type slot `BindingDescriptor` anticipated.

const DATABASES: Database[] = [
  {
    id: "db1aaaa0001",
    name: "pg",
    backend: "postgres",
    status: "running",
    desired_state: "running",
    db_ready: true,
    endpoint: "127.0.0.1:38200",
    created_at: "2026-07-14T08:00:00+00:00"
  },
  {
    id: "db2bbbb0002",
    name: "pg-warming-up",
    backend: "postgres",
    status: "running",
    desired_state: "running",
    db_ready: false,
    endpoint: null,
    created_at: "2026-07-14T08:05:00+00:00"
  }
];

describe("dbBindingEnvKeys", () => {
  it("gives a managed `default` binding NERDIT_DB_DEFAULT_URL plus the DATABASE_URL alias", () => {
    expect(dbBindingEnvKeys("default", { provider: "managed", database: "pg" })).toEqual([
      "NERDIT_DB_DEFAULT_URL",
      "DATABASE_URL"
    ]);
  });

  it("gives an external `default` postgres binding the DATABASE_URL alias", () => {
    const spec = { provider: "external", url: "postgresql://db.example.com:5432/app" };
    expect(dbBindingEnvKeys("default", spec)).toEqual(["NERDIT_DB_DEFAULT_URL", "DATABASE_URL"]);
  });

  it("gives an external `default` redis binding the REDIS_URL alias", () => {
    const spec = { provider: "external", url: "rediss://cache.example.com:6380" };
    expect(dbBindingEnvKeys("default", spec)).toEqual(["NERDIT_DB_DEFAULT_URL", "REDIS_URL"]);
  });

  it("gives a non-default binding only its NERDIT_DB_<NAME>_URL key, no alias", () => {
    const keys = dbBindingEnvKeys("cache", { provider: "managed", database: "redis-cache" });
    expect(keys).toEqual(["NERDIT_DB_CACHE_URL"]);
  });
});

describe("dbManagedReadiness", () => {
  it("is ready when the target row is running and db_ready", () => {
    expect(dbManagedReadiness("pg", DATABASES)).toEqual({
      state: "ready",
      detail: "Database running, ready."
    });
  });

  it("waits when the target row exists but is not yet db_ready", () => {
    expect(dbManagedReadiness("pg-warming-up", DATABASES)).toEqual({
      state: "waiting",
      detail: "Database still starting. App waits at launch."
    });
  });

  it("waits when the target row does not exist", () => {
    expect(dbManagedReadiness("missing-db", DATABASES)).toEqual({
      state: "waiting",
      detail: "Database not served. App waits at launch."
    });
  });

  it("waits when no database is set (null target)", () => {
    expect(dbManagedReadiness(null, DATABASES)).toEqual({
      state: "waiting",
      detail: "No database set. App waits at launch."
    });
  });
});

describe("dbExternalReadiness", () => {
  it("waits when the URL is not set", () => {
    expect(dbExternalReadiness(null, "${secrets.DB_PASSWORD}")).toEqual({
      state: "waiting",
      detail: "URL not set."
    });
  });

  it("waits when the password ref is not set", () => {
    expect(dbExternalReadiness("postgresql://db.example.com/app", null)).toEqual({
      state: "waiting",
      detail: "Password not set."
    });
  });

  it("is unknown (honest, not checked) once both are present", () => {
    expect(
      dbExternalReadiness("postgresql://db.example.com/app", "${secrets.DB_PASSWORD}")
    ).toEqual({ state: "unknown", detail: "External database. Not checked." });
  });
});

describe("describeDbBinding", () => {
  it("projects a managed binding: target is the database name, readiness from the live row", () => {
    const descriptor = describeDbBinding("default", { provider: "managed", database: "pg" }, DATABASES);
    expect(descriptor).toEqual({
      type: "db",
      name: "default",
      provider: "managed",
      target: "pg",
      readiness: { state: "ready", detail: "Database running, ready." },
      envKeys: ["NERDIT_DB_DEFAULT_URL", "DATABASE_URL"]
    });
  });

  it("projects an external binding: target is the credential-free URL, readiness unknown", () => {
    const spec = {
      provider: "external",
      url: "postgresql://db.example.com:5432/app",
      password: "${secrets.DB_PASSWORD}"
    };
    const descriptor = describeDbBinding("default", spec, DATABASES);
    expect(descriptor.target).toBe("postgresql://db.example.com:5432/app");
    expect(descriptor.readiness).toEqual({ state: "unknown", detail: "External database. Not checked." });
  });

  it("projects a managed binding whose target row is missing", () => {
    const descriptor = describeDbBinding("cache", { provider: "managed", database: "gone" }, DATABASES);
    expect(descriptor.target).toBe("gone");
    expect(descriptor.readiness).toEqual({
      state: "waiting",
      detail: "Database not served. App waits at launch."
    });
  });
});

// P15 [db.*] editor (PR3) — the write side twin of the ai editor helpers.

describe("deriveDbSecretName", () => {
  it("uppercases a binding name into DB_<NAME>_PASSWORD", () => {
    expect(deriveDbSecretName("default")).toBe("DB_DEFAULT_PASSWORD");
  });

  it("replaces every non-alphanumeric run char with an underscore", () => {
    expect(deriveDbSecretName("cache-eu")).toBe("DB_CACHE_EU_PASSWORD");
  });
});

describe("dbBindingSpec / dbSpecFromView / isDbDirty", () => {
  const managedView: Record<string, unknown> = {
    provider: "managed",
    database: "pg",
    url: null,
    password: null
  };
  const externalView: Record<string, unknown> = {
    provider: "external",
    database: null,
    url: "postgresql://db.example.com:5432/app",
    password: "${secrets.DB_DEFAULT_PASSWORD}"
  };

  it("managed → only provider + database, never url/password (cross-provider exclusion)", () => {
    const spec = dbBindingSpec(dbDraftFromView("default", managedView));
    expect(spec).toEqual({ provider: "managed", database: "pg" });
    expect("url" in spec).toBe(false);
    expect("password" in spec).toBe(false);
  });

  it("external → only provider + url + password, never database", () => {
    const spec = dbBindingSpec(dbDraftFromView("default", externalView));
    expect(spec).toEqual({
      provider: "external",
      url: "postgresql://db.example.com:5432/app",
      password: "${secrets.DB_DEFAULT_PASSWORD}"
    });
    expect("database" in spec).toBe(false);
  });

  it("omits an empty managed database (only-present-keys rule)", () => {
    const spec = dbBindingSpec({ name: "default", provider: "managed", database: "  ", url: "", passwordRef: "" });
    expect(spec).toEqual({ provider: "managed" });
  });

  it("normalizes a stored managed view entry to the same only-present-keys shape", () => {
    expect(dbSpecFromView(managedView)).toEqual({ provider: "managed", database: "pg" });
  });

  it("is not dirty for a round-trip draft built from the same managed view entry", () => {
    expect(isDbDirty(dbDraftFromView("default", managedView), managedView)).toBe(false);
  });

  it("is not dirty for a round-trip external view entry", () => {
    expect(isDbDirty(dbDraftFromView("default", externalView), externalView)).toBe(false);
  });

  it("is dirty after the provider flips managed → external", () => {
    const draft = { ...dbDraftFromView("default", managedView), provider: "external" };
    expect(isDbDirty(draft, managedView)).toBe(true);
  });

  it("is dirty for a new binding (no original entry)", () => {
    expect(isDbDirty(dbDraftFromView("default", managedView), undefined)).toBe(true);
  });
});

describe("dbBindingValidity", () => {
  const managed: DbBindingDraft = {
    name: "default",
    provider: "managed",
    database: "pg",
    url: "",
    passwordRef: ""
  };
  const external: DbBindingDraft = {
    name: "default",
    provider: "external",
    database: "",
    url: "postgresql://db.example.com:5432/app",
    passwordRef: "${secrets.DB_DEFAULT_PASSWORD}"
  };
  const none: KeySource = { kind: "none" };

  it("accepts a managed binding with a database set", () => {
    expect(dbBindingValidity(managed, none, false)).toEqual({ valid: true, hint: null });
  });

  it("rejects a managed binding with no database", () => {
    expect(dbBindingValidity({ ...managed, database: "  " }, none, false)).toEqual({
      valid: false,
      hint: "Database is required."
    });
  });

  it("accepts an external binding with a valid url + a stored password ref", () => {
    expect(dbBindingValidity(external, none, false)).toEqual({ valid: true, hint: null });
  });

  it("accepts a rediss url", () => {
    expect(
      dbBindingValidity({ ...external, url: "rediss://cache.example.com:6380" }, none, false)
    ).toEqual({ valid: true, hint: null });
  });

  it("rejects an external url with a bad scheme", () => {
    expect(dbBindingValidity({ ...external, url: "mysql://db/app" }, none, false)).toEqual({
      valid: false,
      hint: "URL scheme must be postgresql or redis."
    });
  });

  it("rejects an external url that embeds a userinfo password", () => {
    expect(
      dbBindingValidity({ ...external, url: "postgresql://u:secret@db/app" }, none, false)
    ).toEqual({ valid: false, hint: "URL must not embed a password." });
  });

  it("rejects an external url carrying a query string", () => {
    expect(
      dbBindingValidity({ ...external, url: "postgresql://db/app?sslmode=require" }, none, false)
    ).toEqual({ valid: false, hint: "URL must not carry a query string." });
  });

  it("rejects an unparseable external url", () => {
    expect(dbBindingValidity({ ...external, url: ":::" }, none, false)).toEqual({
      valid: false,
      hint: "URL could not be parsed."
    });
  });

  it("treats a redacted stored password as no key: an external edit must re-pick", () => {
    expect(dbBindingValidity({ ...external, passwordRef: REDACTED_KEY }, none, false)).toEqual({
      valid: false,
      hint: "Re-pick the password."
    });
  });

  it("keeps a valid stored password ref untouched (source none)", () => {
    expect(dbBindingValidity(external, none, false)).toEqual({ valid: true, hint: null });
  });

  it("rejects a source none whose stored password ref is not a secret reference", () => {
    expect(dbBindingValidity({ ...external, passwordRef: "" }, none, false)).toEqual({
      valid: false,
      hint: "Password is required."
    });
  });

  it("accepts a pasted password with a valid uppercase secret name", () => {
    const source: KeySource = { kind: "paste", secretName: "DB_DEFAULT_PASSWORD", value: "pw" };
    expect(dbBindingValidity({ ...external, passwordRef: "" }, source, false)).toEqual({
      valid: true,
      hint: null
    });
  });

  it("rejects a paste with an empty value, naming the password", () => {
    const source: KeySource = { kind: "paste", secretName: "DB_DEFAULT_PASSWORD", value: "  " };
    expect(dbBindingValidity({ ...external, passwordRef: "" }, source, false)).toEqual({
      valid: false,
      hint: "Paste a password."
    });
  });

  it("rejects a NEW binding whose name breaks the [db] name rule", () => {
    expect(dbBindingValidity({ ...managed, name: "Bad-Name" }, none, true)).toEqual({
      valid: false,
      hint: "Name must start with a lowercase letter."
    });
  });
});
