import type { Database, Model } from "../api/types";

/**
 * The mask `GET /config/apps/{name}` writes in place of any set `api_key`
 * (`config/redaction.py` REDACTED). A binding whose stored key is this sentinel
 * has a key, but its `${secrets.*}` reference is not disclosed to the client —
 * so the value cannot be echoed back on a write. Because `_merge_ai` replaces a
 * binding's whole spec (an omitted `api_key` deletes it), "keep the current
 * key" is inexpressible: the panel forces a fresh key-source pick instead.
 */
export const REDACTED_KEY = "***";

/**
 * A binding's launch-time reachability, kept honest: "unknown" is the steady
 * state for an external API (the WP3 probe is out of scope), never a failure.
 */
type BindingReadinessState = "ready" | "waiting" | "unknown";

/** Readiness state plus a one-line, sentence-case explanation for the card. */
export interface BindingReadiness {
  state: BindingReadinessState;
  detail: string;
}

/**
 * The generic card model (D2): `type` is "ai" | "db" (P15 — `describeDbBinding`
 * slots in a `[db.*]` binding without a redesign, exactly as anticipated).
 */
export interface BindingDescriptor {
  type: string;
  name: string;
  provider: string;
  target: string | null;
  readiness: BindingReadiness;
  envKeys: string[];
}

/**
 * Editable draft of one `[ai.*]` binding. `apiKeyRef` always holds a
 * `${secrets.*}` reference or "", never a raw key (D3 — literals stay rejected
 * server-side).
 */
export interface AiBindingDraft {
  name: string;
  provider: string;
  model: string;
  baseUrl: string;
  apiKeyRef: string;
}

/**
 * Derive the default secret name for a binding's API key (D3), e.g.
 * `default` → `AI_DEFAULT_API_KEY`. The user may edit it before the write.
 */
export function deriveSecretName(binding: string): string {
  return "AI_" + binding.toUpperCase().replace(/[^A-Z0-9]/g, "_") + "_API_KEY";
}

/** Build the `${secrets.*}` reference stored as a binding's `api_key` (D3). */
export function secretRef(key: string, shared: boolean): string {
  return shared ? "${secrets.shared." + key + "}" : "${secrets." + key + "}";
}

/**
 * The env keys a binding injects at launch — mirrors `core/models/binding.py`
 * `inject_env` exactly: every binding gets `NERDIT_AI_<NAME>_URL/_KEY/_MODEL`,
 * and `default` additionally gets the plain `OPENAI_*` triplet (listed first).
 */
export function bindingEnvKeys(name: string): string[] {
  const upper = name.toUpperCase();
  const keys: string[] = [];
  if (name === "default") {
    keys.push("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL");
  }
  keys.push(`NERDIT_AI_${upper}_URL`, `NERDIT_AI_${upper}_KEY`, `NERDIT_AI_${upper}_MODEL`);
  return keys;
}

/**
 * Honest readiness for a local (ollama) binding (D4): a not-yet-running model
 * means the app waits at next launch, per `BindingNotReady` semantics.
 */
export function modelReadiness(modelRef: string | null, models: Model[]): BindingReadiness {
  if (!modelRef) return { state: "waiting", detail: "No model set. App waits at launch." };
  const found = models.find((m) => m.model === modelRef);
  if (!found) return { state: "waiting", detail: "Model not served. App waits at launch." };
  if (found.status === "running" && found.model_pulled) {
    return { state: "ready", detail: "Model running, weights pulled." };
  }
  return { state: "waiting", detail: "Model still pulling. App waits at launch." };
}

/**
 * Readiness for an external (api) binding. "unknown" is the honest steady state
 * — the WP3 provider probe is out of scope, so a fully configured API is not
 * checked here.
 */
export function apiReadiness(
  baseUrl: string | null,
  apiKeyRef: string | null
): BindingReadiness {
  if (!baseUrl) return { state: "waiting", detail: "Base URL not set." };
  if (!apiKeyRef) return { state: "waiting", detail: "API key not set." };
  return { state: "unknown", detail: "External API. Not checked." };
}

/**
 * Serialize a draft for the PUT body: always `{provider}`, plus model/base_url/
 * api_key only when non-empty (mirrors `ConfigurePanel.spec` — a null-padded
 * spec never equals the stored one, so it would falsely read as changed).
 */
export function bindingSpec(draft: AiBindingDraft): Record<string, unknown> {
  const out: Record<string, unknown> = { provider: draft.provider };
  const model = draft.model.trim();
  const baseUrl = draft.baseUrl.trim();
  const apiKey = draft.apiKeyRef.trim();
  if (model) out.model = model;
  if (baseUrl) out.base_url = baseUrl;
  if (apiKey) out.api_key = apiKey;
  return out;
}

/**
 * Normalize a stored `view.ai` entry with the same only-present-keys rule
 * (mirrors `ConfigurePanel.origSpec`), so a diff against `bindingSpec` is exact.
 */
export function specFromView(binding: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { provider: asString(binding.provider) || "ollama" };
  const model = (binding.model as string | null) ?? null;
  const baseUrl = (binding.base_url as string | null) ?? null;
  const apiKey = (binding.api_key as string | null) ?? null;
  if (model != null) out.model = model;
  if (baseUrl != null) out.base_url = baseUrl;
  if (apiKey != null) out.api_key = apiKey;
  return out;
}

/** Seed an editable draft from a stored `view.ai` entry (null values → ""). */
export function draftFromView(name: string, binding: Record<string, unknown>): AiBindingDraft {
  return {
    name,
    provider: asString(binding.provider) || "ollama",
    model: asString(binding.model),
    baseUrl: asString(binding.base_url),
    apiKeyRef: asString(binding.api_key)
  };
}

/**
 * Whether a draft differs from its stored entry — a new binding (no original)
 * is always dirty; otherwise compare the normalized specs.
 */
export function isDirty(
  draft: AiBindingDraft,
  original: Record<string, unknown> | undefined
): boolean {
  if (original === undefined) return true;
  return JSON.stringify(bindingSpec(draft)) !== JSON.stringify(specFromView(original));
}

/**
 * Project a draft into the generic card model (D2): target + readiness follow
 * the provider (local model vs external API), env keys from `bindingEnvKeys`.
 */
export function describeBinding(draft: AiBindingDraft, models: Model[]): BindingDescriptor {
  const isOllama = draft.provider === "ollama";
  return {
    type: "ai",
    name: draft.name,
    provider: draft.provider,
    target: isOllama ? draft.model || null : draft.baseUrl || null,
    readiness: isOllama
      ? modelReadiness(draft.model || null, models)
      : apiReadiness(draft.baseUrl || null, draft.apiKeyRef || null),
    envKeys: bindingEnvKeys(draft.name)
  };
}

// ---------------------------------------------------------------------------
// [db.*] bindings (P15) — the type slot the comment above anticipated. Unlike
// the ai side there is no editor yet (the Bindings/DB page is read-only), so
// this operates directly on the stored `view.db[name]` spec rather than an
// editable draft.
// ---------------------------------------------------------------------------

/**
 * Scheme → env alias a `default` db binding additionally injects, mirroring
 * `core/data/binding.py` `_SCHEME_ALIAS` / `inject_db_env` exactly.
 */
const DB_SCHEME_ALIAS: Record<string, string> = {
  postgresql: "DATABASE_URL",
  redis: "REDIS_URL",
  rediss: "REDIS_URL"
};

/**
 * The env keys a `[db.*]` binding injects at launch — mirrors
 * `core/data/binding.py` `inject_db_env` exactly: every binding gets
 * `NERDIT_DB_<NAME>_URL`, and `default` additionally gets its scheme's alias
 * (`DATABASE_URL` for postgres/managed, `REDIS_URL` for redis). A managed spec
 * has no URL to read a scheme from, so it defaults to `DATABASE_URL` — same
 * best-effort fallback as the server's names-only path.
 */
export function dbBindingEnvKeys(name: string, spec: Record<string, unknown>): string[] {
  const keys = [`NERDIT_DB_${name.toUpperCase()}_URL`];
  if (name === "default") {
    let alias = "DATABASE_URL";
    if (spec.provider === "external") {
      const scheme = asString(spec.url).split("://")[0]?.toLowerCase() ?? "";
      alias = DB_SCHEME_ALIAS[scheme] ?? "DATABASE_URL";
    }
    keys.push(alias);
  }
  return keys;
}

/**
 * Honest readiness for a `managed` db binding: cross-references the live
 * `GET /databases` rows the same way `modelReadiness` cross-references served
 * models. Reuses the existing `ready`/`waiting` states — a missing row and a
 * not-yet-ready row are both an honest "waiting" (the app blocks at launch
 * either way), distinguished only by the detail text.
 */
export function dbManagedReadiness(
  databaseName: string | null,
  databases: Database[]
): BindingReadiness {
  if (!databaseName) return { state: "waiting", detail: "No database set. App waits at launch." };
  const found = databases.find((d) => d.name === databaseName);
  if (!found) return { state: "waiting", detail: "Database not served. App waits at launch." };
  if (found.status === "running" && found.db_ready) {
    return { state: "ready", detail: "Database running, ready." };
  }
  return { state: "waiting", detail: "Database still starting. App waits at launch." };
}

/**
 * Readiness for an `external` db binding. "unknown" is the honest steady
 * state, same posture as `apiReadiness` — no live wire-protocol probe runs
 * for an external database here.
 */
export function dbExternalReadiness(url: string | null, passwordRef: string | null): BindingReadiness {
  if (!url) return { state: "waiting", detail: "URL not set." };
  if (!passwordRef) return { state: "waiting", detail: "Password not set." };
  return { state: "unknown", detail: "External database. Not checked." };
}

/**
 * Project a stored `view.db[name]` spec into the generic card model (D2):
 * target + readiness follow the provider (managed row vs external URL), env
 * keys from `dbBindingEnvKeys`. `password` is only ever a `${secrets.*}` ref
 * or the "***" redaction sentinel here — never a raw value (mirrors `ai.*`).
 */
export function describeDbBinding(
  name: string,
  spec: Record<string, unknown>,
  databases: Database[]
): BindingDescriptor {
  const provider = asString(spec.provider) || "managed";
  const isManaged = provider === "managed";
  const databaseName = isManaged ? asString(spec.database) || null : null;
  const url = isManaged ? null : asString(spec.url) || null;
  const password = isManaged ? null : asString(spec.password) || null;
  return {
    type: "db",
    name,
    provider,
    target: isManaged ? databaseName : url,
    readiness: isManaged
      ? dbManagedReadiness(databaseName, databases)
      : dbExternalReadiness(url, password),
    envKeys: dbBindingEnvKeys(name, spec)
  };
}

/**
 * Server grammar mirrors (`config/project.py`) — the client validity gate must
 * match these exactly so a binding the config PUT would 422 can never reach the
 * secret write. Kept in sync with `AI_BINDING_NAME_RE` / `SECRET_REF_RE` there
 * (and the secret key-name rule the `${secrets.KEY}` ref embeds).
 */
const AI_BINDING_NAME_RE = /^[a-z][a-z0-9_]{0,31}$/;
const SECRET_KEY_RE = /^[A-Z][A-Z0-9_]*$/;
const SECRET_REF_RE = /^\$\{secrets\.(?:shared\.)?[A-Z][A-Z0-9_]*\}$/;

/** Whether a value is a well-formed `${secrets.*}` reference (grammar mirror). */
export function isSecretRef(value: string): boolean {
  return SECRET_REF_RE.test(value);
}

/** A binding's client-side validity plus a short hint naming the missing field. */
export interface BindingValidity {
  valid: boolean;
  hint: string | null;
}

/**
 * Whether a chosen key source resolves to a spec the server would accept, or a
 * hint naming what is missing. Source-generic — the three field-specific
 * messages (`paste`/`repick`/`required`) are injected so both the ai `api_key`
 * and the db `password` gates share this logic. The redaction sentinel is
 * treated as "no key": its `${secrets.*}` ref is unknown, so a fresh source is
 * required before any edit can apply.
 */
function keySourceHint(
  source: KeySource,
  currentRef: string,
  messages: { paste: string; repick: string; required: string }
): string | null {
  switch (source.kind) {
    case "paste":
      if (!source.value.trim()) return messages.paste;
      if (!SECRET_KEY_RE.test(source.secretName)) return "Secret name must be uppercase.";
      return null;
    case "existing":
    case "shared":
      return source.key ? null : "Select a secret.";
    case "none":
    default:
      if (currentRef === REDACTED_KEY) return messages.repick;
      return SECRET_REF_RE.test(currentRef) ? null : messages.required;
  }
}

/**
 * Whether the chosen api_key source resolves to a spec the server would accept,
 * or a hint naming what is missing. Mirrors `AiBindingConfig` (`api_key` must be
 * a `${secrets.*}` reference with an uppercase key).
 */
function apiKeyHint(source: KeySource, currentRef: string): string | null {
  return keySourceHint(source, currentRef, {
    paste: "Paste an API key.",
    repick: "Re-pick the API key.",
    required: "API key is required."
  });
}

/**
 * The client-side validity gate (P12.5 fix) — a pure mirror of the server's
 * `AiBindingConfig` + `[ai]` binding-name rules. Apply AND Preview stay disabled
 * until this passes, so the paste-once secret write is unreachable for a binding
 * the config PUT would reject 422 (which would otherwise orphan the secret).
 */
export function bindingValidity(
  draft: AiBindingDraft,
  source: KeySource,
  isNew: boolean
): BindingValidity {
  if (isNew && !AI_BINDING_NAME_RE.test(draft.name)) {
    return { valid: false, hint: "Name must start with a lowercase letter." };
  }
  if (!draft.model.trim()) return { valid: false, hint: "Model is required." };
  if (draft.provider === "api") {
    if (!draft.baseUrl.trim()) return { valid: false, hint: "Base URL is required." };
    const keyHint = apiKeyHint(source, draft.apiKeyRef);
    if (keyHint) return { valid: false, hint: keyHint };
  }
  return { valid: true, hint: null };
}

/**
 * Where a binding's `api_key` reference comes from (D3, WP2). "paste" carries a
 * raw value that is written once as a secret; the other kinds only reference an
 * already-stored name.
 */
export type KeySource =
  | { kind: "none" }
  | { kind: "paste"; secretName: string; value: string }
  | { kind: "existing"; key: string }
  | { kind: "shared"; key: string };

/**
 * Resolve a KeySource into the `${secrets.*}` ref the ai section carries, plus
 * the optional paste-once secret write. INVARIANT (D3): the raw pasted value
 * appears ONLY in `secretWrite.value` — never in `ref` — so the ai section
 * (including every dry-run preview) only ever transports a reference.
 */
export function resolveKeyRef(
  source: KeySource,
  currentRef: string
): { ref: string; secretWrite: { name: string; value: string } | null } {
  switch (source.kind) {
    case "paste":
      return {
        ref: secretRef(source.secretName, false),
        secretWrite: { name: source.secretName, value: source.value }
      };
    case "existing":
      return { ref: secretRef(source.key, false), secretWrite: null };
    case "shared":
      return { ref: secretRef(source.key, true), secretWrite: null };
    case "none":
    default:
      return { ref: currentRef, secretWrite: null };
  }
}

// ---------------------------------------------------------------------------
// [db.*] editor (PR3) — the write side of the db bindings, a literal twin of
// the ai editor above. Writes go through `PUT /config/apps/{name}/db`
// (binding-granularity merge, `null` deletes). A managed binding names a
// served `kind=database` row; an external binding carries a credential-free DSN
// plus a `${secrets.*}` password reference (never a literal — the server
// rejects one). Mirrors `DbBindingConfig` (config/project.py).
// ---------------------------------------------------------------------------

/**
 * Editable draft of one `[db.*]` binding. `passwordRef` always holds a
 * `${secrets.*}` reference, "", or the "***" redaction sentinel from the view —
 * never a raw value (the server rejects a literal password).
 */
export interface DbBindingDraft {
  name: string;
  provider: string;
  database: string;
  url: string;
  passwordRef: string;
}

/** Schemes a `[db.*]` external url may use — mirrors `_DB_URL_SCHEMES` server-side. */
const DB_URL_SCHEMES = new Set(["postgresql", "redis", "rediss"]);

/**
 * Derive the default secret name for a binding's password (twin of
 * `deriveSecretName`), e.g. `default` → `DB_DEFAULT_PASSWORD`.
 */
export function deriveDbSecretName(binding: string): string {
  return "DB_" + binding.toUpperCase().replace(/[^A-Z0-9]/g, "_") + "_PASSWORD";
}

/**
 * Serialize a db draft for the PUT body (only-present-keys rule, mirroring
 * `bindingSpec`). NEVER emits url/password on the managed path or database on
 * the external path — the server `extra="forbid"`s cross-provider keys.
 */
export function dbBindingSpec(draft: DbBindingDraft): Record<string, unknown> {
  if (draft.provider === "managed") {
    const out: Record<string, unknown> = { provider: "managed" };
    const database = draft.database.trim();
    if (database) out.database = database;
    return out;
  }
  const out: Record<string, unknown> = { provider: "external" };
  const url = draft.url.trim();
  const password = draft.passwordRef.trim();
  if (url) out.url = url;
  if (password) out.password = password;
  return out;
}

/**
 * Normalize a stored `view.db` entry with the same only-present-keys rule, so a
 * diff against `dbBindingSpec` is exact (twin of `specFromView`).
 */
export function dbSpecFromView(binding: Record<string, unknown>): Record<string, unknown> {
  const provider = asString(binding.provider) || "managed";
  if (provider === "managed") {
    const out: Record<string, unknown> = { provider: "managed" };
    const database = (binding.database as string | null) ?? null;
    if (database != null) out.database = database;
    return out;
  }
  const out: Record<string, unknown> = { provider: "external" };
  const url = (binding.url as string | null) ?? null;
  const password = (binding.password as string | null) ?? null;
  if (url != null) out.url = url;
  if (password != null) out.password = password;
  return out;
}

/** Seed an editable db draft from a stored `view.db` entry (null values → ""). */
export function dbDraftFromView(name: string, binding: Record<string, unknown>): DbBindingDraft {
  return {
    name,
    provider: asString(binding.provider) || "managed",
    database: asString(binding.database),
    url: asString(binding.url),
    passwordRef: asString(binding.password)
  };
}

/**
 * Whether a db draft differs from its stored entry — a new binding (no
 * original) is always dirty; otherwise compare the normalized specs.
 */
export function isDbDirty(
  draft: DbBindingDraft,
  original: Record<string, unknown> | undefined
): boolean {
  if (original === undefined) return true;
  return JSON.stringify(dbBindingSpec(draft)) !== JSON.stringify(dbSpecFromView(original));
}

/** Validate an external db `url` client-side (mirrors `DbBindingConfig._check_url_shape`). */
function dbUrlHint(url: string): string | null {
  const trimmed = url.trim();
  if (!trimmed) return "URL is required.";
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return "URL could not be parsed.";
  }
  const scheme = parsed.protocol.replace(/:$/, "").toLowerCase();
  if (!DB_URL_SCHEMES.has(scheme)) return "URL scheme must be postgresql or redis.";
  if (parsed.password) return "URL must not embed a password.";
  if (parsed.search) return "URL must not carry a query string.";
  return null;
}

/**
 * The client-side validity gate for a `[db.*]` binding — a pure mirror of the
 * server's `DbBindingConfig` + `[db]` binding-name rules. Apply/Preview stay
 * disabled until this passes, so the paste-once password write is unreachable
 * for a binding the config PUT would reject 422.
 */
export function dbBindingValidity(
  draft: DbBindingDraft,
  source: KeySource,
  isNew: boolean
): BindingValidity {
  if (isNew && !AI_BINDING_NAME_RE.test(draft.name)) {
    return { valid: false, hint: "Name must start with a lowercase letter." };
  }
  if (draft.provider === "managed") {
    if (!draft.database.trim()) return { valid: false, hint: "Database is required." };
    return { valid: true, hint: null };
  }
  const urlHint = dbUrlHint(draft.url);
  if (urlHint) return { valid: false, hint: urlHint };
  const pwHint = keySourceHint(source, draft.passwordRef, {
    paste: "Paste a password.",
    repick: "Re-pick the password.",
    required: "Password is required."
  });
  if (pwHint) return { valid: false, hint: pwHint };
  return { valid: true, hint: null };
}

/** Coerce an unknown config value to a trimmed-safe string ("" for null/undefined). */
function asString(value: unknown): string {
  return value == null ? "" : String(value);
}
