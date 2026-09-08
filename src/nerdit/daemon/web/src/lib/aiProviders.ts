import { isSecretRef, REDACTED_KEY, type KeySource } from "./bindings";

/**
 * External-API provider presets (OpenRouter friction-reduction). Presentation-only
 * sugar over the existing free-text `provider = "api"` path: a preset fills the
 * base URL, wires the conventional secret name, and offers a curated free-model
 * dropdown. The frozen `[ai.*]` contract is untouched — no new provider value,
 * no schema change. The selected preset is DERIVED from `draft.baseUrl`, never a
 * stored field. Only OpenRouter ships as a preset (owner decision, beta): every
 * other endpoint rides the "Custom" path with the full key-source picker.
 */
export interface ApiProviderPreset {
  id: string; // stable, used as <option value>
  label: string;
  baseUrl: string;
  /** Conventional secret name the preset's key flow is wired to (passes SECRET_KEY_RE). */
  secretName: string;
  modelPlaceholder: string; // placeholder for the free-text model input
  freeModels: string[]; // curated dropdown; "Other model name" is the escape hatch
}

/** The known external-API providers, in display order (Custom is appended in the UI). */
export const API_PROVIDER_PRESETS: ApiProviderPreset[] = [
  {
    id: "openrouter",
    label: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    secretName: "OPENROUTER_API_KEY",
    modelPlaceholder: "meta-llama/llama-3.3-70b-instruct:free",
    freeModels: [
      // Curated from GET https://openrouter.ai/api/v1/models on 2026-07-18
      // (`:free` variants, well-known providers, general instruct models).
      // The list WILL go stale — "Other model name" is the permanent escape
      // hatch; refresh opportunistically, never fetched live from the SPA.
      "meta-llama/llama-3.3-70b-instruct:free",
      "openai/gpt-oss-20b:free",
      "qwen/qwen3-next-80b-a3b-instruct:free",
      "qwen/qwen3-coder:free",
      "google/gemma-4-31b-it:free",
      "nvidia/nemotron-3-super-120b-a12b:free",
      "nousresearch/hermes-3-llama-3.1-405b:free",
      "meta-llama/llama-3.2-3b-instruct:free"
    ]
  }
];

/** The preset whose base URL exactly matches (after `trim()`), else null. */
export function presetForBaseUrl(baseUrl: string): ApiProviderPreset | null {
  const trimmed = baseUrl.trim();
  return API_PROVIDER_PRESETS.find((p) => p.baseUrl === trimmed) ?? null;
}

/**
 * The effective key source for a preset-managed binding: the preset owns the
 * key convention, so the user is only ever asked to paste when no key is
 * available. Resolution order (first hit wins):
 *
 * 1. the user already interacted (source is not "none") — respect it;
 * 2. the stored ref is a valid `${secrets.*}` reference — keep it untouched
 *    (an existing binding is never silently re-pointed);
 * 3. the shared scope holds the conventional secret — reference it, nothing
 *    to paste (`${secrets.shared.<NAME>}`);
 * 4. otherwise a paste seeded with the conventional name (written as a
 *    per-app secret; the unscoped ref keeps today's paste-once machinery).
 *
 * A redacted stored key ("***") falls through 2 — its ref is undisclosed, so
 * the shared convention (or a fresh paste) honestly replaces it.
 */
export function presetKeySource(
  preset: ApiProviderPreset,
  source: KeySource,
  currentRef: string,
  sharedSecretKeys: string[]
): KeySource {
  if (source.kind !== "none") return source;
  if (currentRef !== REDACTED_KEY && isSecretRef(currentRef)) return source;
  if (sharedSecretKeys.includes(preset.secretName)) {
    return { kind: "shared", key: preset.secretName };
  }
  return { kind: "paste", secretName: preset.secretName, value: "" };
}
