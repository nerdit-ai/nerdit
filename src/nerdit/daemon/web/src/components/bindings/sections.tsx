import type { ReactNode } from "react";
import type { AppConfigView, Database, Model } from "../../api/types";
import { useDatabases, useModels } from "../../api/queries";
import {
  bindingSpec, bindingValidity, dbBindingSpec, dbBindingValidity, dbDraftFromView,
  deriveDbSecretName, describeBinding, describeDbBinding, draftFromView, isDbDirty, isDirty,
  type AiBindingDraft, type BindingDescriptor, type BindingValidity, type DbBindingDraft,
  type KeySource
} from "../../lib/bindings";
import { presetForBaseUrl, presetKeySource, type ApiProviderPreset } from "../../lib/aiProviders";
import { TextField, type CatalogOption } from "./BindingSwitch";
import { AiEndpointFields } from "./AiEndpointFields";

/** One side of the switch: the provider it selects, and how it clears the other side. */
export interface BindingPath<TDraft> {
  provider: string;
  label: string;
  patch: Partial<TDraft>;
}

/**
 * Everything that differs between `[ai.*]` and `[db.*]`, in one place. `TDraft`
 * is the section's editable-draft shape; `TItem` is the live-row type (`Model` |
 * `Database`) readiness is cross-referenced against. The panel, the switch, the
 * card and the key flow are generic and read only this.
 */
export interface BindingSectionDescriptor<TDraft extends { name: string; provider: string }, TItem> {
  section: "ai" | "db";
  copy: { intro: string; empty: string };
  useItems: () => { items: TItem[] };
  getSection: (view: AppConfigView) => Record<string, Record<string, unknown>>;
  draftFromView: (name: string, binding: Record<string, unknown>) => TDraft;
  newDraft: (name: string) => TDraft;
  spec: (draft: TDraft) => Record<string, unknown>;
  isDirty: (draft: TDraft, original: Record<string, unknown> | undefined) => boolean;
  validity: (draft: TDraft, source: KeySource, isNew: boolean) => BindingValidity;
  describe: (draft: TDraft, items: TItem[]) => BindingDescriptor;
  keyRef: (draft: TDraft) => string;
  withKeyRef: (draft: TDraft, ref: string) => TDraft;
  /** Preset-aware source resolution (ai only); db returns the raw source as-is. */
  resolveSource: (draft: TDraft, raw: KeySource, sharedSecretKeys: string[]) => KeySource;
  /** [catalog path, endpoint path], in the order the segmented control renders. */
  paths: [BindingPath<TDraft>, BindingPath<TDraft>];
  catalog: {
    label: string;
    placeholder: string;
    otherLabel?: string;
    otherPlaceholder?: string;
    options: (items: TItem[]) => CatalogOption[];
  };
  catalogValue: (draft: TDraft) => string;
  catalogPatch: (value: string) => Partial<TDraft>;
  /** The key/password picker's wiring on the endpoint path. */
  key: {
    label: string;
    deriveName?: (binding: string) => string;
    preset?: (draft: TDraft) => ApiProviderPreset | null;
  };
  /** The endpoint path's own fields; the key picker is appended by the switch. */
  renderEndpointFields: (props: { draft: TDraft; onChange: (p: Partial<TDraft>) => void }) => ReactNode;
}

export const aiSection: BindingSectionDescriptor<AiBindingDraft, Model> = {
  section: "ai",
  copy: {
    intro: "Wire this app to a model. Injected as env at launch.",
    empty: "Nothing wired yet. Add one to give this app a model."
  },
  useItems: () => ({ items: useModels().data?.items ?? [] }),
  getSection: (view) => view.ai,
  draftFromView,
  newDraft: (name) => ({ name, provider: "ollama", model: "", baseUrl: "", apiKeyRef: "" }),
  spec: bindingSpec,
  isDirty,
  validity: bindingValidity,
  describe: describeBinding,
  keyRef: (draft) => draft.apiKeyRef,
  withKeyRef: (draft, ref) => ({ ...draft, apiKeyRef: ref }),
  resolveSource: (draft, raw, sharedSecretKeys) => {
    const preset = draft.provider === "api" ? presetForBaseUrl(draft.baseUrl) : null;
    return preset ? presetKeySource(preset, raw, draft.apiKeyRef, sharedSecretKeys) : raw;
  },
  paths: [
    { provider: "ollama", label: "Local model", patch: { baseUrl: "", apiKeyRef: "" } },
    { provider: "api", label: "External API", patch: {} }
  ],
  catalog: {
    label: "Model",
    placeholder: "Select a model",
    otherLabel: "Other model name",
    otherPlaceholder: "llama3.1:8b",
    options: (models) =>
      models
        .filter((m): m is Model & { model: string } => Boolean(m.model))
        .map((m) => ({ value: m.model, label: m.model_pulled ? m.model : `${m.model} (pulling)` }))
  },
  catalogValue: (draft) => draft.model,
  catalogPatch: (model) => ({ model }),
  key: { label: "API key", preset: (draft) => presetForBaseUrl(draft.baseUrl) },
  renderEndpointFields: ({ draft, onChange }) => <AiEndpointFields draft={draft} onChange={onChange} />
};

export const dbSection: BindingSectionDescriptor<DbBindingDraft, Database> = {
  section: "db",
  copy: {
    intro: "Wire this app to a database. Injected as env at launch.",
    empty: "Nothing wired yet. Add one to give this app a database."
  },
  useItems: () => ({ items: useDatabases().data?.items ?? [] }),
  getSection: (view) => view.db ?? {},
  draftFromView: dbDraftFromView,
  newDraft: (name) => ({ name, provider: "managed", database: "", url: "", passwordRef: "" }),
  spec: dbBindingSpec,
  isDirty: isDbDirty,
  validity: dbBindingValidity,
  describe: (draft, items) => describeDbBinding(draft.name, dbBindingSpec(draft), items),
  keyRef: (draft) => draft.passwordRef,
  withKeyRef: (draft, ref) => ({ ...draft, passwordRef: ref }),
  resolveSource: (_draft, raw) => raw,
  paths: [
    { provider: "managed", label: "Managed", patch: { url: "", passwordRef: "" } },
    { provider: "external", label: "External", patch: { database: "" } }
  ],
  catalog: {
    label: "Database",
    placeholder: "Select a database",
    options: (databases) =>
      databases.map((d) => ({
        value: d.name,
        label: `${d.name}${d.backend ? ` · ${d.backend}` : ""}${d.db_ready ? "" : " (starting)"}`
      }))
  },
  catalogValue: (draft) => draft.database,
  catalogPatch: (database) => ({ database }),
  key: { label: "Password", deriveName: deriveDbSecretName },
  renderEndpointFields: ({ draft, onChange }) => (
    <TextField
      label="URL"
      value={draft.url}
      placeholder="postgresql://user@host:5432/db"
      onChange={(url) => onChange({ url })}
    />
  )
};
