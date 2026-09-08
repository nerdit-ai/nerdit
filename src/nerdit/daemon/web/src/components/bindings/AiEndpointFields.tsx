import { useState } from "react";
import { API_PROVIDER_PRESETS, presetForBaseUrl } from "../../lib/aiProviders";
import type { AiBindingDraft } from "../../lib/bindings";
import { Field, Select } from "../ui";
import { CatalogSelect, TextField } from "./BindingSwitch";

/**
 * The external-API fields. The active preset is DERIVED from `draft.baseUrl`,
 * never stored; the single flag exists so "Custom" stays picked while the URL
 * still matches a preset (the operator wants to edit it from there).
 */
export function AiEndpointFields({
  draft,
  onChange
}: {
  draft: AiBindingDraft;
  onChange: (p: Partial<AiBindingDraft>) => void;
}) {
  const [customUrl, setCustomUrl] = useState(false);
  const preset = customUrl ? null : presetForBaseUrl(draft.baseUrl);

  return (
    <>
      <Field label="Provider">
        <Select
          value={preset?.id ?? "custom"}
          onChange={(event) => {
            const picked = API_PROVIDER_PRESETS.find((p) => p.id === event.target.value);
            // A preset only fills the base URL: it must never clear a typed
            // model, and "Custom" leaves the URL untouched to edit.
            setCustomUrl(!picked);
            if (picked) onChange({ baseUrl: picked.baseUrl });
          }}
        >
          {API_PROVIDER_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
          <option value="custom">Custom</option>
        </Select>
      </Field>

      <TextField
        label="Base URL"
        value={draft.baseUrl}
        placeholder="https://api.openai.com/v1"
        onChange={(baseUrl) => onChange({ baseUrl })}
      />

      {preset ? (
        <div className="space-y-2">
          <CatalogSelect
            label="Model"
            placeholder="Select a model"
            otherLabel="Other model name"
            otherPlaceholder={preset.modelPlaceholder}
            options={preset.freeModels.map((m) => ({ value: m, label: m }))}
            value={draft.model}
            onChange={(model) => onChange({ model })}
          />
          <p className="text-12 text-subtle-foreground">Free tier, rate-limited by {preset.label}.</p>
        </div>
      ) : (
        <TextField
          label="Model"
          value={draft.model}
          placeholder="gpt-4o-mini"
          onChange={(model) => onChange({ model })}
        />
      )}
    </>
  );
}
