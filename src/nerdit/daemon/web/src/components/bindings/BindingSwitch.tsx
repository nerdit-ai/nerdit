import { useState } from "react";
import { Button, Field, Input, Select } from "../ui";
import { KeyFlow } from "./KeyFlow";
import type { BindingSectionDescriptor } from "./sections";
import type { KeySource } from "../../lib/bindings";

/** Sentinel `<option>` value for the free-text escape hatch. */
const OTHER = "__other__";

export interface CatalogOption {
  value: string;
  label: string;
}

/** A labelled Mono text input, the shape every endpoint field takes. */
export function TextField(props: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <Field label={props.label}>
      <Input
        mono
        value={props.value}
        placeholder={props.placeholder}
        onChange={(event) => props.onChange(event.target.value)}
      />
    </Field>
  );
}

/**
 * A catalog picker with a free-text escape hatch, shared by every path that
 * offers "pick a known one, or name your own": served models, provisioned
 * databases, a preset's curated model list.
 *
 * The one piece of local state is the escape hatch itself, and it exists for a
 * reason no derivation covers: picking "Other" clears the value, and an empty
 * value is indistinguishable from "nothing picked yet". Everything else is
 * derived, so a value outside the catalog opens the text input on its own.
 */
export function CatalogSelect({
  label,
  value,
  options,
  placeholder,
  otherLabel,
  otherPlaceholder,
  onChange
}: {
  label: string;
  value: string;
  options: CatalogOption[];
  placeholder: string;
  /** Omitted: no escape hatch (the db catalog names a real row or nothing). */
  otherLabel?: string;
  otherPlaceholder?: string;
  onChange: (value: string) => void;
}) {
  const [other, setOther] = useState(false);
  const known = options.some((option) => option.value === value);
  const selected = other ? OTHER : value === "" ? "" : known ? value : OTHER;

  return (
    <div className="space-y-2">
      <Field label={label}>
        <Select
          value={selected}
          onChange={(event) => {
            const next = event.target.value;
            setOther(next === OTHER);
            onChange(next === OTHER ? "" : next);
          }}
        >
          <option value="">{placeholder}</option>
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
          {otherLabel && <option value={OTHER}>{otherLabel}</option>}
        </Select>
      </Field>
      {otherLabel && selected === OTHER && (
        <Input
          mono
          value={value}
          placeholder={otherPlaceholder}
          onChange={(event) => onChange(event.target.value)}
        />
      )}
    </div>
  );
}

/**
 * The one two-path editor behind both `[ai.*]` and `[db.*]`.
 *
 * Path A picks a live row from this daemon's own catalog (a served model, a
 * provisioned database) and stays honest about a row that is not ready yet.
 * Path B edits the endpoint fields for something this daemon does not run and
 * carries the key/password picker. Which is which, what the paths are called
 * and what path B contains all come off the section descriptor, so this
 * component knows nothing about models or databases.
 */
export function BindingSwitch<TDraft extends { name: string; provider: string }, TItem>(props: {
  descriptor: BindingSectionDescriptor<TDraft, TItem>;
  draft: TDraft;
  bindingName: string;
  items: TItem[];
  source: KeySource;
  secretKeys: string[];
  sharedSecretKeys: string[];
  onChange: (patch: Partial<TDraft>) => void;
  onKeySourceChange: (next: KeySource) => void;
}) {
  const { descriptor, draft, onChange } = props;
  const { catalog, key, paths } = descriptor;

  return (
    <div className="space-y-3">
      <div
        role="group"
        aria-label="Provider"
        className="inline-flex gap-1 rounded-button border border-border bg-surface-hover p-0.5"
      >
        {paths.map((path) => (
          <Button
            key={path.provider}
            size="sm"
            variant={draft.provider === path.provider ? "primary" : "ghost"}
            aria-pressed={draft.provider === path.provider}
            onClick={() =>
              onChange({ ...path.patch, provider: path.provider } as unknown as Partial<TDraft>)
            }
          >
            {path.label}
          </Button>
        ))}
      </div>

      {draft.provider === paths[0].provider ? (
        <CatalogSelect
          label={catalog.label}
          placeholder={catalog.placeholder}
          otherLabel={catalog.otherLabel}
          otherPlaceholder={catalog.otherPlaceholder}
          options={catalog.options(props.items)}
          value={descriptor.catalogValue(draft)}
          onChange={(value) => onChange(descriptor.catalogPatch(value))}
        />
      ) : (
        <div className="space-y-3">
          {descriptor.renderEndpointFields({ draft, onChange })}
          <KeyFlow
            bindingName={props.bindingName}
            label={key.label}
            deriveName={key.deriveName}
            preset={key.preset?.(draft) ?? null}
            currentRef={descriptor.keyRef(draft)}
            secretKeys={props.secretKeys}
            sharedSecretKeys={props.sharedSecretKeys}
            source={props.source}
            onChange={props.onKeySourceChange}
          />
        </div>
      )}
    </div>
  );
}
