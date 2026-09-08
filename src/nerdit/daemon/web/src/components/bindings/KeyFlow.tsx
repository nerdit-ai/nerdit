import { Checkbox, Field, Input, Mono, Select } from "../ui";
import type { ApiProviderPreset } from "../../lib/aiProviders";
import { deriveSecretName, isSecretRef, REDACTED_KEY, type KeySource } from "../../lib/bindings";

export interface KeyFlowProps {
  bindingName: string;
  /** The stored `${secrets.*}` ref, shown while the source is untouched. */
  currentRef: string;
  /** Names-only lists from `GET /secrets/{service}` and `GET /secrets/shared`. */
  secretKeys: string[];
  sharedSecretKeys: string[];
  source: KeySource;
  onChange: (source: KeySource) => void;
  /** Paste-once secret name convention (ai default; db passes its own). */
  deriveName?: (binding: string) => string;
  label?: string;
  /** When set, the provider preset owns the convention and the picker collapses. */
  preset?: ApiProviderPreset | null;
}

/**
 * The one key/password source picker for both sections.
 *
 * Controlled and fetch-free: the parent resolves the chosen source into a
 * paste-once secret write plus a `${secrets.*}` reference. A raw pasted value
 * lives only here and in the secret write body, never in the ai/db section.
 */
export function KeyFlow({
  bindingName,
  currentRef,
  secretKeys,
  sharedSecretKeys,
  source,
  onChange,
  deriveName = deriveSecretName,
  label = "API key",
  preset = null
}: KeyFlowProps) {
  if (preset) {
    return (
      <PresetKey
        preset={preset}
        sharedAvailable={sharedSecretKeys.includes(preset.secretName)}
        currentRef={currentRef}
        source={source}
        onChange={onChange}
      />
    );
  }

  const radioName = `keyflow-${bindingName}`;
  const radio = (kind: KeySource["kind"], text: string, pick: () => void) => (
    <Checkbox type="radio" name={radioName} label={text} checked={source.kind === kind} onChange={pick} />
  );
  /** The two reference-only sources are the same control twice. */
  const reference = (kind: "existing" | "shared", text: string, keys: string[], empty: string) => (
    <>
      {radio(kind, text, () => onChange({ kind, key: keys[0] ?? "" }))}
      {source.kind === kind && (
        <div className="pl-6">
          {keys.length === 0 ? (
            <p className="text-12 text-subtle-foreground">{empty}</p>
          ) : (
            <Select
              data-testid={`keyflow-${kind}-key`}
              value={source.key}
              className="font-mono text-13"
              onChange={(event) => onChange({ kind, key: event.target.value })}
            >
              {keys.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </Select>
          )}
        </div>
      )}
    </>
  );

  return (
    <div className="space-y-2">
      <p className="label">{label}</p>

      {source.kind === "none" && currentRef && currentRef !== REDACTED_KEY && (
        <p className="text-12 text-subtle-foreground">
          <Mono className="text-muted-foreground">{currentRef}</Mono> Keep the current reference or
          choose below.
        </p>
      )}
      {source.kind === "none" && currentRef === REDACTED_KEY && (
        <p className="text-12 text-warning-foreground">
          Stored key is hidden. Re-pick a source to apply edits.
        </p>
      )}

      {radio("paste", "Paste a new key", () => {
        if (source.kind !== "paste") {
          onChange({ kind: "paste", secretName: deriveName(bindingName), value: "" });
        }
      })}
      {source.kind === "paste" && (
        <div className="space-y-2 pl-6">
          <Input
            type="password"
            autoComplete="new-password"
            data-testid="keyflow-paste-value"
            value={source.value}
            placeholder="sk-..."
            onChange={(event) => onChange({ ...source, value: event.target.value })}
          />
          <Input
            mono
            data-testid="keyflow-paste-name"
            value={source.secretName}
            onChange={(event) => onChange({ ...source, secretName: event.target.value })}
          />
          <p className="text-12 text-subtle-foreground">
            Stored once as a secret. Value never shown again.
          </p>
        </div>
      )}

      {reference("existing", "Use an existing secret", secretKeys, "No secrets set for this app yet.")}
      {reference("shared", "Use a shared secret", sharedSecretKeys, "No shared secrets available.")}
      {source.kind === "shared" && sharedSecretKeys.length > 0 && (
        <p className="pl-6 text-12 text-subtle-foreground">
          Shared secrets are referenced, never copied.
        </p>
      )}
    </div>
  );
}

/**
 * The preset-managed flow: a key is only ever asked for when none is available.
 * The three states mirror `presetKeySource` — a kept stored ref, the shared
 * convention (nothing to paste), or one paste seeded with the convention.
 */
function PresetKey({
  preset,
  sharedAvailable,
  currentRef,
  source,
  onChange
}: {
  preset: ApiProviderPreset;
  sharedAvailable: boolean;
  currentRef: string;
  source: KeySource;
  onChange: (source: KeySource) => void;
}) {
  const kept = source.kind === "none" && currentRef !== REDACTED_KEY && isSecretRef(currentRef);
  if (kept || source.kind === "shared") {
    return (
      <Field label="API key">
        <p className="text-12 text-subtle-foreground">
          {kept ? (
            <>
              <Mono className="text-muted-foreground">{currentRef}</Mono> Current key reference.
            </>
          ) : (
            <>
              Uses the shared <Mono className="text-muted-foreground">{preset.secretName}</Mono>{" "}
              secret. Nothing to paste.
            </>
          )}
        </p>
      </Field>
    );
  }
  return (
    <Field
      label={`${preset.label} API key`}
      hint={`Stored once as the app secret ${preset.secretName}. Value never shown again.${
        sharedAvailable ? "" : " Admins can share one key node-wide from Settings."
      }`}
    >
      <Input
        type="password"
        autoComplete="new-password"
        data-testid="keyflow-paste-value"
        placeholder="sk-..."
        value={source.kind === "paste" ? source.value : ""}
        onChange={(event) =>
          onChange({ kind: "paste", secretName: preset.secretName, value: event.target.value })
        }
      />
    </Field>
  );
}
