import { KeyRound, Pencil, Plus, Trash2 } from "lucide-react";
import { apiErrorCopy } from "../../lib/apiErrors";
import { resolveKeyRef } from "../../lib/bindings";
import { Button, Confirm, Input, Mono, Notice } from "../ui";
import { BindingSwitch } from "./BindingSwitch";
import { useBindingDraft, type UseBindingDraftResult } from "./useBindingDraft";
import type { BindingSectionDescriptor } from "./sections";

/** The readiness word carries its own colour; a dot beside it would say it twice. */
const READINESS_CLASS: Record<string, string> = {
  ready: "text-success-foreground",
  waiting: "text-warning-foreground",
  unknown: "text-subtle-foreground"
};

const ENV_KEY_CLASS =
  "flex items-center gap-1.5 rounded-button bg-surface px-2 py-1 font-mono text-13 text-foreground";

/** Leaving the add row: drop both the flag and the half-typed name. */
function cancelAdd(state: { setAddingNew: (v: boolean) => void; setNewName: (v: string) => void }) {
  state.setAddingNew(false);
  state.setNewName("");
}

/**
 * One panel for both `[ai.*]` and `[db.*]`.
 *
 * Each card writes only its own `{ [name]: spec }` (or `{ [name]: null }` to
 * delete) through `PUT /config/apps/{name}/{section}`, so two people editing
 * different resources never collide, and a collision on the same one is caught
 * by the config ETag rather than silently overwritten. Everything that differs
 * between the two sections is looked up on `descriptor`.
 */
export function GenericBindingsPanel<TDraft extends { name: string; provider: string }, TItem>({
  name,
  descriptor
}: {
  /** Service name: app config, and its resources, are keyed by name. */
  name: string;
  descriptor: BindingSectionDescriptor<TDraft, TItem>;
}) {
  const state = useBindingDraft(name, descriptor);
  const empty = state.draftNames.length === 0 && !state.addingNew;

  return (
    <div className="space-y-4">
      <p className="text-13 text-subtle-foreground">{descriptor.copy.intro}</p>

      {/* Quiet placeholder in the card's own shape, never the word "Loading". */}
      {state.configLoading && <div aria-hidden="true" className="h-20 rounded-card bg-surface-hover" />}
      {state.configError && <Notice tone="destructive">{apiErrorCopy(state.configError)}</Notice>}
      {state.banner && <Notice tone={state.banner.tone}>{state.banner.text}</Notice>}

      {state.view && (
        <>
          {empty && <p className="text-13 text-muted-foreground">{descriptor.copy.empty}</p>}

          {state.draftNames.map((draftName) => (
            <BindingCard key={draftName} draftName={draftName} descriptor={descriptor} state={state} />
          ))}

          {state.canWrite && !state.addingNew && (
            <Button size="sm" onClick={() => state.setAddingNew(true)}>
              <Plus aria-hidden="true" className="h-4 w-4" /> Add resource
            </Button>
          )}
          {state.canWrite && state.addingNew && (
            <div className="flex items-center gap-2 rounded-card bg-surface-hover p-3">
              <Input
                mono
                autoFocus
                aria-label="Resource name"
                placeholder="default"
                value={state.newName}
                onChange={(event) => state.setNewName(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && state.confirmAddBinding()}
              />
              <Button size="sm" variant="primary" onClick={state.confirmAddBinding}>
                Add
              </Button>
              <Button size="sm" variant="ghost" onClick={() => cancelAdd(state)}>
                Cancel
              </Button>
            </div>
          )}
        </>
      )}

      <Confirm
        open={state.deleting !== null}
        title={`Delete ${state.deleting ?? ""}?`}
        description="This removes the wiring and restarts the app."
        destructive
        confirmLabel="Delete"
        busy={state.busy}
        onConfirm={() => {
          if (state.deleting) state.deleteBinding(state.deleting);
          state.setDeleting(null);
        }}
        onCancel={() => state.setDeleting(null)}
      />
    </div>
  );
}

/**
 * One resource: the read view (target, honest readiness, the env key names the
 * app actually receives) and, while editing, the two-path switch plus Cancel
 * and Save. Nothing here is ai- or db-specific.
 */
function BindingCard<TDraft extends { name: string; provider: string }, TItem>({
  draftName,
  descriptor,
  state
}: {
  draftName: string;
  descriptor: BindingSectionDescriptor<TDraft, TItem>;
  state: UseBindingDraftResult<TDraft, TItem>;
}) {
  const draft = state.drafts[draftName];
  const editing = state.editingName === draftName;
  const original = state.originalFor(draftName);
  const source = state.effectiveSource(draftName);
  const { ref } = resolveKeyRef(source, descriptor.keyRef(draft));
  // A key-source change is also dirty when it resolves to a ref different from
  // the stored one.
  const dirty =
    descriptor.isDirty(draft, original) ||
    (source.kind !== "none" && ref !== descriptor.keyRef(draft));
  const validity = descriptor.validity(draft, source, original === undefined);
  const card = descriptor.describe(draft, state.items);

  return (
    <div data-testid={`binding-card-${card.name}`} className="space-y-3 rounded-card bg-surface-hover p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Mono className="text-foreground">{card.name}</Mono>
          <Mono className="text-muted-foreground">{card.provider}</Mono>
        </div>
        {state.canWrite && !editing && (
          <div className="flex items-center gap-1">
            <Button size="sm" variant="ghost" onClick={() => state.startEdit(draftName)}>
              <Pencil aria-hidden="true" className="h-4 w-4" /> Edit
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="text-destructive"
              onClick={() => state.setDeleting(draftName)}
            >
              <Trash2 aria-hidden="true" className="h-4 w-4" /> Delete
            </Button>
          </div>
        )}
      </div>

      {!editing ? (
        <div className="space-y-2">
          <Mono as="p" className="text-muted-foreground">
            {card.target ?? "Not set"}
          </Mono>
          <p className={`text-12 ${READINESS_CLASS[card.readiness.state] ?? READINESS_CLASS.unknown}`}>
            {card.readiness.detail}
          </p>
          <ul className="flex flex-wrap gap-2">
            {card.envKeys.map((envKey) => (
              <li key={envKey} className={ENV_KEY_CLASS}>
                <KeyRound aria-hidden="true" className="h-4 w-4 text-subtle-foreground" />
                {envKey}
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="space-y-3">
          <BindingSwitch
            descriptor={descriptor}
            draft={draft}
            bindingName={draftName}
            items={state.items}
            source={source}
            secretKeys={state.secretKeys}
            sharedSecretKeys={state.sharedSecretKeys}
            onKeySourceChange={(next) => state.setKeySource(draftName, next)}
            onChange={(patch) => {
              state.patchDraft(draftName, patch);
              // The key picker unmounts on the no-secret path; reset it so a
              // stale paste never leaks into a local or managed save.
              if (patch.provider === descriptor.paths[0].provider) state.resetKeySource(draftName);
            }}
          />
          <div className="flex items-center justify-between gap-3">
            <span className="text-12 text-subtle-foreground">
              {dirty && !validity.valid && validity.hint
                ? validity.hint
                : "Applying restarts the app to inject new env."}
            </span>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="ghost" onClick={() => state.cancelEdit(draftName)}>
                Cancel
              </Button>
              <Button
                size="sm"
                variant="primary"
                disabled={!dirty || !validity.valid || state.busy}
                onClick={() => state.applyBinding(draftName)}
              >
                Save
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
