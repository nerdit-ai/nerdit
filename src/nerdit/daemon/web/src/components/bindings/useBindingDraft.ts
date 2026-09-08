import { useEffect, useRef, useState } from "react";
import { ApiError } from "../../api/client";
import { useAppConfig, useAuthRole, useSecretNames, useSetSecrets, useUpdateAppConfig } from "../../api/queries";
import { apiErrorCopy } from "../../lib/apiErrors";
import { resolveKeyRef, type KeySource } from "../../lib/bindings";
import { toast } from "../../state/toastStore";
import type { BindingSectionDescriptor } from "./sections";

export interface Banner {
  tone: "warning" | "destructive";
  text: string;
}

/** The hook's own return type, so the contract is declared exactly once. */
export type UseBindingDraftResult<TDraft extends { name: string; provider: string }, TItem> =
  ReturnType<typeof useBindingDraft<TDraft, TItem>>;

/**
 * The draft / dirty / save state machine behind both the `[ai.*]` and `[db.*]`
 * panels. Everything that differs between the two is looked up on `descriptor`.
 *
 * Saving is one round trip: the pasted secret first (once, write-only), then a
 * per-binding partial `PUT /config/apps/{name}/{section}` carrying only the
 * `${secrets.*}` reference, guarded by the config ETag so a concurrent edit is
 * caught rather than clobbered. There is no preview step: a save either lands
 * or explains itself.
 */
export function useBindingDraft<TDraft extends { name: string; provider: string }, TItem>(
  name: string,
  descriptor: BindingSectionDescriptor<TDraft, TItem>
) {
  const config = useAppConfig(name);
  const update = useUpdateAppConfig(name);
  const setSecrets = useSetSecrets(name);
  const { items } = descriptor.useItems();
  const secretNames = useSecretNames(name);
  const sharedSecretNames = useSecretNames("shared");
  const authRole = useAuthRole();

  const [drafts, setDrafts] = useState<Record<string, TDraft>>({});
  // Per-binding key/password source. Missing entry => { kind: "none" }, i.e.
  // keep the stored ref. A raw pasted value lives only here.
  const [keySources, setKeySources] = useState<Record<string, KeySource>>({});
  const [editingName, setEditingName] = useState<string | null>(null);
  const [banner, setBanner] = useState<Banner | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [addingNew, setAddingNew] = useState(false);
  const [newName, setNewName] = useState("");

  const view = config.data?.view;
  const etag = config.data?.etag ?? null;
  const sharedSecretKeys = sharedSecretNames.data?.keys ?? [];

  // Re-seed drafts only when the section itself changes (first arrival, or a
  // post-save refetch), so a background poll never clobbers unsaved edits.
  const seededRef = useRef<string | null>(null);
  useEffect(() => {
    if (!view) return;
    const section = descriptor.getSection(view);
    const sig = JSON.stringify(section);
    if (seededRef.current === sig) return;
    seededRef.current = sig;
    const out: Record<string, TDraft> = {};
    for (const [n, binding] of Object.entries(section)) out[n] = descriptor.draftFromView(n, binding);
    setDrafts(out);
    setKeySources({});
    setEditingName(null);
    setAddingNew(false);
    setNewName("");
  }, [view, descriptor]);

  function handleWriteError(err: Error) {
    const stale = err instanceof ApiError && err.code === "config.stale";
    // Someone else changed this app's config: pull the stored one so the
    // operator reviews against what is actually there, and say so.
    if (stale) void config.refetch();
    setBanner({ tone: stale ? "warning" : "destructive", text: apiErrorCopy(err) });
  }

  const setKeySource = (target: string, source: KeySource) =>
    setKeySources((prev) => ({ ...prev, [target]: source }));
  const resetKeySource = (target: string) => setKeySource(target, { kind: "none" });
  const effectiveSource = (target: string): KeySource =>
    descriptor.resolveSource(drafts[target], keySources[target] ?? { kind: "none" }, sharedSecretKeys);

  async function applyBinding(target: string) {
    setBanner(null);
    const draft = drafts[target];
    const { ref, secretWrite } = resolveKeyRef(effectiveSource(target), descriptor.keyRef(draft));
    const spec = descriptor.spec(descriptor.withKeyRef(draft, ref));

    // Step 1: write the pasted secret once. If it fails nothing else runs, so
    // the section is left untouched.
    if (secretWrite) {
      try {
        await setSecrets.mutateAsync({ [secretWrite.name]: secretWrite.value });
      } catch (err) {
        setBanner({
          tone: "destructive",
          text: apiErrorCopy(err, "Nothing was changed. The secret could not be saved")
        });
        return;
      }
      // Persisted: drop the pasted value immediately (write-only contract) and
      // reference it, so a retry reuses the stored secret with no re-post.
      setKeySource(target, { kind: "existing", key: secretWrite.name });
    }

    // Step 2: the section PUT carries only the reference.
    update.mutate(
      { section: descriptor.section, values: { [target]: spec }, etag, restart: true },
      {
        onSuccess: () => {
          toast("success", `Resource ${target} saved, app restarting`);
          resetKeySource(target);
          setEditingName(null);
        },
        onError: (err) => {
          if (!secretWrite) return handleWriteError(err);
          // Secret saved, section write failed. The key source already points
          // at the stored secret, so Save again finishes without re-posting.
          if (err instanceof ApiError && err.code === "config.stale") void config.refetch();
          setBanner({
            tone: "warning",
            text: `Secret ${secretWrite.name} was saved but the app config was not updated. The secret is kept. Save again to finish.`
          });
        }
      }
    );
  }

  function deleteBinding(target: string) {
    setBanner(null);
    update.mutate(
      { section: descriptor.section, values: { [target]: null }, etag, restart: true },
      {
        onSuccess: () => {
          toast("success", `Binding ${target} deleted, app restarting`);
          if (editingName === target) setEditingName(null);
        },
        onError: handleWriteError
      }
    );
  }

  function cancelEdit(target: string) {
    setBanner(null);
    setEditingName(null);
    resetKeySource(target);
    const section = view ? descriptor.getSection(view) : undefined;
    if (section && target in section) {
      setDrafts((prev) => ({ ...prev, [target]: descriptor.draftFromView(target, section[target]) }));
      return;
    }
    // A never-saved new binding: drop it entirely.
    setDrafts((prev) => {
      const next = { ...prev };
      delete next[target];
      return next;
    });
  }

  function confirmAddBinding() {
    const key = newName.trim();
    if (!key) return;
    // An existing draft for this name is kept intact, never clobbered: just
    // re-focus it. Own-property check, not truthiness, so a name like
    // "toString" still seeds a draft.
    setDrafts((prev) =>
      Object.prototype.hasOwnProperty.call(prev, key)
        ? prev
        : { ...prev, [key]: descriptor.newDraft(key) }
    );
    setEditingName(key);
    setAddingNew(false);
    setNewName("");
  }

  return {
    view, items, drafts, banner, deleting, addingNew, newName, editingName,
    setDeleting, setAddingNew, setNewName,
    configLoading: config.isLoading,
    configError: config.error as Error | null,
    canWrite: authRole.data?.role !== "readonly",
    draftNames: Object.keys(drafts),
    secretKeys: secretNames.data?.keys ?? [],
    sharedSecretKeys,
    busy: update.isPending || setSecrets.isPending,
    originalFor: (target: string) => (view ? descriptor.getSection(view)[target] : undefined),
    effectiveSource, resetKeySource, setKeySource, deleteBinding, cancelEdit, confirmAddBinding,
    patchDraft: (target: string, patch: Partial<TDraft>) =>
      setDrafts((prev) => ({ ...prev, [target]: { ...prev[target], ...patch } })),
    applyBinding: (target: string) => void applyBinding(target),
    startEdit: (target: string) => {
      setBanner(null);
      setEditingName(target);
    }
  };
}
