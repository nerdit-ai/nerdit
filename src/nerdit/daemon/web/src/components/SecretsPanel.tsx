import { useId, useState } from "react";
import { KeyRound, ShieldAlert } from "lucide-react";
import { ApiError } from "../api/client";
import { Button, Confirm, Field, Input, Mono, Notice } from "./ui";
import { useDeleteSecret, useSecretNames, useSetSecrets } from "../api/queries";
import { toast } from "../state/toastStore";

export interface SecretsPanelProps {
  /** Service name — secrets are keyed by name, not job id. */
  service: string;
  /**
   * Pre-emptive role gate: false renders the panel read-only up front (used
   * by the shared-scope card, where writes are admin-only and the caller
   * knows the role from /auth/check). Defaults to true — the 403 fallback
   * below still catches ownership denials the caller cannot predict.
   */
  canWrite?: boolean;
}

/** True for a 403 from the API client — the role gate, not a transient failure. */
function isForbidden(error: unknown): boolean {
  return error instanceof ApiError && error.status === 403;
}

/**
 * Write-only secrets management for one service. The API only ever returns
 * key *names*; values are typed in, submitted, and immediately cleared — they
 * are never redisplayed anywhere in the dashboard.
 *
 * Role-aware two ways: callers that know the role up front (from /auth/check)
 * pass `canWrite={false}` and the write form never renders; on top of that,
 * the first 403 from a write flips the panel into a permanent read-only mode
 * for this mount (defense in depth for ownership denials the caller cannot
 * predict). The 403 branch is deliberately NOT delegated to `apiErrorCopy`:
 * it is role truth (the panel changes shape, permanently for this mount), not
 * error copy. The sentence is `toastApiError`'s, written once by the
 * mutations' own `onError` in `api/queries.ts` — these local handlers add the
 * role flip and nothing else, so a denial is never announced twice.
 */
export function SecretsPanel({ service, canWrite = true }: SecretsPanelProps) {
  const names = useSecretNames(service);
  const setSecrets = useSetSecrets(service);
  const deleteSecret = useDeleteSecret(service);
  const fieldId = useId();

  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [pendingDeleteAll, setPendingDeleteAll] = useState(false);
  const [forbidden, setForbidden] = useState(false);
  const readOnly = forbidden || !canWrite;

  const keys = names.data?.keys ?? [];
  const trimmedKey = newKey.trim();
  const canAdd = Boolean(trimmedKey && newValue) && !setSecrets.isPending && !readOnly;

  function onAdd() {
    if (!canAdd) return;
    const isUpdate = keys.includes(trimmedKey);
    setSecrets.mutate(
      { [trimmedKey]: newValue },
      {
        onSuccess: () => {
          toast("success", `Secret ${trimmedKey} ${isUpdate ? "updated" : "set"}`);
          // Write-only contract: clear the inputs, never redisplay the value.
          setNewKey("");
          setNewValue("");
        },
        // The sentence is already written by `useSetSecrets`' own
        // `toastApiError` onError; this handler only owns the role flip, and
        // toasting again here would say the same thing twice.
        onError: (error) => {
          if (isForbidden(error)) setForbidden(true);
        }
      }
    );
  }

  function onDelete(key: string | undefined) {
    deleteSecret.mutate(key, {
      onSuccess: () => toast("success", key ? `Secret ${key} deleted` : "All secrets deleted"),
      // Same split as `onAdd`: `useDeleteSecret` toasts the copy, this owns
      // the role flip.
      onError: (error) => {
        if (isForbidden(error)) setForbidden(true);
      }
    });
  }

  return (
    <div className="space-y-4">
      <p className="text-13 text-muted-foreground">
        Injected as env at launch. Write-only: key names are listed, values are never shown again.
      </p>

      {readOnly && (
        <Notice tone="warning" className="flex items-center gap-2 text-13">
          <ShieldAlert size={16} aria-hidden="true" className="shrink-0" />
          Admin token required to modify these secrets. Showing key names only.
        </Notice>
      )}

      {/* Quiet placeholder in the list's own shape — never the word "Loading". */}
      {names.isLoading && (
        <div className="py-2" data-testid="secrets-placeholder">
          <span className="block h-3 w-40 rounded-full bg-surface-hover" />
        </div>
      )}
      {names.error && (
        <Notice tone="destructive" className="text-13">
          {(names.error as Error).message}
        </Notice>
      )}
      {!names.isLoading && !names.error && keys.length === 0 && (
        <p className="text-13 text-muted-foreground">
          No secrets set. Add a key below, or from a shell:{" "}
          <Mono>nerdit secrets set {service} KEY=value</Mono>
        </p>
      )}

      {keys.length > 0 && (
        <ul className="divide-y divide-border border-y border-border">
          {keys.map((key) => (
            <li
              key={key}
              data-testid={`secret-row-${key}`}
              className="flex items-center justify-between gap-3 py-2"
            >
              <Mono className="flex min-w-0 items-center gap-2 text-foreground">
                <KeyRound size={16} className="shrink-0 text-muted-foreground" aria-hidden="true" />
                <span className="truncate">{key}</span>
                <span className="text-subtle-foreground">= ••••••</span>
              </Mono>
              {!readOnly && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setPendingDelete(key)}
                  disabled={deleteSecret.isPending}
                  className="shrink-0 text-destructive"
                >
                  Delete
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}

      {!readOnly && (
        <div className="grid grid-cols-[1fr_1fr_auto] items-end gap-2">
          <Field label="Key" htmlFor={`${fieldId}-key`}>
            <Input
              id={`${fieldId}-key`}
              mono
              value={newKey}
              onChange={(event) => setNewKey(event.target.value)}
              placeholder="KEY"
              autoComplete="off"
              spellCheck={false}
            />
          </Field>
          <Field label="Value" htmlFor={`${fieldId}-value`}>
            <Input
              id={`${fieldId}-value`}
              value={newValue}
              onChange={(event) => setNewValue(event.target.value)}
              placeholder="value"
              type="password"
              autoComplete="new-password"
            />
          </Field>
          <Button
            variant="primary"
            onClick={onAdd}
            disabled={!canAdd}
            loading={setSecrets.isPending}
          >
            {keys.includes(trimmedKey) ? "Update" : "Add"}
          </Button>
        </div>
      )}

      {!readOnly && keys.length > 0 && (
        <div className="flex justify-end">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPendingDeleteAll(true)}
            disabled={deleteSecret.isPending}
            className="text-destructive"
          >
            Delete all secrets
          </Button>
        </div>
      )}

      <Confirm
        open={pendingDelete !== null}
        title="Delete secret?"
        description={
          <>
            Delete <Mono className="text-foreground">{pendingDelete}</Mono> from{" "}
            <Mono className="text-foreground">{service}</Mono>? The running container keeps its env
            until the next restart.
          </>
        }
        destructive
        confirmLabel="Delete"
        busy={deleteSecret.isPending}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          if (pendingDelete) onDelete(pendingDelete);
          setPendingDelete(null);
        }}
      />

      <Confirm
        open={pendingDeleteAll}
        title="Delete all secrets?"
        description={
          <>
            Delete all {keys.length} secret{keys.length === 1 ? "" : "s"} of{" "}
            <Mono className="text-foreground">{service}</Mono>? This cannot be undone. The values
            are not recoverable.
          </>
        }
        destructive
        confirmLabel="Delete all"
        busy={deleteSecret.isPending}
        onCancel={() => setPendingDeleteAll(false)}
        onConfirm={() => {
          onDelete(undefined);
          setPendingDeleteAll(false);
        }}
      />
    </div>
  );
}

