import { useId, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiUrl } from "../api/client";
import { useAuthRole, useCreateToken, useRevokeToken, useTokens } from "../api/queries";
import {
  Badge,
  Button,
  Confirm,
  CopyField,
  Field,
  Input,
  Mono,
  Notice,
  PageHeader,
  Panel,
  Select,
  TableRow
} from "../components/ui";
import { clearStoredToken, getStoredToken, isDeadBearer } from "../lib/auth";
import { formatRelative } from "../lib/format";
import { toast } from "../state/toastStore";
import type { TokenRole, TokenView } from "../api/types";

// Scoped-token management (`/tokens` surface).
//
// D-P23-5 — the SPA never persists a minted token plaintext. The `POST /tokens`
// response is rendered once, in a dialog, from local component state; it is
// never written to a query cache under a reusable key, never to
// localStorage/sessionStorage, never into a URL, and never into an analytics
// payload. The dialog's close handler clears the local state AND calls
// `mutation.reset()`, which — paired with the hook's `gcTime: 0` — evicts the
// mutation from the `MutationCache` immediately. Neither half is optional.

const ROLE_OPTIONS: { value: TokenRole; label: string; hint: string }[] = [
  { value: "readonly", label: "readonly", hint: "Reads only: every mutation is refused." },
  { value: "submitter", label: "submitter", hint: "Deploy and operate its own apps." },
  { value: "admin", label: "admin", hint: "Full control, including tokens and config." }
];

type TokensAccess = "ok" | "forbidden";

/**
 * One-shot admin-access probe, mirroring `pages/Audit.tsx`. `GET /tokens` is
 * admin-only; an authenticated non-admin gets a 403 which we surface as *data*
 * ("forbidden"), not a thrown error — so react-query never retries and the page
 * renders a friendly empty state instead of an error-toast loop. A dead bearer
 * takes the other branch (see `isDeadBearer`): clear + back to login.
 */
async function probeTokensAccess(): Promise<TokensAccess> {
  const token = getStoredToken();
  const response = await fetch(apiUrl("/tokens"), {
    headers: token ? { Authorization: `Bearer ${token}` } : {}
  });
  if (response.ok) return "ok";

  // Parse the P1 envelope once: its `code` is the only thing that separates a
  // role refusal from a dead token, and it also carries the human message.
  const body = (await response.json().catch(() => null)) as {
    code?: unknown;
    message?: unknown;
    detail?: unknown;
  } | null;
  const code = typeof body?.code === "string" ? body.code : null;

  if (isDeadBearer(response.status, code)) {
    // Mirror api/client.ts: stale token, back to login.
    clearStoredToken();
    if (typeof window !== "undefined") window.location.assign("/login");
    return "forbidden";
  }
  if (response.status === 403) return "forbidden";

  const message =
    typeof body?.message === "string"
      ? body.message
      : typeof body?.detail === "string"
        ? body.detail
        : response.statusText;
  throw new Error(message);
}

const QUOTA_ERROR = "Enter a whole number ≥ 0, or leave blank for uncapped.";

/**
 * Optional quota field. `null` is STRICTLY the blank field's meaning
 * ("uncapped"); a non-empty value that is not a whole number ≥ 0 is a form
 * error, never a silent `null` — coercing a typo to `null` would mint an
 * uncapped token from a mistyped cap, which is the widening direction.
 */
type QuotaParse = { ok: true; value: number | null } | { ok: false };

function parseQuota(raw: string): QuotaParse {
  const trimmed = raw.trim();
  if (!trimmed) return { ok: true, value: null };
  // Digits only: rejects "abc", "-1", "1.5", "1e3", "12x" — all of which
  // `Number.parseInt` would otherwise swallow or coerce.
  if (!/^\d+$/.test(trimmed)) return { ok: false };
  const value = Number.parseInt(trimmed, 10);
  return Number.isSafeInteger(value) ? { ok: true, value } : { ok: false };
}

/** The inline message for a field, or `null` when the field is acceptable. */
function quotaError(raw: string): string | null {
  return parseQuota(raw).ok ? null : QUOTA_ERROR;
}

/**
 * Roles are not run states, so they never borrow the run-state colours: one
 * muted Badge, the role's own word.
 */
function RoleBadge({ role }: { role: TokenRole }) {
  return <Badge tone="muted">{role}</Badge>;
}

/** Quiet placeholder rows in the table's own shape (guidelines §7). */
function PlaceholderRows({ rows = 3, cells = 6 }: { rows?: number; cells?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, row) => (
        <tr key={row} className="border-b border-border">
          {Array.from({ length: cells }).map((__, cell) => (
            <td key={cell} className="px-4 py-4">
              <span className="block h-3 w-full max-w-[8rem] rounded-button bg-surface-hover" />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

/**
 * The plaintext-once dialog. `value === null` is the replay state: `token.create`
 * is in the daemon's no-body-cache set, so a replayed request answers with a
 * non-secret envelope carrying no `token` — a real, explainable outcome, not a
 * parse failure.
 */
function MintedTokenDialog({
  name,
  value,
  onClose
}: {
  name: string;
  value: string | null;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-overlay p-4 backdrop-blur">
      <div
        role="dialog"
        aria-modal="true"
        aria-label="New token"
        className="w-full max-w-lg rounded-panel border border-border bg-surface p-6 shadow-lg"
      >
        <h2 className="text-16 font-semibold text-foreground">
          Token <Mono as="span">{name}</Mono> created
        </h2>
        {value !== null ? (
          <>
            <Notice tone="warning" className="mt-4">
              This is the only time you will see this value. Copy it now: it is stored hashed and
              cannot be shown again.
            </Notice>
            <div data-testid="minted-token" className="mt-4">
              <CopyField value={value} className="w-full" />
            </div>
          </>
        ) : (
          <p className="mt-4 text-14 text-muted-foreground">
            This request replayed an earlier one, so the daemon did not re-issue the secret: the
            plaintext is shown only once, at creation. Create another token if you need a value you
            can copy.
          </p>
        )}
        <div className="mt-6 flex justify-end">
          <Button variant="primary" onClick={onClose}>
            Done
          </Button>
        </div>
      </div>
    </div>
  );
}

/** Create form + the plaintext-once dialog; both live here so the close handler owns `reset()`. */
function CreateTokenPanel() {
  const create = useCreateToken();
  const nameId = useId();
  const roleId = useId();
  const gpusId = useId();
  const jobsId = useId();
  const [name, setName] = useState("");
  const [role, setRole] = useState<TokenRole>("submitter");
  const [maxGpus, setMaxGpus] = useState("");
  const [maxJobs, setMaxJobs] = useState("");
  // The ONLY place a minted plaintext lives (D-P23-5). `token: null` = replay.
  const [minted, setMinted] = useState<{ name: string; token: string | null } | null>(null);

  // Derived, not state: an invalid quota is a property of what is in the field
  // right now, so it can never go stale against the value that would be sent.
  const maxGpusError = quotaError(maxGpus);
  const maxJobsError = quotaError(maxJobs);

  function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    const label = name.trim();
    if (!label) return;
    const gpus = parseQuota(maxGpus);
    const jobs = parseQuota(maxJobs);
    // The disabled submit button already blocks this; the guard keeps the
    // "never mint from an unparsed quota" rule true regardless of the button.
    if (!gpus.ok || !jobs.ok) return;
    create.mutate(
      {
        name: label,
        role,
        max_gpus: gpus.value,
        max_concurrent_jobs: jobs.value
      },
      {
        onSuccess: (data) => {
          const plaintext = typeof data.token === "string" && data.token.length > 0;
          setMinted({ name: label, token: plaintext ? data.token : null });
          setName("");
          setMaxGpus("");
          setMaxJobs("");
        }
      }
    );
  }

  /**
   * Both calls are load-bearing (D-P23-5): `setMinted(null)` drops the
   * plaintext from component state, `create.reset()` — against the hook's
   * `gcTime: 0` — evicts the mutation (and its `data`) from the MutationCache.
   */
  function closeMinted() {
    setMinted(null);
    create.reset();
  }

  return (
    <Panel title="New token">
      <form className="space-y-5 p-4" onSubmit={onSubmit}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Field label="Name" htmlFor={nameId}>
            <Input
              id={nameId}
              type="text"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="ci-runner"
            />
          </Field>
          <Field
            label="Role"
            htmlFor={roleId}
            hint={ROLE_OPTIONS.find((option) => option.value === role)?.hint}
          >
            <Select
              id={roleId}
              value={role}
              onChange={(event) => setRole(event.target.value as TokenRole)}
            >
              {ROLE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </Select>
          </Field>
          {/*
            Deliberately `type="text"` + `inputMode="numeric"`, not
            `type="number"`: a number input reports a non-numeric entry as an
            EMPTY value, which is exactly the blank field's "uncapped" meaning,
            so the browser would hand us the widening reading of a typo. Text
            keeps what was typed, so this component (not the browser) is the
            authority on what is a valid quota, and the error is rendered rather
            than silently coerced.
          */}
          <Field
            label="Max GPUs"
            htmlFor={gpusId}
            hint="Blank means uncapped."
            error={
              maxGpusError ? <span data-testid="max-gpus-error">{maxGpusError}</span> : undefined
            }
          >
            <Input
              id={gpusId}
              type="text"
              inputMode="numeric"
              value={maxGpus}
              aria-invalid={maxGpusError !== null}
              onChange={(event) => setMaxGpus(event.target.value)}
            />
          </Field>
          <Field
            label="Max concurrent jobs"
            htmlFor={jobsId}
            hint="Blank means uncapped."
            error={
              maxJobsError ? <span data-testid="max-jobs-error">{maxJobsError}</span> : undefined
            }
          >
            <Input
              id={jobsId}
              type="text"
              inputMode="numeric"
              value={maxJobs}
              aria-invalid={maxJobsError !== null}
              onChange={(event) => setMaxJobs(event.target.value)}
            />
          </Field>
        </div>
        <div className="flex items-center justify-end">
          <Button
            type="submit"
            variant="primary"
            loading={create.isPending}
            disabled={!name.trim() || maxGpusError !== null || maxJobsError !== null}
          >
            Create token
          </Button>
        </div>
      </form>

      {minted && (
        <MintedTokenDialog name={minted.name} value={minted.token} onClose={closeMinted} />
      )}
    </Panel>
  );
}

const COLUMNS = ["Name", "Role", "Quotas", "Created", "Last used", "Status"];

/** The token table + the revoke confirm; only mounted once the access probe passed. */
function TokensTable() {
  const [includeRevoked, setIncludeRevoked] = useState(false);
  const tokens = useTokens(includeRevoked);
  const revoke = useRevokeToken();
  const [pendingRevoke, setPendingRevoke] = useState<TokenView | null>(null);

  const items = tokens.data ?? [];
  const empty = !tokens.isLoading && !tokens.error && items.length === 0;

  return (
    <Panel
      title={`Tokens (${items.length})`}
      actions={
        <Button
          size="sm"
          variant="ghost"
          role="switch"
          aria-checked={includeRevoked}
          onClick={() => setIncludeRevoked((value) => !value)}
        >
          {includeRevoked ? "Showing revoked" : "Include revoked"}
        </Button>
      }
    >
      {tokens.error && (
        <div className="border-b border-border px-4 py-3 text-14 text-destructive">
          {(tokens.error as Error).message}
        </div>
      )}

      {empty ? (
        <div className="flex flex-col items-start gap-2 px-4 py-10">
          <p className="text-14 text-foreground">No tokens yet.</p>
          <p className="text-14 text-muted-foreground">
            Mint one here, or from a shell with the daemon&apos;s admin token:
          </p>
          <Mono className="text-muted-foreground">nerdit token create</Mono>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[46rem] text-14">
            <thead>
              <tr className="border-b border-border">
                {COLUMNS.map((column) => (
                  <th key={column} className="label px-4 py-2 text-left font-medium">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tokens.isLoading && <PlaceholderRows cells={COLUMNS.length} />}
              {items.map((token) => (
                <TableRow key={token.id}>
                  <td className="max-w-[16rem] px-4 py-3">
                    <span className="block truncate text-foreground" title={token.name}>
                      {token.name}
                    </span>
                    <Mono className="block truncate text-subtle-foreground">{token.id}</Mono>
                  </td>
                  <td className="px-4 py-3">
                    <RoleBadge role={token.role} />
                  </td>
                  <td className="px-4 py-3">
                    <Mono className="text-muted-foreground">
                      {token.max_gpus ?? "∞"} gpu · {token.max_concurrent_jobs ?? "∞"} jobs
                    </Mono>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {formatRelative(token.created_at)}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {token.last_used_at ? formatRelative(token.last_used_at) : "never"}
                  </td>
                  <td className="px-4 py-3">
                    {token.revoked ? (
                      <span className="text-muted-foreground">revoked</span>
                    ) : (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-destructive"
                        disabled={revoke.isPending}
                        onClick={() => setPendingRevoke(token)}
                      >
                        Revoke
                      </Button>
                    )}
                  </td>
                </TableRow>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Confirm
        open={pendingRevoke !== null}
        title="Revoke token"
        description={
          <>
            Every caller using <Mono as="span">{pendingRevoke?.name}</Mono> is refused immediately.
            Revocation cannot be undone.
          </>
        }
        destructive
        confirmLabel="Revoke"
        onCancel={() => setPendingRevoke(null)}
        onConfirm={() => {
          if (pendingRevoke) {
            const label = pendingRevoke.name;
            revoke.mutate(pendingRevoke.id, {
              onSuccess: () => toast("success", `Revoked ${label}`)
            });
          }
          setPendingRevoke(null);
        }}
      />
    </Panel>
  );
}

export default function Tokens() {
  const access = useQuery({
    // Deliberately NOT under the ["tokens"] key: the create/revoke mutations
    // invalidate that prefix, and the probe is a one-shot access check.
    queryKey: ["tokens-access"],
    queryFn: probeTokensAccess,
    retry: false,
    staleTime: 60_000
  });
  const auth = useAuthRole();
  // Cosmetic gate only: `require_role` on the daemon is authoritative.
  const isAdmin = auth.data?.role === "admin";
  const canCreate = access.data === "ok" && isAdmin;

  return (
    <div className="mx-auto max-w-content space-y-7">
      <PageHeader
        title="Tokens"
        subtitle="Scoped API tokens: role, quotas, last use. The secret is shown once, at creation, and stored only as a hash."
      />

      {access.isLoading && (
        <Panel title="Tokens">
          <table className="w-full text-14">
            <tbody>
              <PlaceholderRows cells={COLUMNS.length} />
            </tbody>
          </table>
        </Panel>
      )}

      {access.data === "forbidden" && (
        <Panel title="Tokens">
          <div className="flex flex-col items-start gap-2 px-4 py-10">
            <p className="text-14 text-foreground">Admin token required</p>
            <p className="text-14 text-muted-foreground">Sign in with an admin token.</p>
          </div>
        </Panel>
      )}

      {access.isError && (
        <Panel title="Tokens">
          <div className="flex flex-col items-start gap-3 px-4 py-10">
            <p className="text-14 text-foreground">Could not reach the token list.</p>
            <p className="text-14 text-muted-foreground">{(access.error as Error).message}</p>
            <Button onClick={() => void access.refetch()}>Retry</Button>
          </div>
        </Panel>
      )}

      {canCreate && <CreateTokenPanel />}
      {access.data === "ok" && <TokensTable />}
    </div>
  );
}
