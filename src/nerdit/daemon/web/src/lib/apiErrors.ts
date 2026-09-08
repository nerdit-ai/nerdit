import { ApiError } from "../api/client";
import { toast } from "../state/toastStore";

/**
 * The one place a failed mutation becomes a sentence a human reads.
 *
 * The daemon answers every error with the P1 structured envelope
 * (`{code, message, hint?}`) and its `message` is already written for a human —
 * so it is the default. `CODE_COPY` overrides only the handful of codes whose
 * envelope text is terse, generic (a bare status-code retrofit like
 * `forbidden`) or written for an operator rather than a dashboard user. The
 * `hint` is the daemon's own next step ("run nerdit trust", "set
 * [git].enabled") and is appended as its own sentence, because dropping it —
 * as every call site did before this helper — throws away the most useful half
 * of the answer. The one exception: a code with a `CODE_COPY` entry keeps ONLY
 * that copy — such codes are overridden precisely because the daemon talks
 * operator there, and their hints do too (ETags, cutover timeouts).
 *
 * Guidelines §7: no raw error codes on screen, no apologies; failure copy
 * pairs what happened with the next step.
 */

/**
 * Friendlier copy for codes whose envelope message reads terse or jargon.
 * Deliberately small: an entry here is a claim that the daemon's own wording is
 * worse than this one, and every other code keeps the server's message (which
 * carries the app name, the limit, the reason).
 */
export const CODE_COPY: Record<string, string> = {
  // Status-code retrofits (`errors.py::_CODE_BY_STATUS`) — generic by design.
  unauthenticated: "You need to sign in again.",
  unauthorized: "You need to sign in again.",
  forbidden: "You don't have permission to do that.",
  payload_too_large: "That upload is larger than this daemon accepts.",
  internal_error: "The daemon hit an unexpected error.",
  // Expired scoped token (P25) — denied on every method, including reads.
  token_expired: "Your access token has expired. Sign in again.",
  // Optimistic-concurrency conflicts: the envelope talks about ETags.
  "config.stale": "Someone else changed this config. Reload and retry.",
  // In-flight conflicts: the envelope names daemon internals (cutover, drain).
  "service.cutover_in_progress": "A deploy is finishing for this app. Try again in a moment.",
  "service.run_in_progress": "A command is still running for this app. Try again in a moment.",
  "daemon.restart_in_progress": "The daemon is restarting. Try again in a moment.",
  "system.gc_in_progress": "A cleanup is already running. Try again when it finishes.",
  "workspace.deploy_in_progress": "This app's files are busy. Try again in a moment."
};

/** Fallback when neither a mapped code nor a message yields anything readable. */
const GENERIC = "The request did not go through.";

function isApiError(err: unknown): err is ApiError {
  if (err instanceof ApiError) return true;
  // Structural fallback: an ApiError that crossed a module/realm boundary still
  // carries the envelope fields, and losing the hint to a failed `instanceof`
  // would be exactly the bug this helper exists to fix.
  return (
    err instanceof Error &&
    err.name === "ApiError" &&
    typeof (err as { status?: unknown }).status === "number"
  );
}

/** True when `text` already ends in terminal punctuation. */
function isTerminated(text: string): boolean {
  return /[.!?]$/.test(text);
}

/** `base` + the daemon's hint as its own, properly punctuated sentence. */
function withHint(base: string, hint: string | undefined): string {
  const trimmedHint = hint?.trim();
  if (!trimmedHint) return base;
  const lead = isTerminated(base) ? base : `${base}.`;
  const tail = isTerminated(trimmedHint) ? trimmedHint : `${trimmedHint}.`;
  return `${lead} ${tail}`;
}

/**
 * Turn any thrown value into one line of user-facing copy.
 *
 * `context` is the verb of the thing that failed ("Deploy failed") and is
 * prefixed as `"<context>: <copy>"` — unless the copy already opens with it,
 * so a daemon message that names the action itself is never stuttered back.
 */
export function apiErrorCopy(err: unknown, context?: string): string {
  let copy: string;
  if (isApiError(err)) {
    const mapped = err.code ? CODE_COPY[err.code] : undefined;
    if (mapped) {
      // A CODE_COPY entry is written to be self-sufficient AND exists because
      // the daemon's wording talks operator (its hints for these codes name
      // ETags, cutover timeouts, config keys — vocabulary the guidelines ban
      // from the screen), so the verbatim hint is suppressed with the message.
      copy = mapped;
    } else {
      const base = (err.message ?? "").trim();
      copy = withHint(base || GENERIC, err.hint);
    }
  } else if (err instanceof Error && err.message.trim()) {
    copy = err.message.trim();
  } else {
    copy = String(err).trim() || GENERIC;
  }

  const prefix = context?.trim();
  if (!prefix) return copy;
  if (copy.toLowerCase().startsWith(prefix.toLowerCase())) return copy;
  return `${prefix}: ${copy}`;
}

/** Show `apiErrorCopy` as an error toast — the single mutation `onError` body. */
export function toastApiError(err: unknown, context?: string): void {
  toast("error", apiErrorCopy(err, context));
}
