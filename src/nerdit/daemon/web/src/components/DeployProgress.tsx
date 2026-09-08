import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { useServiceWait } from "../api/queries";
import type { Service } from "../api/types";
import { DEPLOY_PHASE_LABELS } from "../lib/deployPhase";
import { Button, Mono } from "./ui";

/** The walk itself; the words come from `lib/deployPhase` (one vocabulary). */
const PHASE_STEPS = (["queued", "building", "launching", "healthy"] as const).map((key) => ({
  key,
  label: DEPLOY_PHASE_LABELS[key]
}));

function phaseIndex(phase: string | null | undefined): number {
  if (!phase) return 0;
  const idx = PHASE_STEPS.findIndex((s) => s.key === phase);
  return idx >= 0 ? idx : 0;
}

function isKnownPhase(phase: string | null | undefined): boolean {
  return !!phase && PHASE_STEPS.some((s) => s.key === phase);
}

/**
 * Post-deploy progress view — the PhaseWalk of design guidelines §5, and the
 * dashboard's own primitive rather than a `ui.tsx` one because nothing else on
 * the surface walks a generation.
 *
 * It long-polls `GET /services/{id}/wait` pinned to the deploy's
 * `last_deploy.version` and renders queued → building → launching → healthy.
 * Terminal outcomes keep their honest copy: a `failed` service whose old image
 * still serves is not dead; a `timeout` with `waited_s=0` is daemon saturation,
 * not failure.
 *
 * The walk is deliberately still. The current step is a filled dot beside a
 * medium-weight label; it does not blink. A pulsing dot said only "this page is
 * animating", which is not a fact about the deploy.
 */
export function DeployProgress({ service, onClose }: { service: Service; onClose: () => void }) {
  const version = service.last_deploy?.version ?? service.build_version ?? undefined;
  const wait = useServiceWait(service.name, version);
  const data = wait.data;

  const outcome = data?.outcome;
  const phase = data?.phase ?? service.last_deploy?.phase ?? "queued";
  const failed = outcome === "failed";
  const converged = outcome === "converged";
  const superseded = outcome === "superseded";
  const saturated = outcome === "timeout" && data?.waited_s === 0;

  // The wait poll walks queued → building → launching → healthy, then reports
  // `phase: "failed"` on failure — which is NOT one of PHASE_STEPS, so it would
  // fall back to Queued (index 0) and mark the failure on the wrong step. Track
  // the last non-terminal phase we actually observed while polling so the
  // failure marks the step the deploy really reached.
  const lastKnownPhase = useRef<string>(isKnownPhase(phase) ? phase : "queued");
  useEffect(() => {
    if (isKnownPhase(phase)) lastKnownPhase.current = phase;
  }, [phase]);

  const activeIdx = converged
    ? PHASE_STEPS.length - 1
    : phaseIndex(failed ? lastKnownPhase.current : phase);
  const publicUrl = data?.public_url ?? service.endpoint?.public_url ?? null;
  const stillServing = failed && data?.status === "running";
  // `reason` can be a machine token (`build_failed`) or a full sentence
  // (`error_message`). Tokens never reach the screen raw (guidelines §7):
  // a snake_case-only value is humanized; real sentences pass through.
  const rawReason = data?.reason ?? data?.error_message ?? service.last_deploy?.reason ?? null;
  const reason =
    rawReason && /^[a-z0-9_.]+$/.test(rawReason)
      ? rawReason.replace(/[_.]/g, " ").replace(/^./, (c) => c.toUpperCase())
      : rawReason;

  return (
    <div>
      <h3 className="text-16 font-semibold text-foreground">Deploying {service.name}</h3>
      <p className="mt-1 text-14 text-muted-foreground">
        Building and launching the app. This can take a moment.
      </p>

      <ol className="mt-4 space-y-2" data-testid="deploy-phases">
        {PHASE_STEPS.map((step, i) => {
          const done = i < activeIdx || converged;
          const current = i === activeIdx && !converged;
          const isFail = failed && current;

          // Done and current are both filled; only the pending steps are
          // hollow. Nothing here moves.
          let dotCls = "border border-border";
          if (isFail) dotCls = "bg-destructive";
          else if (done || current) dotCls = "bg-primary";

          let textCls = "text-subtle-foreground";
          if (isFail) textCls = "text-destructive font-medium";
          else if (current) textCls = "text-primary font-medium";
          else if (done) textCls = "text-foreground";

          return (
            <li
              key={step.key}
              data-testid={`deploy-phase-${step.key}`}
              data-state={isFail ? "failed" : done ? "done" : current ? "current" : "pending"}
              className="flex items-center gap-2 text-14"
            >
              <span aria-hidden="true" className={`h-2 w-2 shrink-0 rounded-full ${dotCls}`} />
              <span className={textCls}>{step.label}</span>
            </li>
          );
        })}
      </ol>

      {converged && (
        <div className="mt-4 rounded-card border border-border bg-background px-4 py-3 text-14">
          <p className="text-foreground">App is live.</p>
          {publicUrl && (
            <a
              href={publicUrl}
              target="_blank"
              rel="noreferrer"
              className="mt-1 block truncate text-primary hover:underline"
            >
              <Mono>{publicUrl}</Mono>
            </a>
          )}
        </div>
      )}

      {failed && (
        <div className="mt-4 rounded-card border border-destructive-border bg-destructive-subtle px-4 py-3 text-14">
          <p className="font-medium text-destructive">Deploy failed.</p>
          {reason && <p className="mt-1 text-muted-foreground">{reason}</p>}
          {stillServing && (
            <p className="mt-1 text-muted-foreground">Previous version still serving.</p>
          )}
          <Link
            to={`/projects/${encodeURIComponent(service.name)}`}
            onClick={onClose}
            className="mt-2 inline-block text-13 font-medium text-primary hover:underline"
          >
            Open diagnose
          </Link>
        </div>
      )}

      {superseded && (
        <p className="mt-4 rounded-card border border-border bg-background px-4 py-3 text-14 text-muted-foreground">
          Replaced by a newer deploy.
        </p>
      )}

      {saturated && !failed && (
        <p className="mt-4 rounded-card border border-border bg-background px-4 py-3 text-14 text-muted-foreground">
          Daemon busy, check the app page.
        </p>
      )}

      <div className="mt-6 flex justify-end gap-2">
        <Link
          to={`/projects/${encodeURIComponent(service.name)}`}
          onClick={onClose}
          className="inline-flex h-9 items-center justify-center rounded-button border border-border bg-surface px-4 text-14 font-medium text-foreground hover:bg-surface-hover"
        >
          View app
        </Link>
        <Button variant="primary" data-testid="deploy-done" onClick={onClose}>
          Done
        </Button>
      </div>
    </div>
  );
}
