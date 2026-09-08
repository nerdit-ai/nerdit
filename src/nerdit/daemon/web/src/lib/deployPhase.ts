/**
 * The deploy generation's vocabulary, in one place (design guidelines §6).
 *
 * `last_deploy.phase` is a different axis from the run state in `lib/status.ts`
 * — a running app can carry a failed deploy — so it gets its own words rather
 * than being folded into the run-state map. The old `DeployPhaseChip` (a pill
 * with a pulsing dot) is retired: a phase is a word beside a timestamp, and the
 * run-state Badge already carries the colour.
 *
 * `healthy` and anything unknown return null: a settled deploy is not a phase
 * worth announcing, and inventing a word for a value the daemon did not send
 * would be a lie.
 */
export const DEPLOY_PHASE_LABELS: Record<string, string> = {
  queued: "Queued",
  building: "Building",
  launching: "Launching",
  healthy: "Healthy",
  failed: "Failed"
};

/** The word an operator reads for an in-flight or failed deploy, else null. */
export function deployPhaseLabel(phase: string | null | undefined): string | null {
  if (!phase || phase === "healthy") return null;
  return DEPLOY_PHASE_LABELS[phase] ?? null;
}
