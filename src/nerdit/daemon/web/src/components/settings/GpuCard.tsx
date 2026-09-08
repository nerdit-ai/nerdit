import { Badge, Meter, Panel } from "../ui";
import { useClusterStats, useGpus } from "../../api/queries";

// The GPU readout, folded out of the retired Hardware page (design guidelines
// §2/§6). What survived the fold is what an operator acts on: which cards the
// daemon sees, how much of their memory is taken, how busy they are.
// Temperature, vendor badges and the AMD inventory hint did NOT: they were the
// GPU-lab framing, and the product is a local AI PaaS.

/** MB → GB with one decimal. `null` is unknown, never zero. */
function gb(mb: number | null | undefined): string {
  if (mb == null || !Number.isFinite(mb)) return "unknown";
  return `${(mb / 1024).toFixed(1)} GB`;
}

export function GpuCard() {
  const gpus = useGpus();
  const stats = useClusterStats();

  const rows = gpus.data ?? [];
  const inUse = stats.data ? `${stats.data.gpus_in_use} / ${stats.data.gpus_total} in use` : null;

  return (
    <Panel
      title="GPUs"
      actions={inUse ? <span className="text-12 text-muted-foreground">{inUse}</span> : null}
    >
      {gpus.isError && (
        <p className="px-4 py-3 text-14 text-destructive">Could not read GPUs from the daemon.</p>
      )}

      {!gpus.data && !gpus.isError && (
        // Quiet placeholder in the layout's own shape, never a spinner.
        <div className="px-4 py-3">
          <div className="h-4 w-40 rounded-button bg-surface-hover" aria-hidden="true" />
        </div>
      )}

      {gpus.data && rows.length === 0 && (
        <p data-testid="gpu-empty" className="px-4 py-3 text-14 text-muted-foreground">
          No GPUs detected. Apps run on CPU.
        </p>
      )}

      {rows.length > 0 && (
        <ul className="divide-y divide-border">
          {rows.map((gpu, index) => (
            <li key={gpu.id} data-testid="gpu-row" className="px-4 py-3">
              <div className="flex items-baseline justify-between gap-4">
                <span className="min-w-0 truncate text-14 text-foreground">
                  {gpu.name} <span className="text-muted-foreground">#{index}</span>{" "}
                  {!gpu.schedulable && (
                    <Badge tone="muted" data-testid="gpu-inventory-only">
                      Inventory only
                    </Badge>
                  )}
                </span>
                <span className="shrink-0 text-13 text-muted-foreground">
                  {gpu.utilization_percent === null
                    ? "utilization unknown"
                    : `${gpu.utilization_percent}% busy`}
                </span>
              </div>
              <Meter
                className="mt-2"
                value={gpu.memory_used_mb ?? 0}
                max={gpu.memory_mb}
                hint={`${gb(gpu.memory_used_mb)} / ${gb(gpu.memory_mb)}`}
              />
              {/* An unschedulable GPU (an AMD card with [monitor].enable_amd
                  off) is visible but never allocated — showing healthy
                  telemetry without saying so sends operators on a failed
                  deploy (Codex review, PR #147; docs/guide/troubleshooting
                  points here). */}
              {!gpu.schedulable && (
                <p className="mt-1 text-12 text-muted-foreground">
                  Visible but not allocatable. Enable scheduling with{" "}
                  <span className="font-mono text-12">[monitor].enable_amd</span> in the daemon
                  config, then restart.
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
