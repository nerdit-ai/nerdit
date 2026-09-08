import { useState } from "react";
import { Button, Confirm, DataRow, Meter, Mono, Notice, Panel } from "../ui";
import { useAuthRole, useSystemDisk, useSystemGc } from "../../api/queries";
import type { GcResult, SystemDiskReport } from "../../api/types";

// Disk accounting + garbage collection (P14b `/system/disk` + `/system/gc`).
//
// Two halves with very different postures:
//   * the report is a read open to any authenticated principal;
//   * gc is admin-only, destructive and therefore PLAN-FIRST (R4) — it copies
//     the `nerdit gc` contract exactly: always show the dry-run plan, then ask
//     for an explicit confirmation before the real run. The confirm handler
//     hardcodes `dryRun: false` and drives its own mutation instance, so no
//     shared handler can fire a real run from the preview branch. The plan you
//     SEE is the run you confirm: changing the orphan-data scope drops the
//     rendered plan (`onScopeChange`), and the real run takes its scope from
//     the previewed mutation's variables rather than live state.
//
// Image reclaim is scoped to this daemon's `[daemon].instance_id` server-side
// (Track 0.4), so no multi-daemon caveat is needed in this copy.

/**
 * Human byte count. The unit ladder is identical to the CLI's `fmt_bytes`
 * (`cli/commands/system.py`) so the two surfaces never disagree on a size;
 * only the null rendering differs — a `null` here is a timed-out bounded walk,
 * and the one thing it must never look like is `0`.
 */
function formatBytes(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "unknown";
  let num = value;
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  for (const unit of units) {
    if (Math.abs(num) < 1024 || unit === "TiB") {
      return unit === "B" ? `${Math.trunc(num)} B` : `${num.toFixed(1)} ${unit}`;
    }
    num /= 1024;
  }
  return `${num.toFixed(1)} TiB`;
}

function NameList({ names }: { names: string[] }) {
  if (names.length === 0) return <span className="text-subtle-foreground">none</span>;
  return <Mono className="text-foreground">{names.join(", ")}</Mono>;
}

/**
 * The docker aggregate — `null` is "unavailable", never a row of zeros.
 *
 * The four buckets get Meters against docker's own total (the sum of the
 * buckets, which is the only denominator that exists here): the bar answers
 * "what is taking the space", the hint keeps the exact byte count.
 */
function DockerBlock({ docker }: { docker: SystemDiskReport["docker"] }) {
  if (docker === null) {
    return (
      <div className="px-4 py-3">
        <Notice tone="warning" data-testid="disk-docker-unavailable">
          docker unavailable, so sizes could not be read.
        </Notice>
      </div>
    );
  }

  const buckets: [string, number | null][] = [
    ["images", docker.images_bytes],
    ["containers", docker.containers_bytes],
    ["volumes", docker.volumes_bytes],
    ["build cache", docker.build_cache_bytes]
  ];
  const total = buckets.reduce((sum, [, value]) => sum + (value ?? 0), 0);

  return (
    <div className="space-y-4 px-4 py-4">
      {buckets.map(([label, value]) => (
        <Meter
          key={label}
          label={label}
          value={value ?? 0}
          max={total}
          hint={formatBytes(value)}
        />
      ))}
    </div>
  );
}

/**
 * The gc plan / result, mirroring the CLI's `_render_gc`: what would be (or was)
 * removed, what was skipped and why, the reclaim estimate, and the orphan-data
 * half only when it was opted into.
 */
function GcReport({ data }: { data: GcResult }) {
  const verb = data.dry_run ? "would remove" : "removed";
  return (
    <div
      data-testid="gc-report"
      className="mt-4 space-y-2 rounded-card border border-border bg-background p-4 text-14"
    >
      {/* The response's `dry_run` decides the wording; the mechanism's own name
          stays off screen (design guidelines §3). */}
      <p className="font-medium text-foreground">{data.dry_run ? "Preview" : "Result"}</p>
      <p className="text-muted-foreground">
        images {verb}: <NameList names={data.images.removed} />
      </p>
      {data.images.skipped.map((entry) => (
        <p key={entry.repo} className="text-warning-foreground">
          image skipped {entry.repo} ({entry.reason})
        </p>
      ))}
      <p className="text-muted-foreground">
        estimated reclaim:{" "}
        <Mono className="text-foreground">
          {formatBytes(data.images.reclaimed_bytes_estimate)}
        </Mono>
      </p>
      {data.orphan_data.enabled && (
        <>
          <p className="text-muted-foreground">
            data dirs {verb}: <NameList names={data.orphan_data.removed} />
          </p>
          {data.orphan_data.skipped.map((entry) => (
            <p key={entry.name} className="text-warning-foreground">
              data dir skipped {entry.name} ({entry.reason})
            </p>
          ))}
        </>
      )}
    </div>
  );
}

/**
 * Disk usage + the admin garbage-collection flow. `images.by_repo` and
 * `data_dir.services_total_bytes` are deliberately NOT rendered: per-repo
 * attribution is a `nerdit disk` concern (a Settings card listing every app
 * image repo is a wall, and the orphan list already names the actionable ones),
 * and the per-service list already states the same fact a total would restate.
 */
export function DiskCard() {
  const disk = useSystemDisk();
  const auth = useAuthRole();
  // Cosmetic gate only — `require_role(admin)` on the daemon is authoritative.
  const isAdmin = auth.data?.role === "admin";

  // Two mutation instances so each handler has exactly one thing it can do.
  const previewGc = useSystemGc();
  const runGc = useSystemGc();
  const [includeOrphanData, setIncludeOrphanData] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  function onPreview() {
    runGc.reset();
    previewGc.mutate({ includeOrphanData, dryRun: true });
  }

  /**
   * R4's contract is "the plan you see is the run you confirm", so a scope
   * change INVALIDATES the rendered plan: the report disappears and the
   * real-run button re-disables until a fresh dry run with the new scope has
   * rendered. Without this, widening the scope after a preview let an admin
   * confirm a plan that never showed a single data dir — the widening
   * direction, and the unrecoverable one. `runGc` is reset too so a previous
   * Result cannot read as the outcome of the new scope, and the dialog is
   * closed so it can never outlive the plan it describes.
   *
   * It is a NO-OP while the real run is in flight, and the control that calls
   * it is disabled for the same window: resetting `runGc` mid-run would detach
   * the UI from a destructive operation that is still deleting server-side —
   * its "Collecting…" state, its result and its warnings would all vanish while
   * the controls re-armed for a fresh preview. The pending run always gets to
   * render its own outcome; the scope is re-openable the moment it settles.
   */
  function onScopeChange(next: boolean) {
    if (runGc.isPending) return;
    setIncludeOrphanData(next);
    setConfirmOpen(false);
    previewGc.reset();
    runGc.reset();
  }

  // The scope the RENDERED plan was produced with — read from the previewed
  // mutation's own variables, never from live toggle state. Belt and braces:
  // this and `onScopeChange` independently guarantee that what is confirmed is
  // what was planned.
  const plannedScope = previewGc.data ? previewGc.variables?.includeOrphanData === true : false;

  /** Real run ONLY: `dryRun` is hardcoded false and this is the sole caller. */
  function onConfirmRun() {
    setConfirmOpen(false);
    if (!previewGc.data) return; // no plan ⇒ nothing to confirm
    runGc.mutate(
      { includeOrphanData: plannedScope, dryRun: false },
      { onSuccess: () => previewGc.reset() }
    );
  }

  const report = disk.data;

  return (
    <Panel
      title="Disk"
      data-testid="disk-card"
      actions={
        <Button size="sm" onClick={() => void disk.refetch()} disabled={disk.isFetching}>
          {disk.isFetching ? "Refreshing…" : "Refresh"}
        </Button>
      }
    >
      {disk.isLoading && (
        <div className="space-y-2 px-4 py-3" aria-hidden="true">
          <div className="h-4 w-64 rounded-button bg-surface-hover" />
          <div className="h-4 w-40 rounded-button bg-surface-hover" />
        </div>
      )}
      {disk.isError && (
        <p className="px-4 py-3 text-14 text-destructive">
          Could not load disk usage. {(disk.error as Error).message}
        </p>
      )}

      {report && (
        <div className="divide-y divide-border">
          <div>
            <p className="label px-4 pt-3">Docker</p>
            <DockerBlock docker={report.docker} />
          </div>

          <div>
            <p className="label px-4 pt-3">Data directory</p>
            <div className="divide-y divide-border">
              {report.data_dir.services.map((svc) => (
                <DataRow key={svc.name} label={`services/${svc.name}`}>
                  <Mono>{formatBytes(svc.bytes)}</Mono>
                </DataRow>
              ))}
              <DataRow label="models/ollama">
                <Mono>{formatBytes(report.data_dir.models.ollama)}</Mono>
              </DataRow>
              <DataRow label="models/huggingface">
                <Mono>{formatBytes(report.data_dir.models.huggingface)}</Mono>
              </DataRow>
              <DataRow label="archive">
                <Mono>{formatBytes(report.data_dir.archive_bytes)}</Mono>
              </DataRow>
              <DataRow label="backups">
                <Mono>
                  {`${formatBytes(report.data_dir.backups.bytes)} (${report.data_dir.backups.count} files)`}
                </Mono>
              </DataRow>
              <DataRow label="volume backups">
                <Mono>
                  {`${formatBytes(report.data_dir.volume_backups.bytes)} (${report.data_dir.volume_backups.count} files)`}
                </Mono>
              </DataRow>
              <DataRow label="dumps">
                <Mono>
                  {`${formatBytes(report.data_dir.dumps.bytes)} (${report.data_dir.dumps.count} files)`}
                </Mono>
              </DataRow>
              <DataRow label="dump staging">
                <Mono>{formatBytes(report.data_dir.dump_staging_bytes)}</Mono>
              </DataRow>
            </div>
          </div>

          <div className="divide-y divide-border">
            <DataRow label="Orphan images">
              <NameList names={report.orphan_images} />
            </DataRow>
            <DataRow label="Orphan data dirs">
              <NameList names={report.orphan_data_dirs} />
            </DataRow>
          </div>

          {report.warnings.length > 0 && (
            <div className="space-y-2 px-4 py-3">
              {report.warnings.map((warning) => (
                <Notice key={warning} tone="warning" data-testid="disk-warning">
                  warning: {warning}
                </Notice>
              ))}
            </div>
          )}
        </div>
      )}

      {isAdmin && (
        <div className="border-t border-border px-4 py-4">
          <p className="text-14 text-muted-foreground">
            Garbage collection removes orphan <Mono>nerdit-app/*</Mono> images this daemon built
            and no longer references. The plan is always previewed first.
          </p>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <p className={`text-14 text-muted-foreground ${runGc.isPending ? "opacity-50" : ""}`}>
              Also delete app data dirs with no live row. This cannot be undone.
            </p>
            <Button
              size="sm"
              data-testid="gc-orphan-data"
              aria-pressed={includeOrphanData}
              variant={includeOrphanData ? "secondary" : "ghost"}
              disabled={runGc.isPending}
              onClick={() => onScopeChange(!includeOrphanData)}
            >
              {includeOrphanData ? "Included" : "Include"}
            </Button>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <Button
              data-testid="gc-dry-run"
              onClick={onPreview}
              // Also disabled while the REAL run is in flight: a fresh preview
              // would replace the plan the running deletion was confirmed from.
              disabled={previewGc.isPending || runGc.isPending}
            >
              {previewGc.isPending ? "Planning…" : "Preview"}
            </Button>
            <Button
              variant="destructive"
              data-testid="gc-run"
              onClick={() => setConfirmOpen(true)}
              disabled={!previewGc.data || runGc.isPending}
            >
              {runGc.isPending ? "Collecting…" : "Run garbage collection"}
            </Button>
          </div>

          {previewGc.data && <GcReport data={previewGc.data} />}
          {runGc.data && <GcReport data={runGc.data} />}

          <Confirm
            open={confirmOpen}
            title="Run garbage collection"
            description={
              plannedScope ? (
                <>
                  Removes the planned orphan images and{" "}
                  <span className="text-destructive">
                    deletes app data dirs with no live row. This cannot be undone.
                  </span>
                </>
              ) : (
                <>Removes the planned orphan images. Data dirs are left untouched.</>
              )
            }
            confirmLabel="Collect"
            destructive
            onConfirm={onConfirmRun}
            onCancel={() => setConfirmOpen(false)}
          />
        </div>
      )}
    </Panel>
  );
}
