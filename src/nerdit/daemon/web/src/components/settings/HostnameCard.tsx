import { useEffect, useId, useState } from "react";
import { Button, Confirm, Field, Input, Mono, Notice, Panel } from "../ui";
import { ApiError } from "../../api/client";
import {
  useCapabilities,
  useDaemonConfigSection,
  useDaemonRestart,
  useUpdateDaemonConfig
} from "../../api/queries";
import type { ConfigWriteResponse } from "../../api/types";
import { toast } from "../../state/toastStore";

/**
 * Admin-only hostname editor (D11).
 *
 * The write is a compare-and-set against the version the form was seeded from,
 * so a value changed elsewhere is refused rather than clobbered: the daemon
 * answers `config.stale`, the card re-reads and asks the operator to look
 * again. That mechanism is never named on screen — the operator is told what
 * happened, not which header carried it.
 */
export function HostnameCard() {
  const config = useDaemonConfigSection("proxy");
  const capabilities = useCapabilities();
  const update = useUpdateDaemonConfig("proxy");
  const restart = useDaemonRestart();
  const inputId = useId();

  const [hostname, setHostname] = useState("");
  const [seeded, setSeeded] = useState(false);
  const [preview, setPreview] = useState<ConfigWriteResponse | null>(null);
  const [requiresRestart, setRequiresRestart] = useState(false);
  const [staleNote, setStaleNote] = useState<string | null>(null);
  const [confirmRestart, setConfirmRestart] = useState(false);

  // Seed the input from the stored value once (empty = auto).
  useEffect(() => {
    if (!seeded && config.data) {
      setHostname(String(config.data.view.values.hostname_override ?? ""));
      setSeeded(true);
    }
  }, [config.data, seeded]);

  const mdnsOn = Boolean(config.data?.view.values.mdns);
  const advertised = capabilities.data?.proxy.hostname ?? null;

  // Writes stay disabled until the config (value + version) is loaded: an early
  // Apply would send null unconditionally and clear an existing override.
  const loaded = seeded && Boolean(config.data);
  const stored = String(config.data?.view.values.hostname_override ?? "");
  const dirty = loaded && hostname.trim() !== stored;

  async function onPreview() {
    setStaleNote(null);
    try {
      const trimmed = hostname.trim();
      const res = await update.mutateAsync({
        values: { hostname_override: trimmed === "" ? null : trimmed },
        dryRun: true
      });
      setPreview(res);
      setRequiresRestart(res.requires_restart);
    } catch {
      /* toast handled by the hook */
    }
  }

  async function onApply() {
    setStaleNote(null);
    try {
      const trimmed = hostname.trim();
      const res = await update.mutateAsync({
        values: { hostname_override: trimmed === "" ? null : trimmed },
        etag: config.data?.etag,
        dryRun: false
      });
      setPreview(null);
      setRequiresRestart(res.requires_restart);
      toast("success", "Hostname saved");
      void config.refetch();
    } catch (err) {
      if (err instanceof ApiError && err.code === "config.stale") {
        setPreview(null);
        void config.refetch();
        setSeeded(false);
        setStaleNote("Settings changed elsewhere. Review and retry.");
      }
      /* other errors toast via the hook */
    }
  }

  async function onRestart() {
    setConfirmRestart(false);
    try {
      await restart.mutateAsync(undefined);
      toast("success", "Restarting daemon");
    } catch {
      /* 409 daemon.restart_in_progress and others toast via the hook */
    }
  }

  return (
    <Panel title="Hostname" data-testid="hostname-card">
      <div className="flex flex-col gap-4 px-4 py-4">
        <Field
          label="Advertised hostname"
          htmlFor={inputId}
          hint="Empty uses the auto name."
        >
          <Input
            id={inputId}
            mono
            data-testid="hostname-input"
            value={hostname}
            onChange={(e) => setHostname(e.target.value)}
            placeholder="auto"
          />
        </Field>

        {mdnsOn && (
          <div className="text-14">
            {advertised && (
              <p className="text-muted-foreground">
                Advertised as <Mono className="text-foreground">{advertised}</Mono>
              </p>
            )}
            <p className="text-subtle-foreground">mDNS needs a single-label .local name.</p>
          </div>
        )}

        <div className="flex gap-2">
          <Button data-testid="hostname-preview" onClick={onPreview} disabled={!loaded || update.isPending}>
            Preview
          </Button>
          <Button
            variant="primary"
            data-testid="hostname-apply"
            onClick={onApply}
            disabled={!dirty || update.isPending}
          >
            Apply
          </Button>
        </div>

        {staleNote && (
          <Notice tone="warning" data-testid="hostname-stale">
            {staleNote}
          </Notice>
        )}

        {preview && (
          <div
            data-testid="hostname-preview-diff"
            className="rounded-card border border-border bg-background p-3 text-14"
          >
            <p className="mb-2 font-medium text-foreground">Preview</p>
            {preview.diff.length === 0 ? (
              <p className="text-muted-foreground">No changes.</p>
            ) : (
              <ul className="space-y-1">
                {preview.diff.map((d) => (
                  <li key={d.key}>
                    <Mono className="text-muted-foreground">
                      {d.key}: {JSON.stringify(d.old)} → {JSON.stringify(d.new)}
                    </Mono>
                  </li>
                ))}
              </ul>
            )}
            {preview.requires_restart && (
              <p className="mt-2 text-subtle-foreground">Applies after restart.</p>
            )}
          </div>
        )}

        {requiresRestart && !preview && (
          <div className="rounded-card border border-border bg-background p-3 text-14">
            <p className="text-muted-foreground">Applies after restart.</p>
            <Button
              className="mt-2"
              data-testid="daemon-restart"
              onClick={() => setConfirmRestart(true)}
              disabled={restart.isPending}
            >
              Restart daemon
            </Button>
          </div>
        )}
      </div>

      <Confirm
        open={confirmRestart}
        title="Restart daemon"
        description="Drains builds, then restarts. Brief downtime."
        confirmLabel="Restart"
        destructive
        onConfirm={onRestart}
        onCancel={() => setConfirmRestart(false)}
      />
    </Panel>
  );
}
