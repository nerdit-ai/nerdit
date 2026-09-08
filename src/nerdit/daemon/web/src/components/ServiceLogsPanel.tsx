import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Copy, Download } from "lucide-react";
import { useServiceLogs } from "../api/queries";
import { Badge, Button, Notice, Select } from "./ui";
import { serviceStatusLabel, serviceStatusTone } from "../lib/status";
import { toast } from "../state/toastStore";
import type { JobStatus, LogEntry } from "../api/types";

export interface ServiceLogsPanelProps {
  ident: string;
  /** Current service status — shown next to the log header so the `building`
   * phase is distinguishable ((P34) build output now carries stream=build). */
  status: JobStatus;
}

const TAIL_OPTIONS = [100, 500, 1000, 5000];

function format(entry: LogEntry): string {
  const time = entry.timestamp.replace("T", " ").slice(0, 19);
  return `[${time}] ${entry.stream.padEnd(6)} ${entry.message}`;
}

/**
 * Service log viewer. Polling-only by design: `GET /services/{ident}/logs`
 * has no SSE variant (plan trap 1), so this refetches the bounded tail every
 * second while "follow" is on — a services row is long-lived, unlike a batch
 * job, so a bounded rolling window beats an unbounded accumulator.
 */
export function ServiceLogsPanel({ ident, status }: ServiceLogsPanelProps) {
  const [tail, setTail] = useState(500);
  const [follow, setFollow] = useState(true);
  const containerRef = useRef<HTMLPreElement>(null);
  const tailId = useId();

  const logs = useServiceLogs(ident, 0, follow, tail);
  // Wrapped so the `?? []` fallback is not a fresh array every render — an
  // unstable reference here defeats every memo that depends on it.
  const entries = useMemo(() => logs.data ?? [], [logs.data]);

  const text = useMemo(() => entries.map(format).join("\n"), [entries]);

  // Auto-scroll while following.
  const lastId = entries.length > 0 ? entries[entries.length - 1].id : 0;
  useEffect(() => {
    if (!follow) return;
    const node = containerRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [lastId, follow]);

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(text);
      toast("success", "Logs copied");
    } catch {
      toast("error", "Could not access clipboard");
    }
  }

  function onDownload() {
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${ident}.log`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="flex items-center gap-2 text-13 text-muted-foreground">
          <Badge tone={serviceStatusTone(status)}>{serviceStatusLabel(status)}</Badge>
          {status === "building"
            ? "Build in progress. Output streams below."
            : follow
              ? "Polling logs every 1s"
              : "Paused"}
        </span>
        <div className="flex items-center gap-2">
          <label className="label" htmlFor={tailId}>
            Tail
          </label>
          <Select
            id={tailId}
            value={tail}
            onChange={(event) => setTail(Number(event.target.value))}
            className="h-8 w-24"
          >
            {TAIL_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </Select>
          <Button
            size="sm"
            variant={follow ? "primary" : "secondary"}
            aria-pressed={follow}
            onClick={() => setFollow((prev) => !prev)}
          >
            {follow ? "Following" : "Follow"}
          </Button>
          <Button size="sm" onClick={onCopy}>
            <Copy size={16} aria-hidden="true" /> Copy
          </Button>
          <Button size="sm" onClick={onDownload}>
            <Download size={16} aria-hidden="true" /> Download
          </Button>
        </div>
      </div>
      {logs.error && <Notice tone="destructive">{(logs.error as Error).message}</Notice>}
      {/* The log body is a `<pre>`, not `Mono`: it needs the scroll ref and
          `<pre>`'s own whitespace handling. Mono-13 is applied by hand here —
          the one place in the panel that is not a primitive. */}
      <pre
        ref={containerRef}
        className="h-96 overflow-auto rounded-card border border-border bg-surface-hover px-3 py-2 font-mono text-13 text-foreground"
      >
        {entries.length === 0 ? <span className="text-muted-foreground">No logs yet.</span> : text}
      </pre>
    </div>
  );
}
