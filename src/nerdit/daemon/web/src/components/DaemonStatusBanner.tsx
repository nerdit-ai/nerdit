import { useEffect, useState } from "react";
import { AlertTriangle, WifiOff } from "lucide-react";
import { useDaemonStatus } from "../api/queries";
import { Banner } from "./ui";

function formatAgo(date: Date | null): string {
  if (!date) return "–";
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

/**
 * The daemon is unreachable, or reconnecting. Rendered on the shared `Banner`
 * primitive, in normal flow: the shell's banner slot sits full-width above the
 * sidebar + content row, so the strip pushes the page down instead of floating
 * over it. It used to draw its own `fixed inset-x-0 top-0` wrapper, which gave
 * the slot no height and left the offline banner overlapping the mobile
 * header; the slot is the one place that decides banner placement now, and
 * `UnlinkedBanner` already behaved this way.
 *
 * `role="status"` lives on the wrapper rather than inside `Banner`: the
 * primitive is presentational by design, and the semantics differ per banner.
 */
export function DaemonStatusBanner() {
  const { status, lastSeen } = useDaemonStatus();
  const [, force] = useState(0);

  // Re-render every 5s while we are not "online" so the "last seen" string
  // stays current without polling the daemon harder than needed.
  useEffect(() => {
    if (status === "online") return;
    const interval = window.setInterval(() => force((n) => n + 1), 5000);
    return () => window.clearInterval(interval);
  }, [status]);

  if (status === "online") return null;

  const isOffline = status === "offline";
  const Icon = isOffline ? WifiOff : AlertTriangle;
  const label = isOffline
    ? `Daemon offline (last seen ${formatAgo(lastSeen)})`
    : "Reconnecting to nerditd…";

  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="daemon-status-banner"
    >
      <Banner tone={isOffline ? "destructive" : "warning"}>
        <span className="flex items-center gap-2 font-medium">
          <Icon aria-hidden="true" className="h-4 w-4 shrink-0" />
          {label}
        </span>
      </Banner>
    </div>
  );
}
