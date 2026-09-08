import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { SecretsPanel } from "../components/SecretsPanel";
import { AppearanceCard } from "../components/settings/AppearanceCard";
import { HostnameCard } from "../components/settings/HostnameCard";
import { GpuCard } from "../components/settings/GpuCard";
import { DoctorCard } from "../components/settings/DoctorCard";
import { ProxyStatusCard } from "../components/settings/ProxyStatusCard";
import { RoutesCard } from "../components/settings/RoutesCard";
import { DiskCard } from "../components/settings/DiskCard";
import { Button, DataRow, Mono, PageHeader, Panel } from "../components/ui";
import { useAuthRole, useClusterInfo, useClusterStats } from "../api/queries";
import { clearStoredToken, isRememberingToken } from "../lib/auth";

// Settings is the machine page (design guidelines §2): everything about the
// daemon and the box it runs on, in one column of panels. The retired Hardware
// page lives here as the GPU card — a passive telemetry readout was never a
// destination.
//
// There is deliberately no page-level primary action: nothing on this page is
// the one thing an operator came here to do.

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
}

export function Settings() {
  const info = useClusterInfo();
  const stats = useClusterStats();
  const auth = useAuthRole();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const isAdmin = auth.data?.role === "admin";

  function onSignOut() {
    queryClient.clear();
    clearStoredToken();
    navigate("/login", { replace: true });
  }

  const connection: [string, string][] = [
    ["API endpoint", window.location.origin],
    ["Hostname", info.data?.hostname ?? "…"],
    ["Daemon version", info.data?.version ?? stats.data?.daemon_version ?? "…"],
    ["Uptime", info.data ? formatUptime(info.data.uptime_seconds) : "…"],
    ["Token storage", isRememberingToken() ? "this device" : "this tab only"]
  ];

  return (
    <div className="mx-auto max-w-content space-y-7">
      <PageHeader title="Settings" subtitle="This daemon, the machine it runs on, and this browser." />

      <AppearanceCard />

      {isAdmin && <HostnameCard />}

      <GpuCard />

      <DoctorCard />

      <ProxyStatusCard />

      <RoutesCard />

      <DiskCard />

      <Panel title="Shared secrets" data-testid="shared-secrets-card">
        <div className="px-4 py-4">
          <p className="mb-4 text-14 text-muted-foreground">
            Referenced with <Mono>{"${secrets.shared.KEY}"}</Mono>, never injected on their own.
            An app&rsquo;s own secret of the same name wins.
          </p>
          {/* Shared writes are admin-only server-side; hide the form up front.
              Defaults to writable until the role loads, and the panel's 403
              fallback still covers that window. */}
          <SecretsPanel service="shared" canWrite={(auth.data?.role ?? "admin") === "admin"} />
        </div>
      </Panel>

      <Panel title="Connection" data-testid="connection-card">
        <div className="divide-y divide-border">
          {connection.map(([label, value]) => (
            <DataRow key={label} label={label}>
              <Mono>{value}</Mono>
            </DataRow>
          ))}
        </div>
      </Panel>

      <Panel title="Session" data-testid="session-card">
        <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-4">
          <p className="text-14 text-muted-foreground">
            Signing out clears the stored token from this browser.
          </p>
          <Button data-testid="sign-out" onClick={onSignOut}>
            Sign out
          </Button>
        </div>
      </Panel>
    </div>
  );
}
