import type { ReactNode } from "react";
import { CopyField, DataRow, Mono, Panel } from "../ui";
import { useProxyStatus } from "../../api/queries";

function yesNo(value: unknown): string {
  return value ? "Yes" : "No";
}

/** Typed ProxyManager state projection: a compact facts list (D8). */
export function ProxyStatusCard() {
  const proxy = useProxyStatus();
  const data = proxy.data;

  if (proxy.isError) {
    return (
      <Panel title="Proxy">
        <p className="px-4 py-3 text-14 text-destructive">Could not load proxy status.</p>
      </Panel>
    );
  }

  if (!data) {
    return (
      <Panel title="Proxy">
        <div className="px-4 py-3" aria-hidden="true">
          <div className="h-4 w-48 rounded-button bg-surface-hover" />
        </div>
      </Panel>
    );
  }

  if (!data.enabled) {
    return (
      <Panel title="Proxy">
        <p className="px-4 py-3 text-14 text-muted-foreground">
          Proxy off. Apps stay on loopback.
        </p>
      </Panel>
    );
  }

  const tlsSynced = (data.tls as { synced?: boolean }).synced;
  const fingerprint = (data.ca as { fingerprint?: string }).fingerprint ?? null;
  const mdnsRegistered = (data.mdns as { registered?: boolean }).registered;
  const mdnsEnabled = (data.mdns as { enabled?: boolean }).enabled;

  const rows: [string, ReactNode][] = [
    ["State", <Mono key="state">{data.state}</Mono>],
    ["Mode", <Mono key="mode">{data.mode}</Mono>],
    ["Hostname", <Mono key="hostname">{data.hostname}</Mono>],
    ["TLS synced", yesNo(tlsSynced)],
    [
      "CA fingerprint",
      fingerprint ? (
        <CopyField key="ca" value={fingerprint} />
      ) : (
        <span key="ca" className="text-subtle-foreground">
          none
        </span>
      )
    ],
    // No route count here by design (D-P23-7): `RoutesCard` right below owns
    // that fact and shows the rows themselves. One place per fact.
    ["mDNS", mdnsEnabled ? yesNo(mdnsRegistered) : "Off"]
  ];

  return (
    <Panel title="Proxy" data-testid="proxy-card">
      <div className="divide-y divide-border">
        {rows.map(([label, value]) => (
          <DataRow key={label} label={label}>
            {value}
          </DataRow>
        ))}
      </div>
    </Panel>
  );
}
