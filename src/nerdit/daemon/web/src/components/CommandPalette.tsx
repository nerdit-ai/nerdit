import { Fragment, useEffect } from "react";
import { Command } from "cmdk";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchAllServices } from "../api/queries";
import { navGroups } from "../shell/navItems";

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

export function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const navigate = useNavigate();

  // Dynamic app jump-list. Cache-shared with `useAllBindings` (same queryKey +
  // the same cursor-walking queryFn) so the palette rides the apps list's
  // fetch; `enabled: open` keeps it off the wire until the palette is summoned.
  const services = useQuery({
    queryKey: ["services", "all-for-bindings"],
    queryFn: fetchAllServices,
    staleTime: 10000,
    enabled: open
  });
  const apps = (services.data?.items ?? []).filter((s) => s.kind === "service");

  // Close on Escape (cmdk handles this internally for input focus, but the
  // outer overlay needs its own listener for clicks landing elsewhere).
  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Hooks run unconditionally; the overlay is what gates on `open`.
  if (!open) return null;

  function go(to: string, state?: Record<string, unknown>) {
    navigate(to, state ? { state } : undefined);
    onClose();
  }

  // `focus:outline-none` is authored, not an oversight: the selected row is
  // marked by cmdk's `aria-selected` skin below, and the global focus ring
  // would draw a second, competing highlight on the same row.
  const itemCls =
    "flex cursor-pointer items-center justify-between gap-3 rounded-button px-3 py-2 text-14 text-foreground focus:outline-none aria-selected:bg-surface-hover aria-selected:text-foreground";
  // cmdk owns the heading element, so the `.label` style is restated here as
  // utilities: an arbitrary variant can only compose Tailwind utilities, and
  // `.label` is a component class. Values are the same 12/500/uppercase/0.06em.
  const groupCls = [
    "[&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-2",
    "[&_[cmdk-group-heading]]:text-12 [&_[cmdk-group-heading]]:font-medium",
    "[&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-[0.06em]",
    "[&_[cmdk-group-heading]]:text-muted-foreground"
  ].join(" ");
  const suffixCls = "shrink-0 font-mono text-12 text-subtle-foreground";

  return (
    <div
      role="presentation"
      onClick={onClose}
      className="fixed inset-0 z-[60] flex items-start justify-center bg-overlay pt-[15vh] backdrop-blur-sm"
    >
      <div
        onClick={(event) => event.stopPropagation()}
        className="w-full max-w-lg rounded-panel border border-border bg-surface shadow-lg"
      >
        <Command label="Command palette" aria-label="Command palette">
          <Command.Input
            autoFocus
            placeholder="Jump to…"
            className="w-full rounded-t-panel border-b border-border bg-transparent px-4 py-3 text-14 text-foreground placeholder:text-subtle-foreground focus:outline-none"
          />
          <Command.List className="max-h-80 overflow-y-auto p-2">
            <Command.Empty className="px-3 py-6 text-center text-14 text-muted-foreground">
              No matches.
            </Command.Empty>

            {/* Apps first: jump plus per-app quick actions (logs, deploy).
                Best-effort — loading/error renders nothing. */}
            {apps.length > 0 && (
              <Command.Group heading="Apps" className={groupCls}>
                {apps.map((app) => {
                  const to = `/projects/${encodeURIComponent(app.name)}`;
                  return (
                    <Fragment key={app.id}>
                      <Command.Item value={app.name} onSelect={() => go(to)} className={itemCls}>
                        <span className="truncate">{app.name}</span>
                        <span className={suffixCls}>{to}</span>
                      </Command.Item>
                      <Command.Item
                        value={`${app.name} logs`}
                        onSelect={() => go(`${to}/logs`)}
                        className={itemCls}
                      >
                        <span className="truncate">{app.name}</span>
                        <span className={suffixCls}>logs</span>
                      </Command.Item>
                      <Command.Item
                        value={`${app.name} redeploy`}
                        onSelect={() => go(to, { redeploy: true })}
                        className={itemCls}
                      >
                        <span className="truncate">{app.name}</span>
                        <span className={suffixCls}>redeploy</span>
                      </Command.Item>
                    </Fragment>
                  );
                })}
              </Command.Group>
            )}

            {navGroups.map((group) => (
              <Command.Group key={group.id} heading={group.heading} className={groupCls}>
                {group.items.map(({ to, label }) => (
                  <Command.Item key={to} value={label} onSelect={() => go(to)} className={itemCls}>
                    <span>{label}</span>
                    <span className={suffixCls}>{to}</span>
                  </Command.Item>
                ))}
              </Command.Group>
            ))}
          </Command.List>
          <div className="flex items-center justify-between border-t border-border px-3 py-2 text-12 text-subtle-foreground">
            <span>↵ to open</span>
            <span>Esc to close</span>
          </div>
        </Command>
      </div>
    </div>
  );
}
