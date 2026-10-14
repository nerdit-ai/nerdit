// Shared source of truth for the dashboard's primary navigation. Imported by
// the sidebar (`AppShell`) and the ⌘K command palette so the two cannot drift.
//
// One flat group (design guidelines §2): six destinations do not need sections,
// and the audit found the three-group split was mostly filing cabinets for
// pages that have since been retired (Store as a destination, Hardware as a
// page). The `NavGroup` shape survives the flattening because the palette maps
// over groups and renders `heading` above each; keeping it costs one wrapper
// and leaves the door open if the list ever earns a second section.

export interface NavItem {
  to: string;
  label: string;
  /** The page's API is admin-only (GET /tokens, /audit); hidden for other roles. */
  adminOnly?: boolean;
}

export interface NavGroup {
  id: string;
  /** Sidebar section label; null renders no text (the flat list needs none). */
  label: string | null;
  /** Command-palette group heading (always present). */
  heading: string;
  items: readonly NavItem[];
}

/** Show admin-only items while the role is unknown, so admins see no flicker. */
export function navItemVisible(item: NavItem, role: string | undefined): boolean {
  return !item.adminOnly || role === undefined || role === "admin";
}

export const navGroups: readonly NavGroup[] = [
  {
    id: "nav",
    label: null,
    heading: "Navigation",
    items: [
      { to: "/", label: "Apps" },
      { to: "/models", label: "Models" },
      { to: "/databases", label: "Databases" },
      { to: "/tokens", label: "Tokens", adminOnly: true },
      { to: "/activity", label: "Activity", adminOnly: true },
      { to: "/settings", label: "Settings" }
    ]
  }
];
