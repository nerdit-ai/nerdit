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
}

export interface NavGroup {
  id: string;
  /** Sidebar section label; null renders no text (the flat list needs none). */
  label: string | null;
  /** Command-palette group heading (always present). */
  heading: string;
  items: readonly NavItem[];
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
      { to: "/tokens", label: "Tokens" },
      { to: "/activity", label: "Activity" },
      { to: "/settings", label: "Settings" }
    ]
  }
];
