/**
 * Theme preference: light, dark, or follow the OS.
 *
 * The token set in `styles/globals.css` defines light on bare `:root`, dark
 * under `prefers-color-scheme: dark` (guarded by `:root:not([data-theme=
 * "light"])`) and dark again under `:root[data-theme="dark"]`. So the whole job
 * here is stamping one attribute:
 *
 *   light  → data-theme="light"   (beats the OS's dark preference)
 *   dark   → data-theme="dark"    (beats the OS's light default)
 *   system → attribute absent     (the media query decides)
 *
 * Every storage access is wrapped: `localStorage` throws outright in a Safari
 * private window and in any embedding that blocks site data, and a dashboard
 * that white-screens on boot because it wanted to remember a colour would be a
 * poor trade.
 */
export type ThemePref = "light" | "dark" | "system";

const STORAGE_KEY = "nerdit.theme";

function isThemePref(value: unknown): value is ThemePref {
  return value === "light" || value === "dark" || value === "system";
}

/** The stored preference, or `"system"` when absent, unreadable or corrupt. */
export function getThemePref(): ThemePref {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return isThemePref(raw) ? raw : "system";
  } catch {
    return "system";
  }
}

/** Stamp the document for `pref`. Exported for the boot path; safe to re-run. */
function stamp(pref: ThemePref): void {
  const root = document.documentElement;
  if (pref === "system") {
    delete root.dataset.theme;
  } else {
    root.dataset.theme = pref;
  }
}

/**
 * Persist the choice and apply it. The stamp happens even when persistence
 * fails: the operator's click must still change the colours in this tab.
 */
export function setThemePref(pref: ThemePref): void {
  try {
    localStorage.setItem(STORAGE_KEY, pref);
  } catch {
    /* storage unavailable — this tab still honours the choice */
  }
  stamp(pref);
}

/** Called once before render, so the first paint is already the right theme. */
export function initTheme(): void {
  stamp(getThemePref());
}
