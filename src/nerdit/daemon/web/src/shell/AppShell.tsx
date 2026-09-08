import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { LogOut, Menu as MenuIcon, Search, X } from "lucide-react";
import { Link, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useClusterEvents } from "../api/clusterEvents";
import { useAuthRole, useClusterInfo, useDaemonStatus } from "../api/queries";
import { CommandPalette } from "../components/CommandPalette";
import { DaemonStatusBanner } from "../components/DaemonStatusBanner";
import { UnlinkedBanner } from "../components/UnlinkedBanner";
import {
  capturePageview,
  initAnalytics,
  isAnalyticsEnabled,
  registerProperties,
  resetAnalytics
} from "../lib/analytics";
import { clearStoredToken } from "../lib/auth";
import { navGroups, type NavItem } from "./navItems";

// A nav entry is active on an exact match or when the URL is a detail route
// under it. Two carve-outs, both about the Apps row:
// - "/" would prefix-match everything, so it is exact-only...
// - ...but the app detail page still lives at `/projects/:name` (W4 moves it)
//   and `/services/:ident` is the bookmark shim, so both keep Apps highlighted.
function isNavActive(to: string, pathname: string): boolean {
  if (to === "/") {
    return pathname === "/" || pathname.startsWith("/projects") || pathname.startsWith("/services");
  }
  if (pathname === to) return true;
  return pathname.startsWith(`${to}/`);
}

function NavItemLink({
  to,
  label,
  pathname,
  onNavigate
}: NavItem & { pathname: string; onNavigate?: () => void }) {
  const active = isNavActive(to, pathname);
  return (
    <Link
      to={to}
      aria-current={active ? "page" : undefined}
      onClick={onNavigate}
      data-testid={`nav-link-${label.toLowerCase()}`}
      className={`block rounded-button px-3 py-2 text-14 transition-colors ${
        active
          ? "bg-surface-hover font-medium text-foreground"
          : "text-muted-foreground hover:bg-surface-hover hover:text-foreground"
      }`}
    >
      {label}
    </Link>
  );
}

// One flat list. The uppercase group labels are retired with the groups they
// labelled: six destinations do not need filing, and `navGroups` keeps its
// shape only because the palette still groups by heading.
function DashboardNavigation({
  pathname,
  onNavigate
}: {
  pathname: string;
  onNavigate?: () => void;
}) {
  return (
    <nav aria-label="Dashboard" className="space-y-1">
      {navGroups.map((group) =>
        group.items.map((item) => (
          <NavItemLink key={item.to} {...item} pathname={pathname} onNavigate={onNavigate} />
        ))
      )}
    </nav>
  );
}

function Brand({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <Link
      to="/"
      aria-label="Nerdit apps"
      onClick={onNavigate}
      className="inline-flex items-center text-16 font-semibold tracking-tight text-foreground"
    >
      Nerdit
    </Link>
  );
}

/**
 * The one banner slot (design guidelines §4).
 *
 * Ordered precedence, decided here rather than by two components reading each
 * other's conditions in comments: an unreachable daemon outranks an unlinked
 * one, because when the daemon is gone its cached capabilities response is the
 * last thing worth believing about this node's link state. Never both.
 *
 * It sits ABOVE the sidebar/content row at full viewport width, which is the
 * fix for the recorded bug where a banner rendered inside the content column
 * was clipped behind the sidebar.
 */
function BannerSlot() {
  const { status } = useDaemonStatus();
  return (
    <div data-testid="banner-slot">
      {status === "online" ? <UnlinkedBanner /> : <DaemonStatusBanner />}
    </div>
  );
}

function SidebarFooter({
  onSearch,
  onSignOut,
  showSearch
}: {
  onSearch?: () => void;
  onSignOut: () => void;
  showSearch: boolean;
}) {
  const rowCls =
    "flex min-h-11 w-full items-center gap-2 rounded-button px-3 py-2 text-left text-14 text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground";
  return (
    <footer className="shrink-0 space-y-1 border-t border-border px-3 py-3">
      {showSearch && onSearch && (
        <button type="button" onClick={onSearch} className={`${rowCls} justify-between`}>
          <span className="flex items-center gap-2">
            <Search size={16} aria-hidden="true" />
            Search
          </span>
          <kbd className="font-mono text-12 text-subtle-foreground">⌘K</kbd>
        </button>
      )}
      <button type="button" onClick={onSignOut} className={rowCls}>
        <LogOut size={16} aria-hidden="true" />
        Sign out
      </button>
    </footer>
  );
}

export function AppShell() {
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const mobileNavRef = useRef<HTMLElement>(null);
  const restoreMenuFocusRef = useRef(false);

  useClusterEvents(queryClient);

  // Product analytics (feat/posthog). The project key rides on the
  // authenticated /cluster/info read, so init here (post-login) once it lands;
  // then emit a manual page view on every router navigation. All calls no-op
  // when the daemon ships no key (inert dashboard).
  const { data: clusterInfo } = useClusterInfo();
  const { data: authRole } = useAuthRole();
  const [analyticsReady, setAnalyticsReady] = useState(false);

  useEffect(() => {
    if (!clusterInfo?.posthog_key || !clusterInfo.posthog_host) return;
    initAnalytics(
      { key: clusterInfo.posthog_key, host: clusterInfo.posthog_host },
      { hostname: clusterInfo.hostname, daemon_version: clusterInfo.version }
    );
    setAnalyticsReady(isAnalyticsEnabled());
  }, [
    clusterInfo?.posthog_key,
    clusterInfo?.posthog_host,
    clusterInfo?.hostname,
    clusterInfo?.version
  ]);

  // Role resolves from a separate request; register it whenever it lands so
  // events can be segmented by role without ever sending the bearer token.
  useEffect(() => {
    if (authRole?.role) registerProperties({ role: authRole.role });
  }, [authRole?.role]);

  // Emit a page view on every navigation. `analyticsReady` is a dependency so
  // the entry page view fires once PostHog finishes initializing — on a cold
  // load this effect first runs while /cluster/info is still pending (a no-op),
  // and re-runs when init completes so the landing route is not dropped.
  useEffect(() => {
    capturePageview(location.pathname);
  }, [location.pathname, analyticsReady]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((prev) => !prev);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    function onStorage(event: StorageEvent) {
      if (event.key !== "nerdit.token" && event.key !== "nerdit.token.persist") {
        return;
      }
      if (!localStorage.getItem("nerdit.token")) {
        queryClient.clear();
        navigate("/login", { replace: true });
      }
    }
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [navigate, queryClient]);

  // Route changes close the drawer even when navigation happened outside it.
  useEffect(() => {
    setMobileNavOpen(false);
  }, [location.pathname]);

  // Keep keyboard focus inside the modal navigation and prevent the page
  // beneath it from scrolling. Focus returns to the menu trigger on close.
  useEffect(() => {
    if (!mobileNavOpen) {
      if (restoreMenuFocusRef.current) {
        restoreMenuFocusRef.current = false;
        menuButtonRef.current?.focus();
      }
      return;
    }

    restoreMenuFocusRef.current = true;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusFrame = window.requestAnimationFrame(() => closeButtonRef.current?.focus());

    function onDrawerKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        setMobileNavOpen(false);
        return;
      }
      if (event.key !== "Tab") return;

      const focusable = Array.from(
        mobileNavRef.current?.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
        ) ?? []
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", onDrawerKeyDown);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onDrawerKeyDown);
    };
  }, [mobileNavOpen]);

  function onSignOut() {
    setMobileNavOpen(false);
    resetAnalytics();
    queryClient.clear();
    clearStoredToken();
    navigate("/login", { replace: true });
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <BannerSlot />
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />

      <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-border bg-background px-4 lg:hidden">
        <Brand />
        <div className="flex items-center gap-1">
          <button
            type="button"
            aria-label="Open command palette"
            onClick={() => setPaletteOpen(true)}
            className="inline-flex h-11 w-11 items-center justify-center rounded-button text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
          >
            <Search size={18} aria-hidden="true" />
          </button>
          <button
            ref={menuButtonRef}
            type="button"
            aria-label="Open navigation"
            aria-expanded={mobileNavOpen}
            aria-controls="mobile-dashboard-navigation"
            onClick={() => setMobileNavOpen(true)}
            className="inline-flex h-11 w-11 items-center justify-center rounded-button text-foreground transition-colors hover:bg-surface-hover"
          >
            <MenuIcon size={20} aria-hidden="true" />
          </button>
        </div>
      </header>

      <div className="flex">
        <aside
          data-testid="app-sidebar"
          className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-border bg-surface lg:flex"
        >
          <div className="flex h-14 shrink-0 items-center border-b border-border px-4">
            <Brand />
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-3 py-4">
            <DashboardNavigation pathname={location.pathname} />
          </div>
          <SidebarFooter showSearch onSearch={() => setPaletteOpen(true)} onSignOut={onSignOut} />
        </aside>

        <main className="min-w-0 flex-1 px-4 py-6 lg:px-8 lg:py-8">
          <div className="mx-auto max-w-content">
            <Outlet />
          </div>
        </main>
      </div>

      {mobileNavOpen && (
        <>
          <button
            type="button"
            aria-label="Close navigation"
            onClick={() => setMobileNavOpen(false)}
            className="fixed inset-0 z-[60] bg-overlay lg:hidden"
          />
          <aside
            id="mobile-dashboard-navigation"
            ref={mobileNavRef}
            role="dialog"
            aria-modal="true"
            aria-label="Dashboard navigation"
            className="fixed inset-y-0 left-0 z-[70] flex w-[calc(100%-40px)] max-w-xs flex-col border-r border-border bg-surface lg:hidden"
          >
            <div className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
              <Brand onNavigate={() => setMobileNavOpen(false)} />
              <button
                ref={closeButtonRef}
                type="button"
                aria-label="Close navigation"
                onClick={() => setMobileNavOpen(false)}
                className="inline-flex h-11 w-11 items-center justify-center rounded-button text-foreground transition-colors hover:bg-surface-hover"
              >
                <X size={20} aria-hidden="true" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto px-3 py-4">
              <DashboardNavigation
                pathname={location.pathname}
                onNavigate={() => setMobileNavOpen(false)}
              />
            </div>
            <SidebarFooter showSearch={false} onSignOut={onSignOut} />
          </aside>
        </>
      )}
    </div>
  );
}
