import { useState } from "react";
import { Button, DataRow, Mono, Panel } from "../ui";
import { useClusterInfo } from "../../api/queries";
import { getThemePref, setThemePref, type ThemePref } from "../../lib/theme";

// Appearance: the operator's theme choice, and the analytics disclosure.
//
// The disclosure is a READ, never a toggle (design guidelines §6): PostHog is
// configured by whoever runs the daemon (`[posthog]`), and the SPA only reports
// what it was handed. `/cluster/info` returns the publishable key + host when
// analytics are on and nulls otherwise, which is exactly the same condition
// `AppShell` uses to decide whether to init the tracker — so the line can never
// disagree with what the page actually does.

const THEMES: { value: ThemePref; label: string }[] = [
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
  { value: "system", label: "System" }
];

export function AppearanceCard() {
  const [theme, setTheme] = useState<ThemePref>(() => getThemePref());
  const info = useClusterInfo();

  function choose(next: ThemePref) {
    setThemePref(next);
    setTheme(next);
  }

  const analyticsOn = Boolean(info.data?.posthog_key && info.data.posthog_host);

  return (
    <Panel title="Appearance">
      <div className="divide-y divide-border">
        <DataRow label="Theme">
          <div
            data-testid="theme-toggle"
            role="group"
            aria-label="Theme"
            className="inline-flex items-center gap-1"
          >
            {THEMES.map((option) => (
              <Button
                key={option.value}
                size="sm"
                data-testid={`theme-${option.value}`}
                variant={theme === option.value ? "secondary" : "ghost"}
                aria-pressed={theme === option.value}
                onClick={() => choose(option.value)}
              >
                {option.label}
              </Button>
            ))}
          </div>
        </DataRow>

        <DataRow label="Usage analytics">
          <p data-testid="analytics-disclosure" className="text-14 text-muted-foreground">
            {analyticsOn ? (
              <>
                Active. Page views only, configured by this daemon&rsquo;s operator (
                <Mono>[posthog]</Mono> in the daemon config).
              </>
            ) : (
              <>
                Off. Enable via <Mono>[posthog]</Mono> in the daemon config.
              </>
            )}
          </p>
        </DataRow>
      </div>
    </Panel>
  );
}
