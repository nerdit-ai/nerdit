export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Every colour is a plain var() reference to the console palette in
      // styles/globals.css. The values are HEX, so the old
      // `rgb(var(--x) / <alpha-value>)` pattern is retired along with it:
      // opacity modifiers (`bg-surface/70`) no longer work on these
      // utilities. Where a translucent value is needed, the token carries the
      // alpha itself — `overlay` is the one such token today.
      colors: {
        background: "var(--color-background)",
        surface: "var(--color-surface)",
        "surface-hover": "var(--color-surface-hover)",
        foreground: "var(--color-foreground)",
        "muted-foreground": "var(--color-muted-foreground)",
        "subtle-foreground": "var(--color-subtle-foreground)",
        border: {
          DEFAULT: "var(--color-border)",
          strong: "var(--color-border-strong)"
        },
        primary: {
          DEFAULT: "var(--color-primary)",
          hover: "var(--color-primary-hover)",
          subtle: "var(--color-primary-subtle)",
          foreground: "var(--color-primary-foreground)"
        },
        success: {
          DEFAULT: "var(--color-success)",
          subtle: "var(--color-success-subtle)",
          foreground: "var(--color-success-foreground)"
        },
        warning: {
          DEFAULT: "var(--color-warning)",
          subtle: "var(--color-warning-subtle)",
          foreground: "var(--color-warning-foreground)"
        },
        destructive: {
          DEFAULT: "var(--color-destructive)",
          subtle: "var(--color-destructive-subtle)",
          border: "var(--color-destructive-border)"
        },
        statement: {
          DEFAULT: "var(--color-statement)",
          foreground: "var(--color-statement-foreground)",
          muted: "var(--color-statement-muted)"
        },
        overlay: "var(--color-overlay)"
      },
      // `colors.border.strong` flattens to the utility `border-border-strong`
      // (prefix + flattened key), so the contract's `border-strong` class needs
      // this one alias to exist. The colour value is the same token; nothing
      // else is aliased. `bg-border-strong` needs no help — it comes from
      // `colors` directly.
      borderColor: {
        strong: "var(--color-border-strong)"
      },
      fontFamily: {
        sans: [
          '"Geist Variable"',
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif"
        ],
        mono: [
          '"Geist Mono Variable"',
          '"SFMono-Regular"',
          '"Cascadia Code"',
          '"Roboto Mono"',
          "Consolas",
          "monospace"
        ]
      },
      // The console scale, added beside Tailwind's defaults (which W3 removes
      // from the pages as it migrates them). Body reference is 14/22; mono 13.
      fontSize: {
        12: ["12px", { lineHeight: "18px" }],
        13: ["13px", { lineHeight: "20px" }],
        14: ["14px", { lineHeight: "22px" }],
        16: ["16px", { lineHeight: "24px" }],
        20: ["20px", { lineHeight: "28px" }],
        24: ["24px", { lineHeight: "32px" }],
        30: ["30px", { lineHeight: "38px" }]
      },
      borderRadius: {
        button: "6px",
        card: "8px",
        panel: "8px",
        code: "6px"
      },
      transitionDuration: {
        fast: "var(--motion-fast)",
        normal: "var(--motion-normal)",
        reveal: "var(--motion-reveal)"
      },
      transitionTimingFunction: {
        reveal: "var(--motion-reveal-easing)"
      },
      maxWidth: { content: "1200px" }
    }
  },
  plugins: []
};
