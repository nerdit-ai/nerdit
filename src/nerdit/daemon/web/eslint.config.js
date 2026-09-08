import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

// Flat config on eslint 8.57 (flat is auto-detected from this filename since
// 8.23) with typescript-eslint 7 — the ESLint-8-compatible major, so no ESLint
// 9 bump is needed. Rules are deliberately non-type-checked: `tsc -b` in
// `npm run build` already carries the type gate, this config carries the
// correctness gate the type checker cannot see (hook ordering and hook
// dependency arrays, above all).
export default tseslint.config(
  {
    ignores: [
      "dist/**",
      "node_modules/**",
      "playwright-report/**",
      "test-results/**"
    ]
  },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [...tseslint.configs.recommended],
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh
    },
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module"
    },
    rules: {
      // The whole reason this config exists: `rules-of-hooks` had never run
      // and was hiding a real "Rendered fewer hooks than expected" crash on
      // the Login magic-link path. Both react-hooks rules are errors, and CI
      // runs `npm run lint` with `--max-warnings 0`, so neither can rot back.
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "error",
      // `allowConstantExport` is the Vite-idiomatic setting: Vite's React
      // plugin refreshes a module that exports a component alongside literal
      // constants just fine.
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true }
      ],
      // `x == null` / `x != null` is the deliberate "null or undefined" test
      // used ~20× across the dashboard against the daemon's optional fields;
      // it is the idiom, not a bug. Everything else must be strict.
      eqeqeq: ["error", "always", { null: "ignore" }]
    }
  },
  {
    // The app entry mounts the router — it is not a fast-refresh boundary,
    // so "a file must export components" does not apply to it.
    files: ["src/main.tsx"],
    rules: { "react-refresh/only-export-components": "off" }
  },
  {
    // The Playwright mock fixtures hand-build API payloads and deliberately
    // type them loosely; they are test doubles, not shipped types.
    files: ["tests/**/*.ts"],
    rules: { "@typescript-eslint/no-explicit-any": "off" }
  },
  {
    // `(string & {})` is the open-string-union idiom — it keeps autocomplete
    // on the known members while still accepting any string, which is exactly
    // how the daemon's forward-compatible enums must be modelled.
    // `ban-types` has no way to express the exception.
    files: ["src/api/types.ts"],
    rules: { "@typescript-eslint/ban-types": "off" }
  }
);
