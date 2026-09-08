import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:9321"
    }
  },
  test: {
    // Unit tests only — the Playwright e2e suite under tests/e2e uses
    // *.spec.ts and is run via `npm run e2e`, never by vitest.
    include: ["src/**/*.test.{ts,tsx}"],
    environment: "node"
  }
});
