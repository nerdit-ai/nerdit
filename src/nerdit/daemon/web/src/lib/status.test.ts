import { describe, expect, it } from "vitest";
import { serviceStatusLabel, serviceStatusTone } from "./status";

describe("serviceStatusLabel", () => {
  it("uses the §3 word table", () => {
    expect(serviceStatusLabel("running")).toBe("Running");
    expect(serviceStatusLabel("stopped")).toBe("Stopped");
    expect(serviceStatusLabel("failed")).toBe("Failed");
    expect(serviceStatusLabel("building")).toBe("Building");
    expect(serviceStatusLabel("restarting")).toBe("Restarting");
    expect(serviceStatusLabel("queued")).toBe("Queued");
  });

  it("never says Degraded on screen", () => {
    expect(serviceStatusLabel("degraded")).toBe("Running · unhealthy");
    expect(serviceStatusLabel("running", false)).toBe("Running · unhealthy");
    // Repo convention: no em dash in rendered copy.
    expect(serviceStatusLabel("degraded")).not.toContain("—");
  });

  it("treats an unknown health as no claim, not as unhealthy", () => {
    expect(serviceStatusLabel("running", undefined)).toBe("Running");
    expect(serviceStatusLabel("running", true)).toBe("Running");
  });

  it("passes an unrecognised daemon status through capitalized", () => {
    expect(serviceStatusLabel("provisioning")).toBe("Provisioning");
    expect(serviceStatusLabel("")).toBe("");
  });
});

describe("serviceStatusTone", () => {
  it("maps the run states to one tone each", () => {
    expect(serviceStatusTone("running")).toBe("success");
    expect(serviceStatusTone("running", true)).toBe("success");
    expect(serviceStatusTone("running", false)).toBe("warning");
    expect(serviceStatusTone("degraded")).toBe("warning");
    expect(serviceStatusTone("failed")).toBe("destructive");
    expect(serviceStatusTone("stopped")).toBe("muted");
    expect(serviceStatusTone("building")).toBe("muted");
    expect(serviceStatusTone("restarting")).toBe("muted");
    expect(serviceStatusTone("queued")).toBe("muted");
  });

  it("falls back to muted rather than inventing alarm", () => {
    expect(serviceStatusTone("provisioning")).toBe("muted");
  });
});
