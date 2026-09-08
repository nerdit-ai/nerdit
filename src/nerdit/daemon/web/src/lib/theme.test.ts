// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { getThemePref, initTheme, setThemePref } from "./theme";

/**
 * What is worth pinning is not "does it write a string" but the two ways this
 * breaks in a user's face: a wrong stamp (the toggle does nothing, or "system"
 * silently pins one mode forever) and a throwing `localStorage` (Safari private
 * browsing, blocked site data) taking the whole dashboard down on boot.
 */
afterEach(() => {
  delete document.documentElement.dataset.theme;
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("theme", () => {
  it("defaults to system with nothing stored", () => {
    expect(getThemePref()).toBe("system");
  });

  it("ignores a corrupt stored value rather than stamping it", () => {
    localStorage.setItem("nerdit.theme", "chartreuse");
    expect(getThemePref()).toBe("system");
    initTheme();
    expect(document.documentElement.dataset.theme).toBeUndefined();
  });

  it("stamps light and dark, and clears the attribute for system", () => {
    setThemePref("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(getThemePref()).toBe("dark");

    setThemePref("light");
    expect(document.documentElement.dataset.theme).toBe("light");

    // `system` must REMOVE the attribute: leaving `data-theme="light"` behind
    // would beat the OS preference, which is the opposite of what it means.
    setThemePref("system");
    expect(document.documentElement.dataset.theme).toBeUndefined();
    expect(getThemePref()).toBe("system");
  });

  it("applies the stored preference at boot", () => {
    localStorage.setItem("nerdit.theme", "dark");
    initTheme();
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("survives a localStorage that throws, and still honours the click", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(getThemePref()).toBe("system");
    expect(() => initTheme()).not.toThrow();
    expect(() => setThemePref("dark")).not.toThrow();
    // Persistence failed; the tab still changed colour.
    expect(document.documentElement.dataset.theme).toBe("dark");
  });
});
