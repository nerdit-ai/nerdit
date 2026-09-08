import { beforeEach, describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import { useToastStore } from "../state/toastStore";
import { CODE_COPY, apiErrorCopy, toastApiError } from "./apiErrors";

describe("apiErrorCopy", () => {
  it("uses the daemon message when the code has no override", () => {
    const err = new ApiError("A service named 'api' already exists.", 409, "service.name_taken");
    expect(apiErrorCopy(err)).toBe("A service named 'api' already exists.");
  });

  it("appends the hint as its own sentence", () => {
    const err = new ApiError(
      "Deploy-from-git is disabled on this daemon.",
      403,
      "deploy.git_disabled",
      "Set [git].enabled = true to allow POST /deploy/git."
    );
    expect(apiErrorCopy(err)).toBe(
      "Deploy-from-git is disabled on this daemon. Set [git].enabled = true to allow POST /deploy/git."
    );
  });

  it("punctuates both halves when neither ends in a period", () => {
    const err = new ApiError("Certificate not trusted", 502, "proxy.untrusted", "run nerdit trust");
    expect(apiErrorCopy(err)).toBe("Certificate not trusted. run nerdit trust.");
  });

  it("keeps existing terminal punctuation on the hint", () => {
    const err = new ApiError("Nope", 400, "bad_request", "Try again?");
    expect(apiErrorCopy(err)).toBe("Nope. Try again?");
  });

  it("ignores a blank hint", () => {
    const err = new ApiError("Nope.", 400, "bad_request", "   ");
    expect(apiErrorCopy(err)).toBe("Nope.");
  });

  it("overrides mapped codes with friendlier copy", () => {
    const err = new ApiError("Forbidden", 403, "forbidden");
    expect(apiErrorCopy(err)).toBe(CODE_COPY.forbidden);
    expect(apiErrorCopy(err)).toBe("You don't have permission to do that.");
  });

  it("suppresses the hint on an overridden code (the override is self-sufficient)", () => {
    // These codes are overridden precisely because the daemon talks operator —
    // its hints name ETags/cutover timeouts, which the guidelines ban from the
    // screen — so the verbatim hint goes down with the message.
    const err = new ApiError(
      "The config changed since you last read it.",
      409,
      "config.stale",
      "Re-read the section (GET) and retry with the new ETag."
    );
    expect(apiErrorCopy(err)).toBe("Someone else changed this config. Reload and retry.");
  });

  it("keeps CODE_COPY free of em dashes (the e2e copy rule bans them on screen)", () => {
    for (const copy of Object.values(CODE_COPY)) {
      expect(copy).not.toContain("—");
    }
  });

  it("never shows a raw code on screen", () => {
    for (const [code, copy] of Object.entries(CODE_COPY)) {
      expect(copy).not.toContain(code);
      expect(copy[0]).toBe(copy[0].toUpperCase());
    }
    expect(Object.keys(CODE_COPY).length).toBeLessThanOrEqual(12);
  });

  it("prefixes the context", () => {
    const err = new ApiError("Build failed on step 2.", 400, "deploy.invalid");
    expect(apiErrorCopy(err, "Deploy failed")).toBe("Deploy failed: Build failed on step 2.");
  });

  it("does not double-prefix when the copy already opens with the context", () => {
    const err = new ApiError("Deploy failed: no buildpack matched.", 400, "deploy.no_buildpack");
    expect(apiErrorCopy(err, "Deploy failed")).toBe("Deploy failed: no buildpack matched.");
  });

  it("compares the context case-insensitively", () => {
    const err = new ApiError("deploy failed while cloning.", 400, "deploy.invalid");
    expect(apiErrorCopy(err, "Deploy failed")).toBe("deploy failed while cloning.");
  });

  it("falls back to a plain Error message", () => {
    expect(apiErrorCopy(new Error("Failed to fetch"))).toBe("Failed to fetch");
    expect(apiErrorCopy(new Error("Failed to fetch"), "Deploy failed")).toBe(
      "Deploy failed: Failed to fetch"
    );
  });

  it("stringifies a non-Error throw", () => {
    expect(apiErrorCopy("boom")).toBe("boom");
    expect(apiErrorCopy(null)).toBe("null");
  });

  it("falls back to generic copy when nothing readable is present", () => {
    expect(apiErrorCopy(new ApiError("", 500))).toBe("The request did not go through.");
    expect(apiErrorCopy("")).toBe("The request did not go through.");
  });

  it("recognises a structurally-shaped ApiError across a module boundary", () => {
    const err = Object.assign(new Error("Nope."), {
      name: "ApiError",
      status: 409,
      code: "config.stale",
      hint: "Reload"
    });
    expect(apiErrorCopy(err)).toBe("Someone else changed this config. Reload and retry.");
  });
});

describe("toastApiError", () => {
  beforeEach(() => {
    useToastStore.setState({ toasts: [] });
  });

  it("pushes one error toast carrying the formatted copy", () => {
    toastApiError(new ApiError("Boom.", 500, "deploy.build_failed", "Check the build log"), "Deploy failed");
    const toasts = useToastStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].kind).toBe("error");
    expect(toasts[0].message).toBe("Deploy failed: Boom. Check the build log.");
  });
});
