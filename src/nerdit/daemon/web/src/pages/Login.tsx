import { FormEvent, useEffect, useRef, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { Button, Checkbox, Field, Input, Mono, Notice } from "../components/ui";
import { getStoredToken, isRememberingToken, storeToken } from "../lib/auth";

/**
 * Sign-in. One narrow panel on the page ground, one field, one action.
 *
 * The auth flow is untouched by the redesign: the magic-link fragment is still
 * consumed on mount, `/auth/check` is still the only call, and the token is
 * still stored before navigating. Only the skin moved onto the design system.
 */
export function Login() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [remember, setRemember] = useState(isRememberingToken);

  // The magic-link path (`#token=…`). This effect MUST be declared BEFORE the
  // early-return guard below, and every hook in this component with it: the
  // effect calls submitToken, which calls storeToken *before* navigate, so the
  // very next render sees a truthy getStoredToken(), takes the early return,
  // and never reaches a hook declared past it — React then throws "Rendered
  // fewer hooks than expected" and the sign-in screen crashes. `submitToken`
  // is a hoisted function declaration, so referencing it here is safe.
  useEffect(() => {
    const hash = window.location.hash.startsWith("#") ? window.location.hash.slice(1) : "";
    const params = new URLSearchParams(hash);
    const hashToken = params.get("token");
    if (!hashToken) {
      return;
    }

    setToken(hashToken);
    setError(null);
    void submitToken(hashToken).finally(() => {
      window.history.replaceState(null, "", window.location.pathname + window.location.search);
    });
    // Mount-only by design: the magic link is consumed once, and submitToken
    // is re-created on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (getStoredToken()) {
    return <Navigate to="/" replace />;
  }

  async function submitToken(rawToken: string): Promise<boolean> {
    const trimmed = rawToken.trim();
    setToken(trimmed);

    if (!trimmed) {
      setError("Token is required.");
      inputRef.current?.focus();
      return false;
    }

    setSubmitting(true);
    setError(null);

    try {
      await api<{ ok: true }>("/auth/check", {
        headers: { Authorization: `Bearer ${trimmed}` }
      });
      storeToken(trimmed, remember);
      navigate("/", { replace: true });
      return true;
    } catch {
      setError("Invalid token. Please try again.");
      inputRef.current?.focus();
      return false;
    } finally {
      setSubmitting(false);
    }
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await submitToken(token);
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-6">
      <div className="w-full max-w-sm">
        <h1 className="text-20 font-semibold tracking-tight text-foreground">Nerdit</h1>
        <p className="mt-1 text-14 text-muted-foreground">
          Sign in with the token this daemon printed. From a shell on the machine:{" "}
          <Mono>nerdit token</Mono>
        </p>

        <form className="mt-6 space-y-4" onSubmit={onSubmit}>
          <Field label="Paste your nerdit token" htmlFor="token-input">
            <Input
              id="token-input"
              ref={inputRef}
              type="password"
              value={token}
              onChange={(event) => setToken(event.target.value)}
              spellCheck={false}
              autoComplete="off"
              autoFocus
              mono
            />
          </Field>

          <Checkbox
            label="Remember on this device"
            checked={remember}
            onChange={(event) => setRemember(event.target.checked)}
          />

          {error && <Notice tone="destructive">{error}</Notice>}

          <Button type="submit" variant="primary" loading={submitting} className="w-full">
            {submitting ? "Signing in..." : "Sign in"}
          </Button>
        </form>
      </div>
    </div>
  );
}
