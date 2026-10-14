import { useEffect, useState, type ReactNode } from "react";
import { Navigate } from "react-router-dom";
import {
  authErrorCode,
  clearStoredToken,
  CLOUD_MACHINES_URL,
  getStoredToken,
  SESSION_ENDED_EVENT,
  setSessionMode,
  type AuthSession
} from "../lib/auth";
import { buttonClass } from "../lib/button";
import { Button } from "./ui";

function SessionRecovery({ title, retry, cloud }: { title: string; retry: () => void; cloud: boolean }) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-6">
      <div role="alert" className="w-full max-w-sm space-y-4">
        <h1 className="text-20 font-semibold">{title}</h1>
        <p className="text-14 text-muted-foreground">
          {cloud ? "Reopen this machine from Nerdit Cloud." : "This machine did not answer."}
        </p>
        <div className="flex gap-2">
          {cloud && <a href={CLOUD_MACHINES_URL} className={buttonClass("primary")}>Open Nerdit Cloud</a>}
          <Button onClick={retry}>Try again</Button>
        </div>
      </div>
    </div>
  );
}

/** The proxy can authenticate with an HttpOnly cookie, without a local token. */
export function AuthGate({ children, login = false }: { children: ReactNode; login?: boolean }) {
  const [state, setState] = useState<"checking" | "ready" | "login" | "expired" | "error">("checking");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const expired = () => setState("expired");
    window.addEventListener(SESSION_ENDED_EVENT, expired);
    return () => window.removeEventListener(SESSION_ENDED_EVENT, expired);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const token = getStoredToken();
    setState("checking");
    void fetch("/api/auth/check", {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      signal: controller.signal,
      cache: "no-store"
    }).then(async (response) => {
      const body = await response.json().catch(() => null);
      if (controller.signal.aborted) return;
      if (response.ok && body?.ok === true && typeof body.role === "string") {
        setSessionMode((body as AuthSession).mode);
        setState("ready");
      } else if ([401, 403].includes(response.status)) {
        clearStoredToken();
        const code = authErrorCode(body);
        setState(code?.startsWith("proxy_session_") ? "expired" : body?.error ? "error" : "login");
      } else {
        setState("error");
      }
    }).catch(() => {
      if (!controller.signal.aborted) setState("error");
    });
    return () => controller.abort();
  }, [attempt, login]);

  if (state === "checking") return <div role="status" className="p-6 text-muted-foreground">Connecting…</div>;
  const retry = () => setAttempt((value) => value + 1);
  if (state === "expired") return <SessionRecovery title="Session ended" retry={retry} cloud />;
  if (state === "error") {
    // The session mode is unknown here: only a page served through the relay offers Cloud.
    const cloud = window.location.hostname.endsWith(".nodes.nerdit.ai");
    return <SessionRecovery title="Can't reach this machine" retry={retry} cloud={cloud} />;
  }
  if (state === "login") return login ? <>{children}</> : <Navigate to="/login" replace />;
  return login ? <Navigate to="/" replace /> : <>{children}</>;
}
