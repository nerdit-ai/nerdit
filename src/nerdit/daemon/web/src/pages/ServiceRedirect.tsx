import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { useService } from "../api/queries";
import { kindHomePath } from "../lib/projects";
import { toast } from "../state/toastStore";

/**
 * Kind-aware redirect for the retired `/services/:ident` route (D8). That URL
 * historically served services, models and databases alike, so this fetches the
 * row once and routes by kind: a service to its project page (by NAME — the URL
 * may carry an id), a model to Models, a database to Databases; anything else
 * falls back to the project grid.
 * Only a 404 lands on the project grid with a toast — any other error (daemon
 * blip, 5xx) renders honestly instead of claiming the service is gone.
 */
export default function ServiceRedirect() {
  const { ident = "" } = useParams();
  const navigate = useNavigate();
  const service = useService(ident);

  const notFound =
    service.isError && service.error instanceof ApiError && service.error.status === 404;

  useEffect(() => {
    if (notFound) {
      navigate("/", { replace: true });
      toast("error", "App not found");
      return;
    }
    if (service.data) {
      // A service goes to its LABEL URL: that route moves on to the project
      // shape only when the caller can read the project (a label-scoped token cannot).
      const svc = service.data;
      const to =
        svc.kind === "service" ? `/projects/${encodeURIComponent(svc.name)}` : kindHomePath(svc);
      navigate(to, { replace: true });
    }
  }, [notFound, service.data, navigate]);

  if (service.isError && !notFound) {
    return <p className="text-destructive">{(service.error as Error).message}</p>;
  }
  return <p className="text-muted-foreground">Loading app…</p>;
}
