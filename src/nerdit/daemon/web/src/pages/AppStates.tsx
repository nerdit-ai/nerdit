import { Link } from "react-router-dom";
import { Button, Notice, PageHeader } from "../components/ui";
import { buttonClass } from "../lib/button";

// The app page's loading and failure states, shared by every route that lands
// on it (project, label, service and the /services shim). Separate file so
// ProjectPage and ProjectDetail can both import it without a cycle.

export function AppLoading({ name }: { name: string }) {
  return (
    <div className="mx-auto max-w-content space-y-6" aria-busy="true">
      <PageHeader title={name} />
      <p className="text-14 text-muted-foreground">Loading app…</p>
    </div>
  );
}

/** `retry` is omitted when retrying cannot help (a 404). */
export function AppError({
  name,
  message,
  retry,
  retrying
}: {
  name: string;
  message: string;
  retry?: () => void;
  retrying?: boolean;
}) {
  return (
    <div className="mx-auto max-w-content space-y-6">
      <PageHeader title={name} />
      <Notice tone="destructive" role="alert">{message}</Notice>
      <div className="flex gap-2">
        {retry && <Button onClick={retry} loading={retrying}>Try again</Button>}
        <Link to="/" className={buttonClass()}>Back to apps</Link>
      </div>
    </div>
  );
}
