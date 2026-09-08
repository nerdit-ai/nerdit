import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { Button, Mono } from "./ui";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * The top-level recoverable fallback (design guidelines §5): the dashboard must
 * not white-screen mid-incident. The message the app threw is shown — it is
 * often the daemon's own structured error — but never a stack trace: the
 * operator's next move is "reload" or "try again", not "read a trace".
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Kept out of the UI, kept in the console for whoever opens devtools.
    console.error("Dashboard crashed", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-8">
        <div className="w-full max-w-md text-center">
          <h1 className="text-24 font-semibold text-foreground">Something went wrong</h1>
          <p className="mt-1 text-14 text-muted-foreground">
            The page stopped rendering. Your apps keep running.
          </p>
          {error.message && (
            <Mono as="p" className="mt-4 break-words text-muted-foreground">
              {error.message}
            </Mono>
          )}
          <div className="mt-6 flex justify-center gap-2">
            <Button variant="secondary" onClick={() => location.reload()}>
              Reload
            </Button>
            <Button variant="ghost" onClick={() => this.setState({ error: null })}>
              Try again
            </Button>
          </div>
        </div>
      </div>
    );
  }
}
