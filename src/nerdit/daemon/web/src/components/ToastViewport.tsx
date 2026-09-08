import { X } from "lucide-react";
import { useToastStore } from "../state/toastStore";

/**
 * Tone skins on the console token set. The live region is rendered even when
 * empty: a screen reader only announces insertions into a region that already
 * exists, so returning `null` on an empty stack silences the first toast.
 */
const KIND_STYLES: Record<string, string> = {
  success: "border-primary bg-primary-subtle text-foreground",
  error: "border-destructive-border bg-destructive-subtle text-foreground",
  info: "border-border bg-surface text-foreground"
};

export function ToastViewport() {
  const toasts = useToastStore((s) => s.toasts);
  const dismiss = useToastStore((s) => s.dismiss);
  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed bottom-6 right-6 z-50 flex w-80 flex-col gap-2"
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          role="status"
          onClick={() => dismiss(t.id)}
          className={`pointer-events-auto flex items-start gap-2 rounded-card border px-4 py-3 text-left text-14 shadow-lg ${KIND_STYLES[t.kind] ?? KIND_STYLES.info}`}
        >
          <span className="min-w-0 flex-1">{t.message}</span>
          {t.kind === "error" && (
            <button
              type="button"
              aria-label="Dismiss"
              onClick={() => dismiss(t.id)}
              className="shrink-0 rounded-button p-0.5 text-muted-foreground hover:bg-surface-hover"
            >
              <X aria-hidden="true" className="h-4 w-4" />
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
