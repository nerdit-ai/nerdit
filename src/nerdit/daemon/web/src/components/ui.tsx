/**
 * The dashboard's design system, in one file (design guidelines §5).
 *
 * Everything here is built from the console token set: no raw hex, no `dark:`
 * variants (dark mode is the token swap in `styles/globals.css`), no colour
 * opacity modifiers — the tokens are whole hex values, so `bg-primary/10`
 * silently drops the alpha. Where a translucent value is needed the token
 * carries it (`bg-overlay`); where a tinted surface is needed the `-subtle`
 * token is the answer.
 *
 * Radii: 6px controls (`rounded-button`), 8px panels (`rounded-card` /
 * `rounded-panel`), full pills. Type: the console scale (`text-12`/`13`/`14`/
 * `24`), labels via the `.label` class in globals.css, Mono 13 for URLs, ids
 * and commands. Icons: lucide at 16px, imported once here.
 *
 * A raw `<input>`, a bespoke modal or a page-local status-pill map elsewhere in
 * `src/` is a defect — it belongs in this file.
 */
import { Check, ChevronDown, Copy, Loader2, MoreHorizontal, X } from "lucide-react";
import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  InputHTMLAttributes,
  KeyboardEvent as ReactKeyboardEvent,
  ReactNode,
  SelectHTMLAttributes,
  TextareaHTMLAttributes
} from "react";
import { forwardRef, useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

/** Tiny class joiner — the codebase's own pattern, without a dependency. */
function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/* ------------------------------------------------------------------ Button */

export type ButtonVariant = "primary" | "secondary" | "ghost" | "destructive";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: "sm" | "md";
  loading?: boolean;
}

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-primary text-primary-foreground hover:bg-primary-hover",
  secondary: "border border-border bg-surface text-foreground hover:bg-surface-hover",
  ghost: "text-foreground hover:bg-surface-hover",
  destructive: "bg-destructive text-primary-foreground"
};

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  disabled,
  className,
  children,
  type = "button",
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      type={type}
      disabled={disabled || loading}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-button font-medium",
        "disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "h-8 px-3 text-12" : "h-9 px-4 text-14",
        BUTTON_VARIANTS[variant],
        className
      )}
    >
      {loading && <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------- Badge */

export type BadgeTone = "success" | "warning" | "destructive" | "muted" | "primary";

const BADGE_TONES: Record<BadgeTone, string> = {
  success: "bg-success-subtle text-success-foreground",
  warning: "bg-warning-subtle text-warning-foreground",
  // `destructive` has no `-foreground` token; its base value is the readable
  // pairing on `-subtle` in both palettes.
  destructive: "bg-destructive-subtle text-destructive",
  muted: "bg-surface-hover text-muted-foreground",
  primary: "bg-primary-subtle text-primary"
};

export function Badge({
  tone,
  className,
  children,
  ...rest
}: {
  tone: BadgeTone;
  className?: string;
  children: ReactNode;
} & HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      {...rest}
      className={cx(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-12 font-medium",
        BADGE_TONES[tone],
        className
      )}
    >
      {children}
    </span>
  );
}

/* ------------------------------------------------------------------- Panel */

export function Panel({
  title,
  actions,
  className,
  children,
  ...rest
}: {
  title?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
} & HTMLAttributes<HTMLElement>) {
  return (
    <section {...rest} className={cx("rounded-panel border border-border bg-surface", className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          {title ? <h2 className="label">{title}</h2> : <span />}
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

/** One label/value row. Meant to sit in a `divide-y divide-border` stack. */
export function DataRow({
  label,
  className,
  children,
  ...rest
}: {
  label: ReactNode;
  className?: string;
  children: ReactNode;
} & HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...rest}
      className={cx(
        "flex flex-col gap-1 px-4 py-3 sm:flex-row sm:items-baseline sm:justify-between sm:gap-6",
        className
      )}
    >
      <span className="label shrink-0">{label}</span>
      <div className="min-w-0 text-14 text-foreground sm:text-right">{children}</div>
    </div>
  );
}

/* -------------------------------------------------------------- PageHeader */

export function PageHeader({
  title,
  subtitle,
  actions
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-24 font-semibold text-foreground">{title}</h1>
        {subtitle && <p className="mt-1 text-14 text-muted-foreground">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}

/* ------------------------------------------------------------------ Notice */

export type NoticeTone = "info" | "warning" | "destructive" | "success";

/**
 * `success` and `warning` have no `-border` token in the palette; their base
 * value is the solid rule the guidelines ask for. Only `destructive` ships a
 * dedicated border token.
 */
const NOTICE_TONES: Record<NoticeTone, string> = {
  info: "border-border bg-surface text-foreground",
  warning: "border-warning bg-warning-subtle text-warning-foreground",
  destructive: "border-destructive-border bg-destructive-subtle text-destructive",
  success: "border-success bg-success-subtle text-success-foreground"
};

export function Notice({
  tone = "info",
  className,
  children,
  ...rest
}: {
  tone?: NoticeTone;
  className?: string;
  children: ReactNode;
} & HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...rest}
      className={cx("rounded-card border px-4 py-3 text-14", NOTICE_TONES[tone], className)}
    >
      {children}
    </div>
  );
}

/* -------------------------------------------------------------------- Mono */

/** Mono 13 — URLs, ids, commands, log lines. */
export function Mono({
  as: As = "span",
  className,
  children,
  ...rest
}: {
  as?: "span" | "div" | "code" | "p";
  className?: string;
  children: ReactNode;
} & HTMLAttributes<HTMLElement>) {
  return (
    <As {...rest} className={cx("font-mono text-13", className)}>
      {children}
    </As>
  );
}

/* ------------------------------------------------------------- Form fields */

const CONTROL_SKIN = cx(
  "w-full rounded-button border border-border bg-background px-3 text-foreground",
  "placeholder:text-subtle-foreground",
  "disabled:cursor-not-allowed disabled:bg-surface-hover disabled:text-muted-foreground"
);

/**
 * The size lives OUTSIDE `CONTROL_SKIN` because Tailwind emits the numeric
 * fontSize keys in order (`.text-13` before `.text-14`), so a caller composing
 * `text-13` over a skin-pinned `text-14` would silently lose. `mono` is the
 * guidelines' Mono-13 form control (secret refs, commands, paths, URLs).
 */
const controlType = (mono?: boolean) => (mono ? "font-mono text-13" : "text-14");

export const Input = forwardRef<
  HTMLInputElement,
  InputHTMLAttributes<HTMLInputElement> & { mono?: boolean }
>(function Input({ className, mono, ...rest }, ref) {
  return (
    <input ref={ref} {...rest} className={cx(CONTROL_SKIN, controlType(mono), "h-9", className)} />
  );
});

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  TextareaHTMLAttributes<HTMLTextAreaElement> & { mono?: boolean }
>(function Textarea({ className, mono, ...rest }, ref) {
  return (
    <textarea
      ref={ref}
      {...rest}
      className={cx(CONTROL_SKIN, controlType(mono), "py-2", className)}
    />
  );
});

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  function Select({ className, children, ...rest }, ref) {
    return (
      <div className="relative">
        <select
          ref={ref}
          {...rest}
          className={cx(CONTROL_SKIN, controlType(), "h-9 appearance-none pr-9", className)}
        >
          {children}
        </select>
        <ChevronDown
          aria-hidden="true"
          className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
        />
      </div>
    );
  }
);

export function Field({
  label,
  hint,
  error,
  htmlFor,
  children,
  className,
  ...rest
}: {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
} & HTMLAttributes<HTMLDivElement>) {
  return (
    <div {...rest} className={cx("flex flex-col gap-1.5", className)}>
      <label className="label" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
      {hint && !error && <p className="text-12 text-muted-foreground">{hint}</p>}
      {error && <p className="text-12 text-destructive">{error}</p>}
    </div>
  );
}

/**
 * A real `<input type="checkbox">` restyled on the tokens — `accent-primary`
 * paints the native check mark, so the control keeps its own keyboard
 * behaviour, its indeterminate state and every assistive-tech affordance
 * instead of a `div` pretending to be a checkbox.
 */
export const Checkbox = forwardRef<
  HTMLInputElement,
  InputHTMLAttributes<HTMLInputElement> & { label: ReactNode }
>(function Checkbox({ label, className, ...rest }, ref) {
  return (
    <label className={cx("inline-flex items-center gap-2 text-14 text-foreground", className)}>
      <input
        ref={ref}
        type="checkbox"
        {...rest}
        className="h-4 w-4 shrink-0 rounded-[4px] border border-border bg-background accent-primary disabled:cursor-not-allowed disabled:opacity-50"
      />
      {label}
    </label>
  );
});

/* ------------------------------------------------------------------ Dialog */

/**
 * THE modal shell. Every dialog in the dashboard is this component with
 * different children — a bespoke overlay elsewhere in `src/` is a defect.
 *
 * It owns exactly the three things a modal must not get wrong twice: the
 * overlay click, the Escape key (bound on `window`, so a keydown anywhere in
 * the document reaches it) and the `role="dialog"` / `aria-modal` /
 * `aria-labelledby` wiring to the title. Everything else — buttons, fields,
 * busy state — belongs to the caller.
 */
export function Dialog({
  open,
  onClose,
  title,
  footer,
  width = "md",
  className,
  children,
  ...rest
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  footer?: ReactNode;
  width?: "md" | "lg";
  className?: string;
  children?: ReactNode;
} & Omit<HTMLAttributes<HTMLDivElement>, "title">) {
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-overlay p-4 backdrop-blur"
      onClick={onClose}
    >
      <div
        {...rest}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
        className={cx(
          "w-full rounded-panel border border-border bg-surface p-6 shadow-lg",
          width === "lg" ? "max-w-2xl" : "max-w-md",
          className
        )}
      >
        <h2 id={titleId} className="text-16 font-semibold text-foreground">
          {title}
        </h2>
        {children}
        {footer && <div className="mt-6 flex justify-end gap-2">{footer}</div>}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------- Menu */

export type MenuItem =
  | "separator"
  | {
      label: ReactNode;
      onSelect: () => void;
      destructive?: boolean;
      disabled?: boolean;
    };

/**
 * The popover renders through a portal on `document.body`, positioned `fixed`
 * from the trigger's own rect. It has to: a `…` menu opened from the last row
 * of a `overflow-hidden` panel, or from inside a table with its own stacking
 * context, was clipped by its ancestor no matter how high its `z-index` went.
 * Escaping to the body is the only fix that does not ask every caller to
 * remember what its container clips.
 *
 * Consequences the API deliberately hides: the popover is no longer a DOM
 * descendant of the trigger, so the outside-click check has to consult BOTH
 * nodes (otherwise mousedown inside the menu closes it before the item's click
 * lands, and selecting anything becomes impossible), and a scroll or resize
 * would leave the popover behind its trigger — so both simply close it, which
 * is what every native menu does anyway.
 */
export function Menu({
  label,
  items,
  align = "end",
  triggerVariant = "ghost",
  triggerSize = "sm"
}: {
  label?: ReactNode;
  items: MenuItem[];
  align?: "start" | "end";
  /** A menu that IS a page's primary action passes "primary" here. */
  triggerVariant?: ButtonVariant;
  triggerSize?: "sm" | "md";
}) {
  const [open, setOpen] = useState(false);
  const [rect, setRect] = useState<{ top: number; left: number } | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Measure before paint so the menu never flashes at the top-left corner.
  useLayoutEffect(() => {
    if (!open) {
      setRect(null);
      return;
    }
    const trigger = rootRef.current?.getBoundingClientRect();
    if (!trigger) return;
    const width = listRef.current?.offsetWidth ?? 176; // min-w-[11rem]
    setRect({
      top: trigger.bottom + 4,
      left: align === "end" ? trigger.right - width : trigger.left
    });
  }, [open, align]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    const onPointer = (event: MouseEvent) => {
      const target = event.target as Node;
      if (rootRef.current?.contains(target) || listRef.current?.contains(target)) return;
      setOpen(false);
    };
    const onReflow = () => setOpen(false);
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onPointer);
    window.addEventListener("resize", onReflow);
    window.addEventListener("scroll", onReflow, true);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onPointer);
      window.removeEventListener("resize", onReflow);
      window.removeEventListener("scroll", onReflow, true);
    };
  }, [open]);

  /** Arrow keys walk the enabled item buttons; nothing fancier is warranted. */
  function onListKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const buttons = Array.from(
      listRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? []
    );
    if (buttons.length === 0) return;
    const at = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const next =
      event.key === "ArrowDown"
        ? buttons[(at + 1) % buttons.length]
        : buttons[(at <= 0 ? buttons.length : at) - 1];
    next.focus();
  }

  return (
    <div ref={rootRef} className="relative inline-block">
      <Button
        variant={triggerVariant}
        size={triggerSize}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label ? undefined : "More actions"}
        onClick={() => setOpen((v) => !v)}
      >
        {label ?? <MoreHorizontal aria-hidden="true" className="h-4 w-4" />}
      </Button>
      {open &&
        createPortal(
          <div
            ref={listRef}
            role="menu"
            onKeyDown={onListKeyDown}
            style={{ top: rect?.top ?? 0, left: rect?.left ?? 0, visibility: rect ? "visible" : "hidden" }}
            className="fixed z-50 min-w-[11rem] rounded-card border border-border bg-surface py-1 shadow-lg"
          >
            {items.map((item, index) =>
            item === "separator" ? (
              <div key={`sep-${index}`} className="my-1 h-px bg-border" role="separator" />
            ) : (
              <button
                key={index}
                type="button"
                role="menuitem"
                disabled={item.disabled}
                onClick={() => {
                  setOpen(false);
                  item.onSelect();
                }}
                className={cx(
                  "block w-full px-3 py-2 text-left text-14 hover:bg-surface-hover",
                  "disabled:cursor-not-allowed disabled:opacity-50",
                  item.destructive ? "text-destructive" : "text-foreground"
                )}
              >
                {item.label}
              </button>
            )
            )}
          </div>,
          document.body
        )}
    </div>
  );
}

/* --------------------------------------------------------------- CopyField */

export function CopyField({
  value,
  mono = true,
  className,
  ...rest
}: {
  value: string;
  mono?: boolean;
  className?: string;
} & HTMLAttributes<HTMLDivElement>) {
  const [copied, setCopied] = useState(false);

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div
      {...rest}
      className={cx(
        "inline-flex max-w-full items-center gap-2 rounded-button border border-border bg-surface py-1 pl-3 pr-1",
        className
      )}
    >
      <span className={cx("truncate text-foreground", mono ? "font-mono text-13" : "text-14")}>
        {value}
      </span>
      <Button
        variant="ghost"
        size="sm"
        onClick={onCopy}
        aria-label="Copy"
        className="shrink-0 px-2 text-muted-foreground"
      >
        {copied ? (
          <>
            <Check aria-hidden="true" className="h-4 w-4" />
            Copied
          </>
        ) : (
          <Copy aria-hidden="true" className="h-4 w-4" />
        )}
      </Button>
    </div>
  );
}

/* ---------------------------------------------------------------- TableRow */

export function TableRow({
  onClick,
  className,
  children,
  ...rest
}: {
  onClick?: () => void;
  className?: string;
  children: ReactNode;
} & Omit<HTMLAttributes<HTMLTableRowElement>, "onClick">) {
  return (
    <tr
      {...rest}
      onClick={onClick}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (event) => {
              // Row-level activation belongs to the ROW. A descendant that is
              // focusable in its own right (the address link, a button) keeps
              // its own Enter/Space: without this guard the row swallowed the
              // key and navigated to the app page instead of following the
              // link the operator had actually focused.
              if (event.target !== event.currentTarget) return;
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onClick();
              }
            }
          : undefined
      }
      className={cx(
        "border-b border-border",
        onClick && "cursor-pointer hover:bg-surface-hover",
        className
      )}
    >
      {children}
    </tr>
  );
}

/* ----------------------------------------------------------------- Confirm */

export function Confirm({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  busy = false,
  confirmPhrase,
  onConfirm,
  onCancel
}: {
  open: boolean;
  title: ReactNode;
  description?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  busy?: boolean;
  /** When set, the operator must type this string exactly to enable confirm. */
  confirmPhrase?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const [typed, setTyped] = useState("");
  // A busy dialog must not vanish under the operator mid-request: the guard
  // covers BOTH dismissal paths, because `Dialog` routes the overlay click and
  // Escape through this one callback.
  const cancel = useCallback(() => {
    if (!busy) onCancel();
  }, [busy, onCancel]);

  useEffect(() => {
    if (!open) setTyped("");
  }, [open]);

  const blocked = confirmPhrase != null && typed !== confirmPhrase;

  return (
    <Dialog
      open={open}
      onClose={cancel}
      title={title}
      footer={
        <>
          <Button variant="secondary" disabled={busy} onClick={cancel}>
            {cancelLabel}
          </Button>
          <Button
            variant={destructive ? "destructive" : "primary"}
            loading={busy}
            disabled={busy || blocked}
            onClick={onConfirm}
          >
            {confirmLabel}
          </Button>
        </>
      }
    >
      {description && <div className="mt-3 text-14 text-muted-foreground">{description}</div>}
      {confirmPhrase != null && (
        <div className="mt-4">
          <Field label={<>Type {confirmPhrase} to confirm</>}>
            <Input
              data-testid="confirm-phrase-input"
              value={typed}
              autoFocus
              spellCheck={false}
              autoComplete="off"
              aria-label={`Type ${confirmPhrase} to confirm`}
              onChange={(event) => setTyped(event.target.value)}
            />
          </Field>
        </div>
      )}
    </Dialog>
  );
}

/* ------------------------------------------------------------------- Meter */

/**
 * A usage bar: GPU memory, disk, quota (design guidelines §5/§6).
 *
 * The tone is derived, not passed, so two pages cannot disagree about when a
 * disk is "getting full": under 80 % it is primary (a fact), 80–95 % warning,
 * above 95 % destructive. No animation — a bar that slides on every 5 s poll
 * reads as activity where there is none.
 */
export function Meter({
  label,
  value,
  max,
  hint,
  className
}: {
  label?: ReactNode;
  value: number;
  max: number;
  hint?: ReactNode;
  className?: string;
}) {
  // A zero or nonsense `max` means "no denominator", not "full": show empty.
  const ratio = max > 0 && Number.isFinite(value / max) ? Math.max(0, Math.min(1, value / max)) : 0;
  const fill = ratio > 0.95 ? "bg-destructive" : ratio >= 0.8 ? "bg-warning" : "bg-primary";

  return (
    <div className={cx("flex flex-col gap-1.5", className)}>
      {(label || hint) && (
        <div className="flex items-baseline justify-between gap-3">
          {label ? <span className="label">{label}</span> : <span />}
          {hint && <span className="text-12 text-muted-foreground">{hint}</span>}
        </div>
      )}
      <div
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        className="h-1.5 w-full overflow-hidden rounded-full bg-surface-hover"
      >
        <div
          data-testid="meter-fill"
          className={cx("h-full rounded-full", fill)}
          style={{ width: `${ratio * 100}%` }}
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ Banner */

export type BannerTone = "warning" | "destructive" | "info";

const BANNER_TONES: Record<BannerTone, string> = {
  info: "border-border bg-surface text-foreground",
  warning: "border-warning bg-warning-subtle text-warning-foreground",
  destructive: "border-destructive-border bg-destructive-subtle text-destructive"
};

export function Banner({
  tone = "warning",
  actions,
  onDismiss,
  className,
  children
}: {
  tone?: BannerTone;
  actions?: ReactNode;
  onDismiss?: () => void;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={cx(
        "flex w-full flex-wrap items-center justify-between gap-3 border-b px-4 py-2 text-14",
        BANNER_TONES[tone],
        className
      )}
    >
      <div className="min-w-0">{children}</div>
      <div className="flex shrink-0 items-center gap-2">
        {actions}
        {onDismiss && (
          <Button variant="ghost" size="sm" aria-label="Dismiss" onClick={onDismiss}>
            <X aria-hidden="true" className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
}
