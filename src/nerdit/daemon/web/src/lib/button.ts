// The Button look, shared by `<Button>` and links styled as actions. Lives
// outside components/ui.tsx so that file keeps exporting only components.

export type ButtonVariant = "primary" | "secondary" | "ghost" | "destructive";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-primary text-primary-foreground hover:bg-primary-hover",
  secondary: "border border-border bg-surface text-foreground hover:bg-surface-hover",
  ghost: "text-foreground hover:bg-surface-hover",
  destructive: "bg-destructive text-primary-foreground"
};

export function buttonClass(variant: ButtonVariant = "secondary", size: "sm" | "md" = "md"): string {
  return [
    "inline-flex items-center justify-center gap-2 rounded-button font-medium",
    "disabled:cursor-not-allowed disabled:opacity-50",
    size === "sm" ? "h-8 px-3 text-12" : "h-9 px-4 text-14",
    BUTTON_VARIANTS[variant]
  ].join(" ");
}
