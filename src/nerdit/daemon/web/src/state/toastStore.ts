import { create } from "zustand";

export type ToastKind = "success" | "error" | "info";

interface Toast {
  id: string;
  kind: ToastKind;
  message: string;
  /** `null` when the toast never expires on its own (errors). */
  expiresAt: number | null;
}

interface ToastState {
  toasts: Toast[];
  push: (kind: ToastKind, message: string, ttlMs?: number) => string;
  dismiss: (id: string) => void;
}

const TTL_DEFAULT = 4000;

export const useToastStore = create<ToastState>((set, get) => ({
  toasts: [],
  push: (kind, message, ttlMs) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    // Design guidelines §5: an error persists until the operator dismisses it —
    // a failed mutation is the one thing on this surface that must not scroll
    // past unread. Success and info keep the 4s TTL, and an explicit `ttlMs`
    // still wins for every kind, so existing call sites are unaffected.
    const ttl = ttlMs ?? (kind === "error" ? null : TTL_DEFAULT);
    const expiresAt = ttl == null ? null : Date.now() + ttl;
    set((s) => ({ toasts: [...s.toasts, { id, kind, message, expiresAt }] }));
    if (ttl != null) setTimeout(() => get().dismiss(id), ttl);
    return id;
  },
  dismiss: (id) =>
    set((s) => ({ toasts: s.toasts.filter((toast) => toast.id !== id) }))
}));

export function toast(kind: ToastKind, message: string, ttlMs?: number) {
  return useToastStore.getState().push(kind, message, ttlMs);
}
