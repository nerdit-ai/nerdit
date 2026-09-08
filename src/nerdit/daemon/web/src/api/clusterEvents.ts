import { useEffect } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { streamEvents } from "../lib/sse";

/**
 * Subscribe to /api/events/stream and invalidate the services/models caches on
 * every `job.status_changed` event (the ServiceController emits it for service
 * and model transitions).
 *
 * Rewritten on the authenticated fetch-stream (P12 D2): unlike the old raw
 * `EventSource`, this sends the bearer, so the stream actually authenticates
 * and the 401 flood disappears by construction. On 401/403 it stops
 * permanently (no retry); on transport errors it backs off up to 30 s. Polling
 * still drives baseline freshness, so the stream is only a booster.
 */
export function useClusterEvents(queryClient: QueryClient): void {
  useEffect(() => {
    const controller = new AbortController();

    void streamEvents("/events/stream", {
      signal: controller.signal,
      onEvent: (event) => {
        if (event.event !== "job.status_changed") return;
        try {
          JSON.parse(event.data);
        } catch {
          return; // ignore malformed payloads
        }
        void queryClient.invalidateQueries({ queryKey: ["services"] });
        void queryClient.invalidateQueries({ queryKey: ["models"] });
      }
    });

    return () => controller.abort();
  }, [queryClient]);
}
