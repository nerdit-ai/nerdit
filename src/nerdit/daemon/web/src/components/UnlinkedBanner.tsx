import { Unlink } from "lucide-react";
import { useCapabilities } from "../api/queries";
import { Banner, Mono } from "./ui";

/**
 * (P34 D4) One quiet line telling the operator this node has no Nerdit account.
 *
 * Why it exists: after X16 the account is free and `install.sh` drives the link,
 * so an unlinked node stops being the normal case and becomes something worth
 * naming once, where the operator already is. The dashboard is the surface a
 * human looks at; `nerdit doctor`'s `link` row is the same fact for a terminal.
 *
 * Why the second sentence is not optional: an unlinked daemon is FULLY
 * functional — it deploys, proxies, reconciles, serves models and databases,
 * forever, and this programme adds no runtime gate anywhere (D-X16-O15,
 * D-X16-57, D-ENT-2). A banner that let a reader infer their local plane is
 * degraded would be a lie the code does not tell, so the copy states what is
 * off (the cloud half) and what is not (everything else) in that order. The
 * redesign adds the guidelines' §4 clause — "linking is optional" — which says
 * the same thing about the operator's choice rather than about the machine.
 *
 * Deliberately NOT dismissable (OD-P34-4 open question 4): it is one
 * warning-tone line whose whole job is to be seen, and nothing here has ever
 * persisted a dismissal — so `Banner`'s `onDismiss` is deliberately unused.
 *
 * Deliberately says nothing about GitHub deploys, though the plan's draft copy
 * did: the account-gated GitHub App path is P33 and is not on this branch, and
 * naming an unshipped feature in shipped UI is a promise (the §6 Q5 rule).
 */
export function UnlinkedBanner() {
  const capabilities = useCapabilities();

  // Never guess from a query in flight: `data` is `undefined` while loading and
  // on error alike, and rendering "not linked" off that would flash the banner
  // on every cold load of a perfectly well-linked node (failure mode (a)).
  if (!capabilities.isSuccess) return null;

  // Precedence against `DaemonStatusBanner` is NOT decided here any more: the
  // shell owns one banner slot with an ordered list (daemon offline wins), so
  // this component no longer reads the daemon status at all. The reasoning is
  // unchanged and now lives in `AppShell`'s `BannerSlot`: a cached
  // capabilities response outlives the daemon going away, and "nerditd is
  // unreachable" is both more urgent and the reason this node's link state
  // cannot be trusted at that moment.

  const link = capabilities.data.link;

  // Enrolment, not connection health, and not the feature flag either.
  //
  // A linked node whose tunnel is merely reconnecting reports `enabled: true`
  // with a `backoff`/`connecting` state and must NOT be told it is unlinked
  // (failure mode (b)) — that episode belongs to the doctor `link` row and to
  // `DaemonStatusBanner`.
  //
  // `enabled` alone is NOT the discriminator, which is failure mode (c): the
  // daemon reports no manager both for a node that never linked and for one
  // that linked and then set `[link].enabled = false` — a deliberate opt-out,
  // and also the state a node lands in when its key becomes unreadable. Reading
  // the flag as enrolment put a permanent, non-dismissable warning on top of
  // both. So this reads `node_id`, exactly as the doctor `link` check does: the
  // claim staging tail writes it, `enabled = false` never clears it, and only
  // `nerdit unlink` does — which makes it the enrolment fact and the flag a
  // statement about the tunnel.
  const unlinked = !link || (!link.node_id && !link.slug);
  if (!unlinked) return null;

  return (
    // In normal flow, NOT `fixed` — the one place this deliberately differs
    // from `DaemonStatusBanner`.
    //
    // That banner is episodic and urgent: the daemon is unreachable right now,
    // nothing else on the page is usable, and floating over the chrome for a
    // few seconds is the point. Being unlinked is neither — it is a persistent
    // state in which the whole local plane keeps working, so the banner has to
    // coexist with the chrome rather than sit on top of it.
    //
    // Sitting on top of it was a real bug, not a nicety: the mobile header is
    // `sticky top-0 z-30 h-[72px]` and the nav trigger is centred in it, so a
    // fixed z-50 strip covered the trigger's click point and mobile navigation
    // became unreachable on every unlinked node.
    <div role="status" aria-live="polite" data-testid="unlinked-banner">
      <Banner tone="warning">
        <span className="flex items-center gap-2 font-medium">
          <Unlink aria-hidden="true" className="h-4 w-4 shrink-0" />
          <span>
            {/* No em dash: dashboard copy avoids them by repo convention,
                pinned for every page by tests/e2e/07-audit.spec.ts. This banner
                renders on top of every route, so it is subject to that rule
                everywhere. */}
            This node is not linked to a Nerdit account, so remote access and hosted shares are
            off. Linking is optional: everything local works without it. Run{" "}
            <Mono>nerdit link --device</Mono> (free).
          </span>
        </span>
      </Banner>
    </div>
  );
}
