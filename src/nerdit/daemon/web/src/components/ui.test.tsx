// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Checkbox, Confirm, Dialog, Menu, Meter, TableRow } from "./ui";
import { ErrorBoundary } from "./ErrorBoundary";
import { toast, useToastStore } from "../state/toastStore";

/**
 * The primitives are mostly skin, and skin is not worth pinning. What is worth
 * pinning is the behaviour a page can get wrong on the operator's behalf: the
 * typed confirm that guards a destructive purge, the `…` menu's dismissal
 * paths, whether an error toast can scroll away unread, and whether a thrown
 * render leaves the dashboard white mid-incident.
 */
afterEach(() => {
  cleanup();
  useToastStore.setState({ toasts: [] });
  vi.useRealTimers();
});

describe("Confirm", () => {
  const props = {
    open: true,
    title: "Delete app",
    confirmLabel: "Delete",
    onConfirm: vi.fn(),
    onCancel: vi.fn()
  };

  beforeEach(() => {
    props.onConfirm.mockReset();
    props.onCancel.mockReset();
  });

  it("keeps confirm disabled until the phrase matches exactly", () => {
    render(<Confirm {...props} confirmPhrase="paperless" destructive />);
    const confirm = screen.getByRole("button", { name: "Delete" });
    expect((confirm as HTMLButtonElement).disabled).toBe(true);

    const input = screen.getByTestId("confirm-phrase-input");
    // A prefix is not a match, and neither is a case variation — the phrase is
    // the app's own name, and near-misses are exactly what the gate is for.
    fireEvent.change(input, { target: { value: "paper" } });
    expect((confirm as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(input, { target: { value: "Paperless" } });
    expect((confirm as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(input, { target: { value: "paperless" } });
    expect((confirm as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(confirm);
    expect(props.onConfirm).toHaveBeenCalledTimes(1);
  });

  it("enables confirm immediately when no phrase is required", () => {
    render(<Confirm {...props} />);
    expect(screen.queryByTestId("confirm-phrase-input")).toBeNull();
    expect((screen.getByRole("button", { name: "Delete" }) as HTMLButtonElement).disabled).toBe(
      false
    );
  });

  it("cancels on Escape and disables both buttons while busy", () => {
    const { rerender } = render(<Confirm {...props} />);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(props.onCancel).toHaveBeenCalledTimes(1);

    rerender(<Confirm {...props} busy />);
    expect((screen.getByRole("button", { name: "Delete" }) as HTMLButtonElement).disabled).toBe(
      true
    );
    expect((screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement).disabled).toBe(
      true
    );
    // A busy dialog must not vanish under the operator mid-request.
    fireEvent.keyDown(window, { key: "Escape" });
    expect(props.onCancel).toHaveBeenCalledTimes(1);
  });

  it("renders nothing when closed", () => {
    render(<Confirm {...props} open={false} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("Checkbox", () => {
  it("is a real checkbox, labelled and toggleable", () => {
    const onChange = vi.fn();
    render(<Checkbox label="Remember on this device" checked={false} onChange={onChange} />);

    const box = screen.getByLabelText("Remember on this device") as HTMLInputElement;
    expect(box.type).toBe("checkbox");
    // Clicking the LABEL text is the affordance a fake checkbox loses; a real
    // input keeps it for free.
    fireEvent.click(screen.getByText("Remember on this device"));
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("toggles when uncontrolled, so the native behaviour is intact", () => {
    render(<Checkbox label="Follow logs" defaultChecked={false} />);
    const box = screen.getByLabelText("Follow logs") as HTMLInputElement;
    fireEvent.click(box);
    expect(box.checked).toBe(true);
  });

  it("reflects the checked prop and honours disabled", () => {
    const { rerender } = render(<Checkbox label="Wired" checked readOnly />);
    expect((screen.getByLabelText("Wired") as HTMLInputElement).checked).toBe(true);

    rerender(<Checkbox label="Wired" checked readOnly disabled />);
    expect((screen.getByLabelText("Wired") as HTMLInputElement).disabled).toBe(true);
  });
});

describe("Dialog", () => {
  const props = { open: true, title: "Deploy app", onClose: vi.fn() };

  beforeEach(() => props.onClose.mockReset());

  it("wires role, aria-modal and aria-labelledby to its own title", () => {
    render(
      <Dialog {...props}>
        <p>body</p>
      </Dialog>
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    const heading = screen.getByText("Deploy app");
    expect(dialog.getAttribute("aria-labelledby")).toBe(heading.id);
    expect(heading.id).toBeTruthy();
  });

  it("closes on Escape and on an overlay click, but not on a click inside", () => {
    render(<Dialog {...props}>body</Dialog>);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(props.onClose).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("dialog"));
    expect(props.onClose).toHaveBeenCalledTimes(1);

    // The overlay is the dialog panel's parent element.
    fireEvent.click(screen.getByRole("dialog").parentElement as HTMLElement);
    expect(props.onClose).toHaveBeenCalledTimes(2);
  });

  it("renders nothing when closed, and unbinds Escape with it", () => {
    render(
      <Dialog {...props} open={false}>
        body
      </Dialog>
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(props.onClose).not.toHaveBeenCalled();
  });

  it("renders a footer and widens on request", () => {
    render(
      <Dialog {...props} width="lg" footer={<button type="button">Deploy</button>}>
        body
      </Dialog>
    );
    expect(screen.getByRole("button", { name: "Deploy" })).toBeTruthy();
    expect(screen.getByRole("dialog").className).toContain("max-w-2xl");
  });
});

describe("Menu", () => {
  it("opens, selects an item and closes", () => {
    const onSelect = vi.fn();
    render(<Menu items={[{ label: "Restart", onSelect }]} />);

    expect(screen.queryByRole("menu")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "More actions" }));
    expect(screen.getByRole("menu")).toBeTruthy();

    fireEvent.click(screen.getByRole("menuitem", { name: "Restart" }));
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("closes on Escape without selecting", () => {
    const onSelect = vi.fn();
    render(<Menu items={[{ label: "Restart", onSelect }]} />);
    fireEvent.click(screen.getByRole("button", { name: "More actions" }));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("menu")).toBeNull();
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("closes on an outside click", () => {
    render(<Menu items={[{ label: "Restart", onSelect: vi.fn() }]} />);
    fireEvent.click(screen.getByRole("button", { name: "More actions" }));
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("skips disabled items when arrow-keying and never fires them", () => {
    const enabled = vi.fn();
    const disabled = vi.fn();
    render(
      <Menu
        items={[
          { label: "Stop", onSelect: disabled, disabled: true },
          "separator",
          { label: "Delete", onSelect: enabled, destructive: true }
        ]}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: "More actions" }));
    fireEvent.keyDown(screen.getByRole("menu"), { key: "ArrowDown" });
    expect(document.activeElement?.textContent).toBe("Delete");

    fireEvent.click(screen.getByRole("menuitem", { name: "Stop" }));
    expect(disabled).not.toHaveBeenCalled();
  });

  it("renders the popover on the body so a clipping ancestor cannot hide it", () => {
    // The reason the portal exists: a `…` opened from inside an
    // `overflow-hidden` panel used to be clipped whatever its z-index.
    const { container } = render(
      <div style={{ overflow: "hidden" }}>
        <Menu items={[{ label: "Restart", onSelect: vi.fn() }]} />
      </div>
    );
    fireEvent.click(screen.getByRole("button", { name: "More actions" }));

    const menu = screen.getByRole("menu");
    expect(container.contains(menu)).toBe(false);
    expect(document.body.contains(menu)).toBe(true);
  });

  it("closes on a scroll rather than floating away from its trigger", () => {
    render(<Menu items={[{ label: "Restart", onSelect: vi.fn() }]} />);
    fireEvent.click(screen.getByRole("button", { name: "More actions" }));
    expect(screen.getByRole("menu")).toBeTruthy();

    fireEvent.scroll(window);
    expect(screen.queryByRole("menu")).toBeNull();
  });
});

describe("Meter", () => {
  const fill = () => screen.getByTestId("meter-fill");

  it("renders the fill as a clamped percentage of max", () => {
    render(<Meter label="Disk" value={25} max={100} hint="25 GB of 100 GB" />);
    expect(fill().style.width).toBe("25%");
    expect(screen.getByText("Disk")).toBeTruthy();
    expect(screen.getByText("25 GB of 100 GB")).toBeTruthy();
  });

  it("clamps out-of-range and undefined denominators instead of overflowing", () => {
    const { rerender } = render(<Meter value={150} max={100} />);
    expect(fill().style.width).toBe("100%");

    rerender(<Meter value={-5} max={100} />);
    expect(fill().style.width).toBe("0%");

    // No denominator is "unknown", never "full".
    rerender(<Meter value={40} max={0} />);
    expect(fill().style.width).toBe("0%");
  });

  it("derives its tone from the thresholds, so no page can disagree", () => {
    const { rerender } = render(<Meter value={79} max={100} />);
    expect(fill().className).toContain("bg-primary");

    rerender(<Meter value={80} max={100} />);
    expect(fill().className).toContain("bg-warning");

    rerender(<Meter value={95} max={100} />);
    expect(fill().className).toContain("bg-warning");

    rerender(<Meter value={96} max={100} />);
    expect(fill().className).toContain("bg-destructive");
  });
});

describe("toastStore", () => {
  it("keeps an error toast until it is dismissed", () => {
    vi.useFakeTimers();
    const id = toast("error", "Deploy failed");
    vi.advanceTimersByTime(60_000);
    expect(useToastStore.getState().toasts).toHaveLength(1);

    useToastStore.getState().dismiss(id);
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it("expires success and info toasts after the default TTL", () => {
    vi.useFakeTimers();
    toast("success", "Deployed");
    toast("info", "Reconciling");
    expect(useToastStore.getState().toasts).toHaveLength(2);
    vi.advanceTimersByTime(4000);
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it("honours an explicit ttl even for an error", () => {
    // Source compatibility: `toast(kind, message, ttlMs)` still wins outright.
    vi.useFakeTimers();
    toast("error", "Transient", 1000);
    vi.advanceTimersByTime(1000);
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });
});

describe("ErrorBoundary", () => {
  // The child decides at ITS OWN render time, which is what "Try again" means:
  // the boundary re-renders the same children and the underlying condition —
  // a daemon that has come back, say — has changed since the crash.
  let broken = true;
  function Boom() {
    if (broken) throw new Error("render exploded");
    return <p>recovered</p>;
  }

  it("catches a throwing child and resets on Try again", () => {
    // React logs the caught error; silence it so the run stays readable.
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    broken = true;
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );

    expect(screen.getByText("Something went wrong")).toBeTruthy();
    expect(screen.getByText("render exploded")).toBeTruthy();
    // No stack trace ever reaches the screen.
    expect(document.body.textContent).not.toContain("at Boom");

    broken = false;
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(screen.getByText("recovered")).toBeTruthy();
    expect(screen.queryByText("Something went wrong")).toBeNull();
    spy.mockRestore();
  });

  it("offers a reload as the second recovery path", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    broken = true;
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );
    expect(screen.getByRole("button", { name: "Reload" })).toBeTruthy();
    spy.mockRestore();
  });
});

describe("TableRow", () => {
  it("activates on the row's own Enter, and never hijacks a descendant's", () => {
    const onClick = vi.fn();
    render(
      <table>
        <tbody>
          <TableRow onClick={onClick}>
            <td>
              <a href="https://example.test/">open</a>
            </td>
          </TableRow>
        </tbody>
      </table>
    );

    // Enter on the focused link belongs to the link: the row must stay out of
    // it, or the address is unreachable by keyboard.
    fireEvent.keyDown(screen.getByRole("link", { name: "open" }), { key: "Enter" });
    expect(onClick).not.toHaveBeenCalled();

    // Enter on the row itself still activates the row.
    fireEvent.keyDown(screen.getByRole("row"), { key: "Enter" });
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
