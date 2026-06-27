import { copy_text } from "./clipboard";

// How long the "Copied!" state lingers before the button reverts.
const COPIED_RESET_MS = 1400;

// Wires a button so each click copies the current `text()` result and gives
// feedback two ways: a `data-copied` flag for a transient visual, and a polite
// live-region announcement (the motion-independent channel, so the affordance
// works under prefers-reduced-motion and for screen readers alike). `text` is a
// thunk so the button always copies the latest value, not a snapshot from wiring
// time. The visible label names what was copied ("Code copied", "npm command
// copied").
export function wire_copy_button(
  button: HTMLButtonElement,
  live_el: HTMLElement,
  text: () => string,
  label: string,
): void {
  let reset_handle: ReturnType<typeof setTimeout> | undefined;
  button.addEventListener("click", () => {
    void copy_text(text()).then((ok) => {
      live_el.textContent = ok
        ? `${label} copied`
        : "Copy failed — select the text and copy manually.";
      button.dataset.copied = ok ? "true" : "false";
      if (reset_handle !== undefined) clearTimeout(reset_handle);
      reset_handle = setTimeout(() => {
        button.dataset.copied = "false";
        live_el.textContent = "";
      }, COPIED_RESET_MS);
    });
  });
}
