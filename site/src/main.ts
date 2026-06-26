import "./style.css";
import { make_race_ui } from "./race_ui";

function init(): void {
  // The controller owns the run button and the n-slider, wiring its own listeners
  // and the single-flight scheduler — there is nothing left for main to wire.
  make_race_ui();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
