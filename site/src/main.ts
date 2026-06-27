import "./style.css";
import { make_race_ui } from "./race_ui";
import { make_grid_ui } from "./grid_ui";

function init(): void {
  // Each controller owns its own section: the race wires the run button and
  // n-slider; the grid builds its 5×5 DOM and kicks off the worker. Nothing is
  // left for main to wire beyond mounting them.
  make_race_ui();
  make_grid_ui();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
