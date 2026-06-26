import "./style.css";
import { make_race_ui } from "./race_ui";

function init(): void {
  const ui = make_race_ui();
  const button = document.querySelector<HTMLButtonElement>("#run-race");
  button?.addEventListener("click", () => {
    void ui.run();
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
