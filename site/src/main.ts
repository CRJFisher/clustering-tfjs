import "./style.css";
import { run_race } from "./race";
import type { RaceResult } from "./race_protocol";

// A minimal harness driver: a button kicks the race and the results render as
// plain text. The screenshot-worthy dual-panel UI (live timers, racing bars,
// speedup tiles) lands in task-55.4; this proves the worker harness end to end.

function format_lane(result: RaceResult): string {
  const label =
    result.actual_backend === result.requested_lane
      ? result.actual_backend
      : `${result.actual_backend} (requested ${result.requested_lane})`;
  return [
    `${label}: median ${result.median_ms.toFixed(1)} ms`,
    `(min ${result.min_ms.toFixed(1)} / max ${result.max_ms.toFixed(1)})`,
    `· first run ${result.first_run_ms.toFixed(1)} ms`,
    `· ${Math.round(result.points_per_sec).toLocaleString()} pts/s`,
  ].join(" ");
}

function init(): void {
  const button = document.querySelector<HTMLButtonElement>("#run-race");
  const output = document.querySelector<HTMLPreElement>("#race-output");
  if (!button || !output) return;

  button.addEventListener("click", async () => {
    button.disabled = true;
    output.textContent = "Running race…\n";
    try {
      const outcome = await run_race(undefined, {
        on_progress: (lane, phase, rep) => {
          output.textContent += `  [${lane}] ${phase}${rep != null ? ` ${rep}` : ""}\n`;
        },
      });
      output.textContent = [
        format_lane(outcome.cpu),
        format_lane(outcome.gpu),
        "",
        `${outcome.gpu_backend.toUpperCase()} is ${outcome.speedup.toFixed(2)}× faster than CPU (median)`,
        `Result checksum match: ${
          Math.abs(outcome.cpu.result_checksum - outcome.gpu.result_checksum) <
          Math.abs(outcome.cpu.result_checksum) * 1e-3
            ? "yes"
            : "DIVERGED"
        }`,
        "",
        "Numbers come from YOUR hardware.",
      ].join("\n");
    } catch (error) {
      output.textContent += `\nRace failed: ${
        error instanceof Error ? error.message : String(error)
      }`;
    } finally {
      button.disabled = false;
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
