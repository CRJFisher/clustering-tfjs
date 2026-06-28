// A main-thread requestAnimationFrame stopwatch. It reports ELAPSED wall-clock
// only — never a measured number. This separation is the honesty contract: the
// live ticking value is page-elapsed time (worker spawn, backend init, shader
// compile, discarded warmups, and message latency all included), which is
// materially larger than the per-rep `median_ms` the harness measures. The UI
// labels the ticking value "elapsed" and snaps to the authoritative median when
// the result lands, so nothing dishonest is ever shown. The stopwatch itself
// never knows the median, which is what keeps that boundary unbreakable.

export interface Stopwatch {
  start(): void;
  freeze(): void;
  reset(): void;
}

export function make_stopwatch(
  on_tick: (elapsed_ms: number) => void,
): Stopwatch {
  let start_ms = 0;
  let frame = 0;
  let running = false;

  function tick(): void {
    if (!running) return;
    on_tick(performance.now() - start_ms);
    frame = requestAnimationFrame(tick);
  }

  return {
    start(): void {
      start_ms = performance.now();
      running = true;
      frame = requestAnimationFrame(tick);
    },
    freeze(): void {
      running = false;
      if (frame) cancelAnimationFrame(frame);
    },
    reset(): void {
      running = false;
      if (frame) cancelAnimationFrame(frame);
      on_tick(0);
    },
  };
}
