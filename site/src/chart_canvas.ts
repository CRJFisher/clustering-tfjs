// A dependency-free, log–log line chart drawn straight to a canvas — the same
// render-once-on-demand discipline as scatter_canvas.ts, no animation loop. The
// O(n²·d) affinity scaling law plots as a straight line per backend on log–log
// axes, so a glance reads both the absolute speed (the accelerated line sits
// below CPU) and the scaling (parallel straight lines, widening gap).
//
// The pixel-mapping and tick math are pure functions, exported and unit-tested
// without a DOM; render_chart is the thin canvas shell around them.

export interface ChartPoint {
  x: number;
  y: number;
}

export interface ChartSeries {
  label: string;
  color: string;
  points: ChartPoint[];
}

export interface ChartOptions {
  x_domain: [number, number];
  // Explicit x tick values (the swept sample sizes); each is labelled on the axis.
  x_ticks: number[];
}

const PADDING_LEFT = 56;
const PADDING_RIGHT = 16;
const PADDING_TOP = 16;
const PADDING_BOTTOM = 40;

const AXIS_COLOR = "#2b3340";
const GRID_COLOR = "#1b212b";
const LABEL_COLOR = "#9aa4b2";
const POINT_RADIUS = 3;

// Map a positive value onto a pixel coordinate along a log axis. Exported and
// pure: the whole geometry of the chart reduces to this one call per axis.
export function log_scale(
  value: number,
  domain_min: number,
  domain_max: number,
  range_min: number,
  range_max: number,
): number {
  const log_min = Math.log10(domain_min);
  const log_max = Math.log10(domain_max);
  const t = (Math.log10(value) - log_min) / (log_max - log_min);
  return range_min + t * (range_max - range_min);
}

// 1–2–5 tick values within [min, max] inclusive of the bounding decades — the
// conventional log-axis gridline set. Pure, so it is unit-tested directly.
export function nice_log_ticks(min: number, max: number): number[] {
  if (!(min > 0) || !(max > 0) || max < min) return [];
  const ticks: number[] = [];
  const start = Math.floor(Math.log10(min));
  const end = Math.ceil(Math.log10(max));
  for (let decade = start; decade <= end; decade++) {
    for (const mantissa of [1, 2, 5]) {
      const value = mantissa * 10 ** decade;
      if (value >= min - 1e-9 && value <= max + 1e-9) ticks.push(value);
    }
  }
  return ticks;
}

// Snap a [min, max] data range out to its bounding decades so the y-axis domain
// stays stable as points stream in (it only ever jumps by whole decades, never
// jitters per point). Returns a safe default when there is no data yet.
export function fit_log_domain(values: number[]): [number, number] {
  const positive = values.filter((v) => v > 0);
  if (positive.length === 0) return [1, 1000];
  const min = Math.min(...positive);
  const max = Math.max(...positive);
  const lo = 10 ** Math.floor(Math.log10(min));
  const hi = 10 ** Math.ceil(Math.log10(max));
  // A single-decade range (all points within one power of 10) still needs hi>lo.
  return [lo, hi === lo ? lo * 10 : hi];
}

function format_tick(value: number): string {
  if (value >= 1000) return `${value / 1000}k`;
  if (value >= 1) return String(value);
  return String(value);
}

export function render_chart(
  canvas: HTMLCanvasElement,
  series: ChartSeries[],
  options: ChartOptions,
): void {
  const context = canvas.getContext("2d");
  if (!context) return;
  const ctx = context;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const css_w = rect.width || canvas.clientWidth || canvas.width;
  const css_h = rect.height || canvas.clientHeight || canvas.height;
  canvas.width = Math.round(css_w * dpr);
  canvas.height = Math.round(css_h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, css_w, css_h);

  const plot_left = PADDING_LEFT;
  const plot_right = css_w - PADDING_RIGHT;
  const plot_top = PADDING_TOP;
  const plot_bottom = css_h - PADDING_BOTTOM;

  const [x_min, x_max] = options.x_domain;
  const all_y = series.flatMap((s) => s.points.map((p) => p.y));
  const [y_min, y_max] = fit_log_domain(all_y);

  const to_x = (value: number): number =>
    log_scale(value, x_min, x_max, plot_left, plot_right);
  // Pixels grow downward, so the larger y (slower) maps to the smaller pixel.
  const to_y = (value: number): number =>
    log_scale(value, y_min, y_max, plot_bottom, plot_top);

  // Y gridlines + labels (ms).
  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  for (const tick of nice_log_ticks(y_min, y_max)) {
    const py = to_y(tick);
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plot_left, py);
    ctx.lineTo(plot_right, py);
    ctx.stroke();
    ctx.fillStyle = LABEL_COLOR;
    ctx.textAlign = "right";
    ctx.fillText(`${format_tick(tick)} ms`, plot_left - 8, py);
  }

  // X gridlines + labels (sample size n).
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const tick of options.x_ticks) {
    const px = to_x(tick);
    ctx.strokeStyle = GRID_COLOR;
    ctx.beginPath();
    ctx.moveTo(px, plot_top);
    ctx.lineTo(px, plot_bottom);
    ctx.stroke();
    ctx.fillStyle = LABEL_COLOR;
    ctx.fillText(format_tick(tick), px, plot_bottom + 8);
  }

  // Axis frame.
  ctx.strokeStyle = AXIS_COLOR;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(plot_left, plot_top);
  ctx.lineTo(plot_left, plot_bottom);
  ctx.lineTo(plot_right, plot_bottom);
  ctx.stroke();

  // Axis titles.
  ctx.fillStyle = LABEL_COLOR;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("samples (n)", (plot_left + plot_right) / 2, css_h);

  // Each series: a polyline through its measured points plus a marker per point.
  // Points stream in ascending n, so they are already in draw order.
  for (const s of series) {
    if (s.points.length === 0) continue;
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.points.forEach((point, index) => {
      const px = to_x(point.x);
      const py = to_y(point.y);
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    for (const point of s.points) {
      ctx.beginPath();
      ctx.arc(to_x(point.x), to_y(point.y), POINT_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Legend, top-left inside the plot.
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  let legend_y = plot_top + 10;
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.beginPath();
    ctx.arc(plot_left + 12, legend_y, POINT_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = LABEL_COLOR;
    ctx.fillText(s.label, plot_left + 22, legend_y);
    legend_y += 16;
  }
}
