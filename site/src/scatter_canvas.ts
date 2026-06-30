import type { Projection2d } from "./project_2d";

// Fixed high-contrast cluster palette mirroring the theme tokens in style.css
// (--accent / --accent-2 / --accent-3) plus a 4th hue for the default 4 centers.
// A canvas fill needs concrete colors, not CSS variables, so these are kept in
// sync with the stylesheet by intent.
const CLUSTER_COLORS = ["#7c9cff", "#63e6be", "#ffa94d", "#ff8787"] as const;

// Noise points (label < 0, emitted by HDBSCAN) are drawn in a muted gray so
// "belongs to no cluster" reads as visually distinct from every cluster colour —
// the grid's no-structure cell depends on this to show all-noise honestly.
const NOISE_COLOR = "#6b7280";

const DEFAULT_POINT_RADIUS = 2.2;
const DEFAULT_PADDING = 12;

export interface ScatterOptions {
  point_radius?: number;
  padding?: number;
}

// Draws the seeded dataset once. There is no animation loop and no
// requestAnimationFrame: the caller invokes this a single time, before timing
// starts, so a redraw can never land inside a measured run and compete with the
// GPU for frames.
export function render_scatter(
  canvas: HTMLCanvasElement,
  projection: Projection2d,
  labels: Int32Array,
  options: ScatterOptions = {},
): void {
  const context = canvas.getContext("2d");
  if (!context) return;
  // Bound to a non-null const so the draw_group closure below keeps the narrowing
  // (TS drops control-flow narrowing for variables captured in nested functions).
  const ctx = context;

  const point_radius = options.point_radius ?? DEFAULT_POINT_RADIUS;
  const padding = options.padding ?? DEFAULT_PADDING;

  // Render at device resolution so the scatter stays crisp on high-DPI displays
  // — this fold doubles as the 1200×630 og:image, where blur is fatal. Drawing
  // is done in CSS pixels after scaling by the device pixel ratio.
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const css_w = rect.width || canvas.clientWidth || canvas.width;
  const css_h = rect.height || canvas.clientHeight || canvas.height;
  canvas.width = Math.round(css_w * dpr);
  canvas.height = Math.round(css_h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, css_w, css_h);

  const { x, y } = projection;
  let x_min = Infinity;
  let x_max = -Infinity;
  let y_min = Infinity;
  let y_max = -Infinity;
  for (let i = 0; i < x.length; i++) {
    if (x[i] < x_min) x_min = x[i];
    if (x[i] > x_max) x_max = x[i];
    if (y[i] < y_min) y_min = y[i];
    if (y[i] > y_max) y_max = y[i];
  }
  const x_span = x_max - x_min || 1;
  const y_span = y_max - y_min || 1;
  const plot_w = css_w - 2 * padding;
  const plot_h = css_h - 2 * padding;

  function draw_group(matches: (label: number) => boolean): void {
    for (let i = 0; i < x.length; i++) {
      if (!matches(labels[i])) continue;
      const px = padding + ((x[i] - x_min) / x_span) * plot_w;
      // Flip y so a larger projected value draws higher on screen.
      const py = padding + (1 - (y[i] - y_min) / y_span) * plot_h;
      ctx.beginPath();
      ctx.arc(px, py, point_radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Group draws by color so fillStyle is set once per cluster, not once per
  // point. The axes are meaningless (axis-free styling), so x and y fit the box
  // independently to maximize the visible separation. Noise is drawn first, under
  // the clusters, so a stray noise point never sits on top of cluster structure.
  ctx.globalAlpha = 0.85;
  ctx.fillStyle = NOISE_COLOR;
  draw_group((label) => label < 0);
  for (let c = 0; c < CLUSTER_COLORS.length; c++) {
    ctx.fillStyle = CLUSTER_COLORS[c];
    draw_group((label) => label >= 0 && label % CLUSTER_COLORS.length === c);
  }
  ctx.globalAlpha = 1;
}
