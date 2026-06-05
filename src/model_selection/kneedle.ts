/**
 * Options for knee/elbow detection.
 */
export interface KneedleOptions {
  /**
   * Direction of the curve:
   * - 'concave': for curves that decrease and flatten (e.g., WSS vs k)
   * - 'convex': for curves that increase and flatten
   * Default: 'concave'
   */
  direction?: 'concave' | 'convex';

  /**
   * Sensitivity parameter S. Higher values require a more pronounced knee.
   * Default: 1.0
   */
  sensitivity?: number;
}

/**
 * Result of knee detection.
 */
export interface KneedleResult {
  /** The x-value at the detected knee, or null if no knee found */
  knee_x: number | null;
  /** Index into the input arrays, or null if no knee found */
  knee_index: number | null;
  /** Normalized difference values for each point (for scoring) */
  differences: number[];
}

/**
 * Detects the "knee" or "elbow" point in a curve using a simplified
 * Kneedle algorithm (Satopaa et al., 2011).
 *
 * The algorithm:
 * 1. Normalizes x and y values to [0, 1]
 * 2. Computes the difference between the curve and a straight line
 *    connecting the first and last points
 * 3. Finds the point of maximum deviation as the knee
 *
 * @param x_values - Monotonically increasing x-values (e.g., k values)
 * @param y_values - Corresponding y-values (e.g., WSS values)
 * @param options - Configuration options
 * @returns KneedleResult with knee location and difference values
 */
export function find_knee(
  x_values: number[],
  y_values: number[],
  options: KneedleOptions = {},
): KneedleResult {
  const { direction = 'concave', sensitivity = 1.0 } = options;

  const n = x_values.length;

  if (n < 3) {
    return { knee_x: null, knee_index: null, differences: [] };
  }

  // Normalize x and y to [0, 1]
  const x_min = x_values[0];
  const x_max = x_values[n - 1];
  const x_range = x_max - x_min;

  if (x_range === 0) {
    return { knee_x: null, knee_index: null, differences: new Array(n).fill(0) };
  }

  let y_min = y_values[0];
  let y_max = y_values[0];
  for (let i = 1; i < n; i++) {
    if (y_values[i] < y_min) y_min = y_values[i];
    if (y_values[i] > y_max) y_max = y_values[i];
  }
  const y_range = y_max - y_min;

  if (y_range === 0) {
    return { knee_x: null, knee_index: null, differences: new Array(n).fill(0) };
  }

  const x_norm = x_values.map((x) => (x - x_min) / x_range);
  const y_norm = y_values.map((y) => (y - y_min) / y_range);

  // Compute difference from the diagonal line connecting first and last points
  const y_first = y_norm[0];
  const y_last = y_norm[n - 1];
  const x_first = x_norm[0];
  const x_last = x_norm[n - 1];

  const differences: number[] = [];
  for (let i = 0; i < n; i++) {
    // Point on the straight line at this x
    const y_line =
      y_first + ((y_last - y_first) * (x_norm[i] - x_first)) / (x_last - x_first);
    differences.push(y_norm[i] - y_line);
  }

  // For a concave/decreasing curve (WSS), the curve drops below the
  // diagonal. The knee is where it deviates most (most negative difference).
  // For a convex/increasing curve, the curve rises above the diagonal.
  // The knee is where it deviates most (most positive difference).

  // Threshold: sensitivity * average step size
  const threshold = sensitivity / (n - 1);

  let best_index = -1;
  let best_deviation = -Infinity;

  for (let i = 1; i < n - 1; i++) {
    // Absolute deviation from the diagonal
    const deviation =
      direction === 'concave' ? -differences[i] : differences[i];
    if (deviation > best_deviation) {
      best_deviation = deviation;
      best_index = i;
    }
  }

  // Check if the knee is significant enough
  if (best_index === -1 || best_deviation < threshold) {
    return { knee_x: null, knee_index: null, differences };
  }

  return {
    knee_x: x_values[best_index],
    knee_index: best_index,
    differences,
  };
}
