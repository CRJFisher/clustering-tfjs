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
  kneeX: number | null;
  /** Index into the input arrays, or null if no knee found */
  kneeIndex: number | null;
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
 * @param xValues - Monotonically increasing x-values (e.g., k values)
 * @param yValues - Corresponding y-values (e.g., WSS values)
 * @param options - Configuration options
 * @returns KneedleResult with knee location and difference values
 */
export function findKnee(
  xValues: number[],
  yValues: number[],
  options: KneedleOptions = {},
): KneedleResult {
  const { direction = 'concave', sensitivity = 1.0 } = options;

  const n = xValues.length;

  if (n < 3) {
    return { kneeX: null, kneeIndex: null, differences: [] };
  }

  // Normalize x and y to [0, 1]
  const xMin = xValues[0];
  const xMax = xValues[n - 1];
  const xRange = xMax - xMin;

  if (xRange === 0) {
    return { kneeX: null, kneeIndex: null, differences: new Array(n).fill(0) };
  }

  let yMin = yValues[0];
  let yMax = yValues[0];
  for (let i = 1; i < n; i++) {
    if (yValues[i] < yMin) yMin = yValues[i];
    if (yValues[i] > yMax) yMax = yValues[i];
  }
  const yRange = yMax - yMin;

  if (yRange === 0) {
    return { kneeX: null, kneeIndex: null, differences: new Array(n).fill(0) };
  }

  const xNorm = xValues.map((x) => (x - xMin) / xRange);
  const yNorm = yValues.map((y) => (y - yMin) / yRange);

  // Compute difference from the diagonal line connecting first and last points
  const yFirst = yNorm[0];
  const yLast = yNorm[n - 1];
  const xFirst = xNorm[0];
  const xLast = xNorm[n - 1];

  const differences: number[] = [];
  for (let i = 0; i < n; i++) {
    // Point on the straight line at this x
    const yLine =
      yFirst + ((yLast - yFirst) * (xNorm[i] - xFirst)) / (xLast - xFirst);
    differences.push(yNorm[i] - yLine);
  }

  // For a concave/decreasing curve (WSS), the curve drops below the
  // diagonal. The knee is where it deviates most (most negative difference).
  // For a convex/increasing curve, the curve rises above the diagonal.
  // The knee is where it deviates most (most positive difference).

  // Threshold: sensitivity * average step size
  const threshold = sensitivity / (n - 1);

  let bestIndex = -1;
  let bestDeviation = -Infinity;

  for (let i = 1; i < n - 1; i++) {
    // Absolute deviation from the diagonal
    const deviation =
      direction === 'concave' ? -differences[i] : differences[i];
    if (deviation > bestDeviation) {
      bestDeviation = deviation;
      bestIndex = i;
    }
  }

  // Check if the knee is significant enough
  if (bestIndex === -1 || bestDeviation < threshold) {
    return { kneeX: null, kneeIndex: null, differences };
  }

  return {
    kneeX: xValues[bestIndex],
    kneeIndex: bestIndex,
    differences,
  };
}
