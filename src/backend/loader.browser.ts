export async function load_tensor_flow() {
  if (typeof window !== 'undefined') {
    const global_window = window as Window & {
      tf?: typeof import('@tensorflow/tfjs');
    };
    if (global_window.tf) {
      return global_window.tf;
    }
  }

  try {
    const tf = await import('@tensorflow/tfjs');
    return tf as typeof import('@tensorflow/tfjs');
  } catch {
    throw new Error(
      'TensorFlow.js not found. Please load it before using this library:\n' +
      '<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>'
    );
  }
}
