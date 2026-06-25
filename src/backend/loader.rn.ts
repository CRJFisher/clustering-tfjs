export async function load_tensor_flow() {
  try {
    // Module name as variable + webpackIgnore: prevents bundlers from statically
    // resolving this optional peer dep and avoids TS type errors when it's absent.
    const tf_rn_module = '@tensorflow/tfjs-react-native';
    const tf = await import(/* webpackIgnore: true */ tf_rn_module as string) as typeof import('@tensorflow/tfjs');

    await tf.ready();

    try {
      await tf.setBackend('rn-webgl');
      console.log('Using TensorFlow.js React Native WebGL backend (GPU accelerated)');
    } catch {
      console.warn('WebGL backend not available, falling back to CPU');
      await tf.setBackend('cpu');
      console.log('Using TensorFlow.js CPU backend');
    }

    return tf;
  } catch {
    throw new Error(
      'TensorFlow.js React Native not found. Please install:\n' +
      '- @tensorflow/tfjs-react-native\n' +
      '- Platform-specific GL dependencies:\n' +
      '  For Expo: expo-gl and expo-gl-cpp\n' +
      '  For bare RN: gl-react-native\n\n' +
      'Also ensure you have called tf.ready() before using the library.'
    );
  }
}
