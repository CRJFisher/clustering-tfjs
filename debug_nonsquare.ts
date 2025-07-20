import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from './src';

(async () => {
  const X = tf.randomUniform([5, 2]);
  const badCallable = (_: any) => tf.zeros([X.shape[0], X.shape[0] + 1]);
  const model = new SpectralClustering({ affinity: badCallable, nClusters: 2 });
  try {
    await model.fit(X);
    console.log('NO ERROR');
  } catch (e: any) {
    console.error('Error', e.message);
  }
})();

