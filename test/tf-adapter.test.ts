import * as tf from '../src/tf-adapter';
import { resetBackend } from '../src/tf-backend';

describe('tf-adapter lazy wrappers', () => {
  afterEach(() => {
    resetBackend();
  });

  it('provides tensor creation functions', () => {
    const t = tf.tensor2d([[1, 2], [3, 4]]);
    expect(t.shape).toEqual([2, 2]);
    t.dispose();
  });

  it('provides tidy for memory management', () => {
    const result = tf.tidy(() => {
      const a = tf.tensor1d([1, 2, 3]);
      return a.sum();
    });
    expect(result.dataSync()[0]).toBe(6);
    result.dispose();
  });

  it('provides linalg namespace with qr', () => {
    const result = tf.tidy(() => {
      const a = tf.tensor2d([[1, 2], [3, 4]]);
      const [q, r] = tf.linalg.qr(a);
      return { qShape: q.shape, rShape: r.shape };
    });
    expect(result.qShape).toEqual([2, 2]);
    expect(result.rShape).toEqual([2, 2]);
  });

  it('provides math operations', () => {
    const result = tf.tidy(() => {
      const a = tf.tensor1d([1, 4, 9]);
      return tf.sqrt(a);
    });
    expect(Array.from(result.dataSync())).toEqual([1, 2, 3]);
    result.dispose();
  });

  it('provides eye, ones, zeros', () => {
    tf.tidy(() => {
      const e = tf.eye(3);
      expect(e.shape).toEqual([3, 3]);

      const o = tf.ones([2, 3]);
      expect(o.shape).toEqual([2, 3]);

      const z = tf.zeros([4]);
      expect(z.shape).toEqual([4]);
    });
  });

  it('default export works as namespace', () => {
    const result = tf.default.tidy(() => {
      return tf.default.scalar(42);
    });
    expect(result.dataSync()[0]).toBe(42);
    result.dispose();
  });
});
