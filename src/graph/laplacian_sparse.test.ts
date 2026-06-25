import * as tf from '../../test_support/tensorflow_helper';

import {
  normalised_laplacian,
  sparse_normalised_laplacian_operator,
} from './laplacian';
import { sparse_matrix_from_row_maps } from './sparse';

describe('sparse_normalised_laplacian_operator', () => {
  it('matches dense normalised Laplacian matvec', () => {
    const affinity = sparse_matrix_from_row_maps(
      [
        new Map([[0, 1], [1, 1]]),
        new Map([[0, 1], [1, 1], [2, 0.5]]),
        new Map([[1, 0.5], [2, 1]]),
      ],
      3,
    );

    const dense_affinity = tf.tensor2d([
      [1, 1, 0],
      [1, 1, 0.5],
      [0, 0.5, 1],
    ]);
    const dense_laplacian = normalised_laplacian(dense_affinity);
    const v = tf.tensor2d([[2], [-1], [0.5]]);
    const Lv = dense_laplacian.matMul(v);
    const dense_result = Lv.arraySync() as number[][];
    Lv.dispose();
    v.dispose();
    dense_laplacian.dispose();
    dense_affinity.dispose();

    const sparse_result = sparse_normalised_laplacian_operator(affinity).operator.matvec(
      new Float64Array([2, -1, 0.5]),
    );

    for (let i = 0; i < sparse_result.length; i++) {
      expect(sparse_result[i]).toBeCloseTo(dense_result[i][0], 6);
    }
  });
});
