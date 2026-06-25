import {
  detect_connected_components,
  detect_sparse_connected_components,
} from './connected_components';
import { sparse_matrix_from_row_maps, sparse_to_dense_tensor } from './sparse';

describe('detect_sparse_connected_components – sparse/dense parity', () => {
  it('matches dense connected component labels', () => {
    const sparse = sparse_matrix_from_row_maps(
      [
        new Map([[1, 1]]),
        new Map([[0, 1]]),
        new Map([[3, 1]]),
        new Map([[2, 1]]),
      ],
      4,
    );
    const dense = sparse_to_dense_tensor(sparse);

    const sparse_result = detect_sparse_connected_components(sparse);
    const dense_result = detect_connected_components(dense);

    expect(sparse_result.num_components).toBe(dense_result.num_components);
    expect(sparse_result.is_fully_connected).toBe(false);
    expect(Array.from(sparse_result.component_labels)).toEqual(
      Array.from(dense_result.component_labels),
    );

    dense.dispose();
  });
});
