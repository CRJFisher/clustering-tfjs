import { sparse_matrix_from_row_maps, sparse_matvec, sparse_stats } from './sparse';

describe('SparseMatrix CSR helpers', () => {
  it('stores sorted CSR rows and applies matvec', () => {
    const rows = [
      new Map([
        [2, 3],
        [0, 1],
      ]),
      new Map([[1, 2]]),
    ];

    const matrix = sparse_matrix_from_row_maps(rows, 3);

    expect(Array.from(matrix.indptr)).toEqual([0, 2, 3]);
    expect(Array.from(matrix.indices)).toEqual([0, 2, 1]);
    expect(Array.from(matrix.data)).toEqual([1, 3, 2]);

    const result = sparse_matvec(matrix, new Float64Array([10, 20, 30]));
    expect(Array.from(result)).toEqual([100, 40]);
  });

  it('reports dense-aware matrix statistics', () => {
    const matrix = sparse_matrix_from_row_maps(
      [
        new Map([[0, 1]]),
        new Map([[1, 2]]),
      ],
      2,
    );

    expect(sparse_stats(matrix)).toEqual({
      shape: [2, 2],
      nnz: 2,
      min: 0,
      max: 2,
      mean: 0.75,
    });
  });
});
