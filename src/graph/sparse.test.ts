import {
  sparse_matrix_from_row_maps,
  sparse_matvec,
  sparse_stats,
  sparse_to_dense_array,
  sparse_to_dense_tensor,
  sparse_row_sums,
} from './sparse';

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

  it('throws for a column index outside the declared width', () => {
    expect(() =>
      sparse_matrix_from_row_maps([new Map([[3, 1]])], 2),
    ).toThrow('outside matrix width');
  });

  it('handles an empty row list', () => {
    const m = sparse_matrix_from_row_maps([], 5);
    expect(m.rows).toBe(0);
    expect(m.cols).toBe(5);
    expect(m.data.length).toBe(0);
  });

  it('sparse_to_dense_array fills implicit zeros', () => {
    const m = sparse_matrix_from_row_maps(
      [
        new Map([[0, 1], [2, 3]]),
        new Map([[1, 2]]),
      ],
      3,
    );
    expect(sparse_to_dense_array(m)).toEqual([
      [1, 0, 3],
      [0, 2, 0],
    ]);
  });

  it('sparse_to_dense_tensor returns float32 with correct shape and values', () => {
    const m = sparse_matrix_from_row_maps(
      [new Map([[0, 1]]), new Map([[1, 2]])],
      2,
    );
    const t = sparse_to_dense_tensor(m);
    expect(t.dtype).toBe('float32');
    expect(t.shape).toEqual([2, 2]);
    expect(t.arraySync()).toEqual([[1, 0], [0, 2]]);
    t.dispose();
  });

  it('sparse_row_sums computes per-row totals', () => {
    const m = sparse_matrix_from_row_maps(
      [
        new Map([[0, 1], [1, 2], [2, 3]]),
        new Map([[0, 4]]),
      ],
      3,
    );
    expect(Array.from(sparse_row_sums(m))).toEqual([6, 4]);
  });

  it('sparse_row_sums with ignore_diagonal excludes the diagonal entry', () => {
    const m = sparse_matrix_from_row_maps(
      [
        new Map([[0, 10], [1, 2]]),
        new Map([[0, 3], [1, 10]]),
        new Map([[2, 10]]),
      ],
      3,
    );
    expect(Array.from(sparse_row_sums(m, true))).toEqual([2, 3, 0]);
  });

  it('sparse_matvec throws when vector length mismatches cols', () => {
    const m = sparse_matrix_from_row_maps([new Map([[0, 1]])], 3);
    expect(() => sparse_matvec(m, new Float64Array([1, 2]))).toThrow(
      'does not match',
    );
  });

  it('sparse_stats uses stored min when matrix is fully dense (no implicit zeros)', () => {
    const m = sparse_matrix_from_row_maps(
      [
        new Map([[0, 1], [1, 2]]),
        new Map([[0, 3], [1, 4]]),
      ],
      2,
    );
    const stats = sparse_stats(m);
    expect(stats.nnz).toBe(4);
    expect(stats.min).toBe(1);
    expect(stats.max).toBe(4);
    expect(stats.mean).toBe(2.5);
  });

  it('sparse_stats handles zero stored entries', () => {
    const m = sparse_matrix_from_row_maps([new Map(), new Map()], 2);
    const stats = sparse_stats(m);
    expect(stats.nnz).toBe(0);
    expect(stats.min).toBe(0);
    expect(stats.max).toBe(0);
    expect(stats.mean).toBe(0);
  });

});
