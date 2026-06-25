import {
  sparse_matrix_from_row_maps,
  sparse_stats,
  sparse_to_dense_array,
  sparse_to_dense_tensor,
  sparse_row_sums,
} from './sparse';

describe('sparse_matrix_from_row_maps', () => {
  it('builds a CSR matrix with column indices sorted ascending', () => {
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
  });

  it('throws for a column index outside the declared width', () => {
    expect(() =>
      sparse_matrix_from_row_maps([new Map([[3, 1]])], 2),
    ).toThrow('outside matrix width');
  });

  it('throws for a negative column index', () => {
    expect(() =>
      sparse_matrix_from_row_maps([new Map([[-1, 1]])], 3),
    ).toThrow('outside matrix width');
  });

  it('defaults cols to rows.length for a square matrix', () => {
    const m = sparse_matrix_from_row_maps([
      new Map([[0, 1]]),
      new Map([[1, 2]]),
      new Map([[2, 3]]),
    ]);
    expect(m.rows).toBe(3);
    expect(m.cols).toBe(3);
    expect(Array.from(sparse_to_dense_array(m))).toEqual([
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3],
    ]);
  });

  it('handles an empty row list', () => {
    const m = sparse_matrix_from_row_maps([], 5);
    expect(m.rows).toBe(0);
    expect(m.cols).toBe(5);
    expect(m.data.length).toBe(0);
  });
});

describe('sparse_to_dense_array', () => {
  it('fills implicit zeros in the output grid', () => {
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
});

describe('sparse_to_dense_tensor', () => {
  it('returns a float32 tensor with correct shape and values', () => {
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
});

describe('sparse_row_sums', () => {
  it('computes per-row totals including diagonal', () => {
    const m = sparse_matrix_from_row_maps(
      [
        new Map([[0, 1], [1, 2], [2, 3]]),
        new Map([[0, 4]]),
      ],
      3,
    );
    expect(Array.from(sparse_row_sums(m))).toEqual([6, 4]);
  });

  it('excludes the diagonal entry when ignore_diagonal is true', () => {
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
});

describe('sparse_stats', () => {
  it('reports shape, nnz, min, max, and mean — accounting for implicit zeros', () => {
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

  it('uses the stored minimum when the matrix is fully dense (no implicit zeros)', () => {
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

  it('reports all-zero stats for an empty matrix', () => {
    const m = sparse_matrix_from_row_maps([new Map(), new Map()], 2);
    const stats = sparse_stats(m);
    expect(stats.nnz).toBe(0);
    expect(stats.min).toBe(0);
    expect(stats.max).toBe(0);
    expect(stats.mean).toBe(0);
  });
});
