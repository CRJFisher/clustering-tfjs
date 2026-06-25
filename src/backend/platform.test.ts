import { is_react_native, is_node, get_platform } from './platform';

const original_global_this = { ...globalThis };

function set_global(key: string, value: unknown): void {
  (globalThis as Record<string, unknown>)[key] = value;
}

function delete_global(key: string): void {
  delete (globalThis as Record<string, unknown>)[key];
}

afterEach(() => {
  for (const key of ['HermesInternal', '__fbBatchedBridge', 'nativeCallSyncHook']) {
    if (!(key in original_global_this)) {
      delete_global(key);
    }
  }
});

describe('is_react_native', () => {
  it('returns false in standard Node.js environment', () => {
    expect(is_react_native()).toBe(false);
  });

  it('returns true when HermesInternal is defined', () => {
    set_global('HermesInternal', {});
    expect(is_react_native()).toBe(true);
  });

  it('returns true when __fbBatchedBridge is defined', () => {
    set_global('__fbBatchedBridge', {});
    expect(is_react_native()).toBe(true);
  });

  it('returns true when nativeCallSyncHook is defined', () => {
    set_global('nativeCallSyncHook', () => {});
    expect(is_react_native()).toBe(true);
  });
});

describe('is_node', () => {
  it('returns true in Node.js environment', () => {
    expect(is_node()).toBe(true);
  });
});

describe('get_platform', () => {
  it('returns "node" in Node.js environment', () => {
    expect(get_platform()).toBe('node');
  });

  it('returns "react-native" when RN globals are present', () => {
    set_global('HermesInternal', {});
    expect(get_platform()).toBe('react-native');
  });
});
