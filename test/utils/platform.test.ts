import { isReactNative, isNode, isBrowser, getPlatform } from '../../src/utils/platform';

describe('platform detection', () => {
  // Store original globals for cleanup
  const originalGlobalThis = { ...globalThis };

  function setGlobal(key: string, value: unknown): void {
    (globalThis as Record<string, unknown>)[key] = value;
  }

  function deleteGlobal(key: string): void {
    delete (globalThis as Record<string, unknown>)[key];
  }

  afterEach(() => {
    // Clean up any RN globals we set
    for (const key of ['HermesInternal', '__fbBatchedBridge', 'nativeCallSyncHook']) {
      if (!(key in originalGlobalThis)) {
        deleteGlobal(key);
      }
    }
  });

  describe('isReactNative', () => {
    it('returns false in standard Node.js environment', () => {
      expect(isReactNative()).toBe(false);
    });

    it('returns true when HermesInternal is defined', () => {
      setGlobal('HermesInternal', {});
      expect(isReactNative()).toBe(true);
    });

    it('returns true when __fbBatchedBridge is defined', () => {
      setGlobal('__fbBatchedBridge', {});
      expect(isReactNative()).toBe(true);
    });

    it('returns true when nativeCallSyncHook is defined', () => {
      setGlobal('nativeCallSyncHook', () => {});
      expect(isReactNative()).toBe(true);
    });
  });

  describe('isNode', () => {
    it('returns true in Node.js environment', () => {
      expect(isNode()).toBe(true);
    });
  });

  describe('isBrowser', () => {
    it('returns false in Node.js environment', () => {
      expect(isBrowser()).toBe(false);
    });
  });

  describe('getPlatform', () => {
    it('returns "node" in Node.js environment', () => {
      expect(getPlatform()).toBe('node');
    });

    it('returns "react-native" when RN globals are present', () => {
      setGlobal('HermesInternal', {});
      expect(getPlatform()).toBe('react-native');
    });
  });
});
