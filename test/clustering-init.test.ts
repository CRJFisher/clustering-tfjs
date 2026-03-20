import { Clustering } from '../src/clustering';
import { isInitialized } from '../src/tf-backend';

describe('Clustering.init() idempotency and concurrency', () => {
  afterEach(() => {
    Clustering.reset();
  });

  describe('concurrent calls with same config', () => {
    it('returns the exact same promise for concurrent calls with no config', async () => {
      const p1 = Clustering.init();
      const p2 = Clustering.init();
      expect(p1).toBe(p2);
      await Promise.all([p1, p2]);
      expect(isInitialized()).toBe(true);
    });

    it('returns the exact same promise for concurrent calls with identical config', async () => {
      const p1 = Clustering.init({ backend: 'cpu' });
      const p2 = Clustering.init({ backend: 'cpu' });
      expect(p1).toBe(p2);
      await Promise.all([p1, p2]);
    });

    it('resolves all concurrent callers without error', async () => {
      const promises = Array.from({ length: 10 }, () => Clustering.init());
      await expect(Promise.all(promises)).resolves.not.toThrow();
      expect(isInitialized()).toBe(true);
    });
  });

  describe('sequential calls with same config', () => {
    it('second call with no config after completion is a no-op', async () => {
      await Clustering.init();
      expect(isInitialized()).toBe(true);
      await Clustering.init();
      expect(isInitialized()).toBe(true);
    });

    it('second call with identical config after completion is a no-op', async () => {
      await Clustering.init({ backend: 'cpu' });
      await Clustering.init({ backend: 'cpu' });
      expect(isInitialized()).toBe(true);
    });
  });

  describe('conflicting config throws', () => {
    it('throws synchronously when called with different config during init', () => {
      Clustering.init({ backend: 'cpu' });
      expect(() => Clustering.init({ backend: 'wasm' })).toThrow(
        /already been called with a different configuration/,
      );
    });

    it('throws synchronously when called with different config after init', async () => {
      await Clustering.init({ backend: 'cpu' });
      expect(() => Clustering.init({ backend: 'wasm' })).toThrow(
        /already been called with a different configuration/,
      );
    });

    it('throws when flags differ', async () => {
      await Clustering.init({ flags: { WEBGL_PACK: true } });
      expect(() => Clustering.init({ flags: { WEBGL_PACK: false } })).toThrow(
        /already been called with a different configuration/,
      );
    });

    it('throws when going from no-config to specific config', async () => {
      await Clustering.init();
      expect(() => Clustering.init({ backend: 'cpu' })).toThrow(
        /already been called with a different configuration/,
      );
    });
  });

  describe('config normalization', () => {
    it('treats {} and { flags: undefined } as equivalent', async () => {
      const p1 = Clustering.init({});
      const p2 = Clustering.init({ flags: undefined });
      expect(p1).toBe(p2);
      await p1;
    });

    it('treats { flags: {} } and {} as equivalent', async () => {
      const p1 = Clustering.init({ flags: {} });
      const p2 = Clustering.init({});
      expect(p1).toBe(p2);
      await p1;
    });

    it('treats configs with different key order as equivalent', async () => {
      const p1 = Clustering.init({ backend: 'cpu', flags: { A: 1 } });
      const p2 = Clustering.init({ flags: { A: 1 }, backend: 'cpu' });
      expect(p1).toBe(p2);
      await p1;
    });
  });

  describe('reset and re-initialization', () => {
    it('allows re-initialization with different config after reset', async () => {
      await Clustering.init({ backend: 'cpu' });
      Clustering.reset();
      expect(isInitialized()).toBe(false);
      await Clustering.init();
      expect(isInitialized()).toBe(true);
    });

    it('allows re-initialization with same config after reset', async () => {
      await Clustering.init();
      Clustering.reset();
      await Clustering.init();
      expect(isInitialized()).toBe(true);
    });
  });

  describe('error recovery', () => {
    it('allows retry after failed initialization', async () => {
      // Force a failure by using a platform that will fail to load
      const badInit = Clustering.init({ forcePlatform: 'react-native' });
      await expect(badInit).rejects.toThrow();
      expect(isInitialized()).toBe(false);

      // Should be able to retry with valid config
      await Clustering.init();
      expect(isInitialized()).toBe(true);
    });
  });
});
