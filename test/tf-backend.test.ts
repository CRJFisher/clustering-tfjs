import { ensureBackend, resetBackend, isInitialized, initializeBackend } from '../src/tf-backend';

describe('tf-backend', () => {
  afterEach(() => {
    resetBackend();
  });

  describe('ensureBackend', () => {
    it('auto-loads a backend in Node.js without explicit init', () => {
      const tf = ensureBackend();
      expect(tf).toBeDefined();
      expect(typeof tf.tensor2d).toBe('function');
      expect(typeof tf.tidy).toBe('function');
    });

    it('returns the same instance on subsequent calls', () => {
      const tf1 = ensureBackend();
      const tf2 = ensureBackend();
      expect(tf1).toBe(tf2);
    });

    it('sets isInitialized to true after auto-load', () => {
      expect(isInitialized()).toBe(false);
      ensureBackend();
      expect(isInitialized()).toBe(true);
    });

    it('throws if async init is in progress (race condition prevention)', async () => {
      // Start async init but don't await it yet.
      // initializeBackend sets initializationPromise synchronously before awaiting.
      const initPromise = initializeBackend({ backend: 'cpu' });

      // At this point initializationPromise is set but tfInstance is still null
      // (the promise hasn't resolved yet). A synchronous ensureBackend() call
      // should detect the in-flight promise and throw rather than silently
      // loading a different backend via loadBackendSync().
      expect(() => ensureBackend()).toThrow(
        'TensorFlow.js is being initialized asynchronously'
      );

      // Clean up: let the init complete
      await initPromise;
    });
  });

  describe('initializeBackend', () => {
    it('initializes and returns a tf instance', async () => {
      const tf = await initializeBackend();
      expect(tf).toBeDefined();
      expect(typeof tf.tensor2d).toBe('function');
    });

    it('returns cached instance on subsequent calls', async () => {
      const tf1 = await initializeBackend();
      const tf2 = await initializeBackend();
      expect(tf1).toBe(tf2);
    });

    it('returns the cached instance regardless of config after initialization', async () => {
      const tf = await initializeBackend();
      // Config conflict detection is handled at the Clustering.init() level,
      // not at the initializeBackend() level. Once initialized, initializeBackend
      // always returns the cached instance.
      const tf2 = await initializeBackend({ backend: 'cpu' });
      expect(tf2).toBe(tf);
    });

    it('returns the in-flight promise on concurrent calls', async () => {
      const promise1 = initializeBackend();
      const promise2 = initializeBackend();
      const [tf1, tf2] = await Promise.all([promise1, promise2]);
      expect(tf1).toBe(tf2);
    });
  });

  describe('resetBackend', () => {
    it('clears the cached instance', () => {
      ensureBackend();
      expect(isInitialized()).toBe(true);
      resetBackend();
      expect(isInitialized()).toBe(false);
    });

    it('allows re-initialization after reset', async () => {
      await initializeBackend();
      resetBackend();
      const tf = await initializeBackend();
      expect(tf).toBeDefined();
      expect(typeof tf.tensor2d).toBe('function');
    });
  });
});
