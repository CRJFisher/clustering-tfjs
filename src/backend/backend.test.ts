import { ensure_backend, reset_backend, is_initialized, initialize_backend } from './backend';

describe('tf-backend', () => {
  afterEach(() => {
    reset_backend();
  });

  describe('ensureBackend', () => {
    it('auto-loads a backend in Node.js without explicit init', () => {
      const tf = ensure_backend();
      expect(tf).toBeDefined();
      expect(typeof tf.tensor2d).toBe('function');
      expect(typeof tf.tidy).toBe('function');
    });

    it('returns the same instance on subsequent calls', () => {
      const tf1 = ensure_backend();
      const tf2 = ensure_backend();
      expect(tf1).toBe(tf2);
    });

    it('sets isInitialized to true after auto-load', () => {
      expect(is_initialized()).toBe(false);
      ensure_backend();
      expect(is_initialized()).toBe(true);
    });

    it('throws if async init is in progress (race condition prevention)', async () => {
      // Start async init but don't await it yet.
      // initialize_backend sets initialization_promise synchronously before awaiting.
      const init_promise = initialize_backend({ backend: 'cpu' });

      // At this point initialization_promise is set but tf_instance is still null
      // (the promise hasn't resolved yet). A synchronous ensure_backend() call
      // should detect the in-flight promise and throw rather than silently
      // loading a different backend via load_backend_sync().
      expect(() => ensure_backend()).toThrow(
        'TensorFlow.js is being initialized asynchronously'
      );

      // Clean up: let the init complete
      await init_promise;
    });
  });

  describe('initializeBackend', () => {
    it('initializes and returns a tf instance', async () => {
      const tf = await initialize_backend();
      expect(tf).toBeDefined();
      expect(typeof tf.tensor2d).toBe('function');
    });

    it('returns cached instance on subsequent calls', async () => {
      const tf1 = await initialize_backend();
      const tf2 = await initialize_backend();
      expect(tf1).toBe(tf2);
    });

    it('returns the cached instance regardless of config after initialization', async () => {
      const tf = await initialize_backend();
      // Config conflict detection is handled at the Clustering.init() level,
      // not at the initialize_backend() level. Once initialized, initialize_backend
      // always returns the cached instance.
      const tf2 = await initialize_backend({ backend: 'cpu' });
      expect(tf2).toBe(tf);
    });

    it('returns the in-flight promise on concurrent calls', async () => {
      const promise1 = initialize_backend();
      const promise2 = initialize_backend();
      const [tf1, tf2] = await Promise.all([promise1, promise2]);
      expect(tf1).toBe(tf2);
    });
  });

  describe('resetBackend', () => {
    it('clears the cached instance', () => {
      ensure_backend();
      expect(is_initialized()).toBe(true);
      reset_backend();
      expect(is_initialized()).toBe(false);
    });

    it('allows re-initialization after reset', async () => {
      await initialize_backend();
      reset_backend();
      const tf = await initialize_backend();
      expect(tf).toBeDefined();
      expect(typeof tf.tensor2d).toBe('function');
    });
  });
});
