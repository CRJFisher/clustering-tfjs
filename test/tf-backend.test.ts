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

    it('respects backend switch when already initialized', async () => {
      const tf = await initializeBackend();
      const currentBackend = tf.getBackend();
      expect(currentBackend).toBeDefined();

      // Switching to cpu should work
      const tf2 = await initializeBackend({ backend: 'cpu' });
      expect(tf2).toBe(tf);
      expect(tf2.getBackend()).toBe('cpu');
    });
  });

  describe('resetBackend', () => {
    it('clears the cached instance', () => {
      ensureBackend();
      expect(isInitialized()).toBe(true);
      resetBackend();
      expect(isInitialized()).toBe(false);
    });
  });
});
