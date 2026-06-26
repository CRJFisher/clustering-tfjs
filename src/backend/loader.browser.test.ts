function set_global(key: string, value: unknown): void {
  (globalThis as Record<string, unknown>)[key] = value;
}

function delete_global(key: string): void {
  delete (globalThis as Record<string, unknown>)[key];
}

interface FakeTfOptions {
  // Models setBackend resolving false: the backend registered but failed to init.
  set_ok?: boolean;
  // Models a silent fallback: getBackend reports a different already-registered
  // backend than the one requested even though setBackend resolved true.
  active_after_set?: string;
}

interface MockState {
  imported: string[];
  set_backend_calls: string[];
  flag_calls: Record<string, unknown>[];
}

function mock_tensorflow(options: FakeTfOptions = {}): MockState {
  const state: MockState = { imported: [], set_backend_calls: [], flag_calls: [] };
  const set_ok = options.set_ok ?? true;
  let active = '';

  const set_flags = (flags: Record<string, unknown>): void => {
    state.flag_calls.push(flags);
  };
  const set_backend = async (name: string): Promise<boolean> => {
    state.set_backend_calls.push(name);
    if (set_ok) {
      active = options.active_after_set ?? name;
    }
    return set_ok;
  };
  const get_backend = (): string => active;
  const ready = async (): Promise<void> => {};
  const env = () => ({ setFlags: set_flags });

  const core: Record<string, unknown> = { env, setBackend: set_backend, getBackend: get_backend, ready };
  jest.doMock('@tensorflow/tfjs-core', () => core);

  for (const name of ['cpu', 'webgl', 'webgpu', 'wasm']) {
    jest.doMock(`@tensorflow/tfjs-backend-${name}`, () => {
      state.imported.push(name);
      return {};
    });
  }

  return state;
}

async function load_loader() {
  return import('./loader.browser');
}

afterEach(() => {
  jest.resetModules();
  delete_global('navigator');
  delete_global('window');
});

describe('load_tensor_flow backend selection', () => {
  it('imports only the cpu backend package when cpu is requested', async () => {
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await load_tensor_flow('cpu');

    expect(state.imported).toEqual(['cpu']);
    expect(state.set_backend_calls).toEqual(['cpu']);
  });

  it('imports only the webgl backend package when webgl is requested', async () => {
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await load_tensor_flow('webgl');

    expect(state.imported).toEqual(['webgl']);
  });

  it('imports only the wasm backend package when wasm is requested', async () => {
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await load_tensor_flow('wasm');

    expect(state.imported).toEqual(['wasm']);
    expect(state.set_backend_calls).toEqual(['wasm']);
  });

  it('defaults to webgl when no backend is requested', async () => {
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await load_tensor_flow();

    expect(state.imported).toEqual(['webgl']);
    expect(state.set_backend_calls).toEqual(['webgl']);
  });

  it('rejects a backend that has no browser package without importing anything', async () => {
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await expect(load_tensor_flow('node')).rejects.toThrow(/Unsupported browser backend 'node'/);
    expect(state.imported).toEqual([]);
  });

  it('applies requested flags before selecting the backend', async () => {
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await load_tensor_flow('cpu', { WEBGL_PACK: false });

    expect(state.flag_calls).toEqual([{ WEBGL_PACK: false }]);
  });

  it('throws an install hint when tfjs-core cannot be loaded', async () => {
    jest.doMock('@tensorflow/tfjs-core', () => {
      throw new Error('Cannot find module');
    });
    const { load_tensor_flow } = await load_loader();

    await expect(load_tensor_flow('cpu')).rejects.toThrow(/Install @tensorflow\/tfjs-core/);
  });
});

describe('load_tensor_flow webgpu feature detection', () => {
  it('rejects cleanly when navigator.gpu is absent without importing the webgpu package', async () => {
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await expect(load_tensor_flow('webgpu')).rejects.toThrow(/WebGPU is not available/);
    expect(state.imported).toEqual([]);
    expect(state.set_backend_calls).toEqual([]);
  });

  it('initializes webgpu when navigator.gpu is present and the backend activates', async () => {
    set_global('navigator', { gpu: {} });
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    const tf = await load_tensor_flow('webgpu');

    expect(state.imported).toEqual(['webgpu']);
    expect(state.set_backend_calls).toEqual(['webgpu']);
    expect(tf.getBackend()).toBe('webgpu');
  });
});

describe('load_tensor_flow getBackend verification', () => {
  it('rejects when setBackend reports failure', async () => {
    set_global('navigator', { gpu: {} });
    mock_tensorflow({ set_ok: false });
    const { load_tensor_flow } = await load_loader();

    await expect(load_tensor_flow('webgpu')).rejects.toThrow(/Failed to initialize the 'webgpu'/);
  });

  it('rejects when the backend silently falls back to a different backend', async () => {
    set_global('navigator', { gpu: {} });
    mock_tensorflow({ active_after_set: 'webgl' });
    const { load_tensor_flow } = await load_loader();

    await expect(load_tensor_flow('webgpu')).rejects.toThrow(/active backend is 'webgl'/);
  });
});

describe('load_tensor_flow script-tag escape hatch', () => {
  function set_window_tf(active_backend: string): void {
    const get_backend = (): string => active_backend;
    set_global('window', { tf: { getBackend: get_backend } });
  }

  it('returns a globally provided tf without importing a backend package', async () => {
    set_window_tf('webgl');
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    const tf = await load_tensor_flow();

    expect(tf.getBackend()).toBe('webgl');
    expect(state.imported).toEqual([]);
  });

  it('returns the global tf when the requested backend matches its active backend', async () => {
    set_window_tf('cpu');
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await load_tensor_flow('cpu');

    expect(state.imported).toEqual([]);
  });

  it('loads the requested backend when it differs from the global tf active backend', async () => {
    set_window_tf('webgl');
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await load_tensor_flow('cpu');

    expect(state.imported).toEqual(['cpu']);
    expect(state.set_backend_calls).toEqual(['cpu']);
  });

  it('ignores a global tf that has no active backend', async () => {
    set_window_tf('');
    const state = mock_tensorflow();
    const { load_tensor_flow } = await load_loader();

    await load_tensor_flow('webgl');

    expect(state.imported).toEqual(['webgl']);
  });
});
