import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useOllamaStatus } from '../src/hooks/useOllamaStatus';
import { apiService } from '../src/services/api';
import { OllamaStatus } from '../src/types';

describe('useOllamaStatus Hook Lifecycle & Abort Regression Suite', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('aborts in-flight request and suppresses unhandled errors on unmount', async () => {
    let capturedSignal: AbortSignal | undefined;

    vi.spyOn(apiService, 'getOllamaStatus').mockImplementation((signal?: AbortSignal) => {
      capturedSignal = signal;
      return new Promise<OllamaStatus>((_, reject) => {
        signal?.addEventListener('abort', () => {
          reject(new DOMException('The user aborted a request.', 'AbortError'));
        });
      });
    });

    const { unmount } = renderHook(() => useOllamaStatus(true));

    expect(capturedSignal).toBeDefined();
    expect(capturedSignal?.aborted).toBe(false);

    // Unmount before the promise settles
    unmount();

    expect(capturedSignal?.aborted).toBe(true);
  });

  it('aborts previous in-flight request when checkStatus is invoked again', async () => {
    const signals: AbortSignal[] = [];

    vi.spyOn(apiService, 'getOllamaStatus').mockImplementation((signal?: AbortSignal) => {
      if (signal) signals.push(signal);
      return new Promise<OllamaStatus>(() => {});
    });

    const { result } = renderHook(() => useOllamaStatus(false));

    act(() => {
      void result.current.checkStatus();
    });

    expect(signals.length).toBe(1);
    expect(signals[0].aborted).toBe(false);

    act(() => {
      void result.current.checkStatus();
    });

    expect(signals.length).toBe(2);
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);
  });

  it('updates status and resets isLoading on successful fetch', async () => {
    const mockStatus: OllamaStatus = {
      connected: true,
      base_url: 'http://127.0.0.1:11434',
      model: 'qwen2.5:1.5b',
      model_installed: true,
      available_models: ['qwen2.5:1.5b'],
      error: null,
    };

    vi.spyOn(apiService, 'getOllamaStatus').mockResolvedValue(mockStatus);

    const { result } = renderHook(() => useOllamaStatus(true));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.status).toEqual(mockStatus);
    });
  });

  it('handles network error with fallback status and resets isLoading', async () => {
    vi.spyOn(apiService, 'getOllamaStatus').mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useOllamaStatus(true));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.status?.connected).toBe(false);
      expect(result.current.status?.error).toBe('Failed to contact status endpoint');
    });
  });
});
