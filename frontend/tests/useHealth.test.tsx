import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useHealth } from '../src/hooks/useHealth';
import { apiService } from '../src/services/api';
import { HealthResponse } from '../src/types';

describe('useHealth Hook Lifecycle & Abort Suite', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('aborts in-flight health check on unmount', async () => {
    let capturedSignal: AbortSignal | undefined;

    vi.spyOn(apiService, 'getHealth').mockImplementation((signal?: AbortSignal) => {
      capturedSignal = signal;
      return new Promise<HealthResponse>((_, reject) => {
        signal?.addEventListener('abort', () => {
          reject(new DOMException('The user aborted a request.', 'AbortError'));
        });
      });
    });

    const { unmount } = renderHook(() => useHealth(true));

    expect(capturedSignal).toBeDefined();
    expect(capturedSignal?.aborted).toBe(false);

    unmount();

    expect(capturedSignal?.aborted).toBe(true);
  });

  it('aborts previous in-flight request when refetch is invoked again', async () => {
    const signals: AbortSignal[] = [];

    vi.spyOn(apiService, 'getHealth').mockImplementation((signal?: AbortSignal) => {
      if (signal) signals.push(signal);
      return new Promise<HealthResponse>(() => {});
    });

    const { result } = renderHook(() => useHealth(false));

    act(() => {
      void result.current.refetch();
    });

    expect(signals.length).toBe(1);
    expect(signals[0].aborted).toBe(false);

    act(() => {
      void result.current.refetch();
    });

    expect(signals.length).toBe(2);
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);
  });

  it('updates health data on successful fetch', async () => {
    const mockHealth: HealthResponse = {
      status: 'ok',
      app_name: 'Argus',
      version: '0.1.0',
      timestamp: new Date().toISOString(),
      environment: 'development',
    };

    vi.spyOn(apiService, 'getHealth').mockResolvedValue(mockHealth);

    const { result } = renderHook(() => useHealth(true));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.health).toEqual(mockHealth);
      expect(result.current.error).toBeNull();
    });
  });

  it('handles fetch failure and records error message', async () => {
    vi.spyOn(apiService, 'getHealth').mockRejectedValue(new Error('Connection refused'));

    const { result } = renderHook(() => useHealth(true));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.health).toBeNull();
      expect(result.current.error).toBe('Connection refused');
    });
  });
});
