import { useState, useEffect, useCallback, useRef } from 'react';
import { HealthResponse } from '@/types';
import { apiService } from '@/services/api';

interface UseHealthReturn {
  health: HealthResponse | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useHealth(autoCheck = false): UseHealthReturn {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef<boolean>(true);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchHealth = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    if (isMountedRef.current) {
      setIsLoading(true);
      setError(null);
    }

    try {
      const data = await apiService.getHealth(controller.signal);
      if (isMountedRef.current && !controller.signal.aborted) {
        setHealth(data);
      }
    } catch (err) {
      if (controller.signal.aborted || (err as { name?: string })?.name === 'AbortError') {
        return;
      }
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to reach API');
      }
    } finally {
      if (isMountedRef.current && !controller.signal.aborted) {
        setIsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    if (autoCheck) {
      void fetchHealth();
    }
    return () => {
      isMountedRef.current = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [autoCheck, fetchHealth]);

  return { health, isLoading, error, refetch: fetchHealth };
}
