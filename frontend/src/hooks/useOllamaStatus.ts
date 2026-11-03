import { useState, useEffect, useCallback, useRef } from 'react';
import { OllamaStatus } from '@/types';
import { apiService } from '@/services/api';

interface UseOllamaStatusReturn {
  status: OllamaStatus | null;
  isLoading: boolean;
  checkStatus: () => Promise<void>;
}

export function useOllamaStatus(autoCheck: boolean = true): UseOllamaStatusReturn {
  const [status, setStatus] = useState<OllamaStatus | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const isMountedRef = useRef<boolean>(true);
  const abortControllerRef = useRef<AbortController | null>(null);

  const checkStatus = useCallback(async () => {
    // Abort previous in-flight request if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    if (isMountedRef.current) {
      setIsLoading(true);
    }

    try {
      const data = await apiService.getOllamaStatus(controller.signal);
      if (isMountedRef.current && !controller.signal.aborted) {
        setStatus(data);
      }
    } catch (err) {
      if (controller.signal.aborted || (err as { name?: string })?.name === 'AbortError') {
        return;
      }
      if (isMountedRef.current) {
        setStatus({
          connected: false,
          base_url: 'http://127.0.0.1:11434',
          model: 'qwen2.5:1.5b',
          model_installed: false,
          available_models: [],
          error: 'Failed to contact status endpoint',
        });
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
      void checkStatus();
    }
    return () => {
      isMountedRef.current = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [autoCheck, checkStatus]);

  return { status, isLoading, checkStatus };
}
