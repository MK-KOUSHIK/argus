import { useState, useEffect, useCallback } from 'react';
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

  const checkStatus = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await apiService.getOllamaStatus();
      setStatus(data);
    } catch {
      setStatus({
        connected: false,
        base_url: 'http://127.0.0.1:11434',
        model: 'qwen2.5:1.5b',
        model_installed: false,
        available_models: [],
        error: 'Failed to contact status endpoint',
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (autoCheck) {
      void checkStatus();
    }
  }, [autoCheck, checkStatus]);

  return { status, isLoading, checkStatus };
}
