import { useState, useEffect, useCallback } from 'react';
import { MainLLMStatus } from '@/types';
import { apiService } from '@/services/api';

interface UseMainLLMStatusReturn {
  status: MainLLMStatus | null;
  isLoading: boolean;
  checkStatus: () => Promise<void>;
}

export function useMainLLMStatus(autoCheck: boolean = true): UseMainLLMStatusReturn {
  const [status, setStatus] = useState<MainLLMStatus | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const checkStatus = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await apiService.getMainLLMStatus();
      setStatus(data);
    } catch {
      setStatus({
        provider: 'openai',
        model: 'gpt-4o',
        is_configured: false,
        base_url: 'https://api.openai.com/v1',
        message: 'Backend service offline',
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
