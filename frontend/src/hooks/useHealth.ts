import { useState, useEffect, useCallback } from 'react';
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

  const fetchHealth = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await apiService.getHealth();
      setHealth(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reach API');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    if (autoCheck) {
      apiService.getHealth()
        .then((data) => {
          if (mounted) {
            setHealth(data);
            setIsLoading(false);
          }
        })
        .catch((err) => {
          if (mounted) {
            setError(err instanceof Error ? err.message : 'Failed to reach API');
            setIsLoading(false);
          }
        });
    }
    return () => {
      mounted = false;
    };
  }, [autoCheck]);

  return { health, isLoading, error, refetch: fetchHealth };
}
