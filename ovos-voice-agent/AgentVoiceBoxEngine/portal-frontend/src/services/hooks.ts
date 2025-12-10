/**
 * SWR Data Fetching Hooks
 * Implements Requirements 20.5, 20.8: SWR caching and optimistic updates
 */

import useSWR, { SWRConfiguration, mutate } from 'swr';
import { apiClient, ApiResponse } from './api-client';

// Default SWR configuration
const defaultConfig: SWRConfiguration = {
  revalidateOnFocus: true,
  revalidateOnReconnect: true,
  dedupingInterval: 2000,
  errorRetryCount: 3,
};

/**
 * Generic fetcher for SWR
 */
const fetcher = async <T>(url: string): Promise<T> => {
  const response = await apiClient.get<T>(url);
  return response.data;
};

/**
 * Hook for fetching data with SWR (stale-while-revalidate)
 */
export function useFetch<T>(
  url: string | null,
  config?: SWRConfiguration<T>
) {
  const { data, error, isLoading, isValidating, mutate: boundMutate } = useSWR<T>(
    url,
    fetcher,
    { ...defaultConfig, ...config }
  );

  return {
    data,
    error,
    isLoading,
    isValidating,
    isError: !!error,
    mutate: boundMutate,
  };
}

/**
 * Hook for fetching paginated data
 */
export function usePaginatedFetch<T>(
  baseUrl: string,
  cursor?: string | null,
  config?: SWRConfiguration<{ data: T[]; cursor: string | null; hasMore: boolean }>
) {
  const url = cursor ? `${baseUrl}?cursor=${cursor}` : baseUrl;
  return useFetch<{ data: T[]; cursor: string | null; hasMore: boolean }>(url, config);
}

/**
 * Optimistic update helper
 * Updates the cache immediately, then revalidates
 */
export async function optimisticUpdate<T>(
  key: string,
  updateFn: (current: T | undefined) => T,
  apiCall: () => Promise<ApiResponse<T>>
): Promise<T> {
  // Optimistically update the cache
  await mutate(
    key,
    async (current: T | undefined) => {
      return updateFn(current);
    },
    { revalidate: false }
  );

  try {
    // Make the actual API call
    const response = await apiCall();
    
    // Update cache with actual response
    await mutate(key, response.data, { revalidate: false });
    
    return response.data;
  } catch (error) {
    // Revalidate to restore correct state on error
    await mutate(key);
    throw error;
  }
}

/**
 * Hook for mutations with optimistic updates
 */
export function useMutation<TData, TVariables>(
  mutationFn: (variables: TVariables) => Promise<ApiResponse<TData>>,
  options?: {
    onSuccess?: (data: TData) => void;
    onError?: (error: Error) => void;
    invalidateKeys?: string[];
  }
) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const mutate = async (variables: TVariables): Promise<TData | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await mutationFn(variables);
      
      // Invalidate related cache keys
      if (options?.invalidateKeys) {
        await Promise.all(
          options.invalidateKeys.map(key => globalMutate(key))
        );
      }

      options?.onSuccess?.(response.data);
      return response.data;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      options?.onError?.(error);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return { mutate, isLoading, error };
}

// Import useState for useMutation hook
import { useState } from 'react';
import { mutate as globalMutate } from 'swr';

/**
 * Prefetch data for a URL
 */
export async function prefetch<T>(url: string): Promise<void> {
  await mutate(url, fetcher<T>(url));
}

/**
 * Invalidate cache for a URL pattern
 */
export async function invalidateCache(urlPattern: string | RegExp): Promise<void> {
  await mutate(
    (key) => {
      if (typeof key !== 'string') return false;
      if (typeof urlPattern === 'string') {
        return key.startsWith(urlPattern);
      }
      return urlPattern.test(key);
    },
    undefined,
    { revalidate: true }
  );
}

/**
 * Clear all cache
 */
export async function clearCache(): Promise<void> {
  await mutate(() => true, undefined, { revalidate: false });
}
