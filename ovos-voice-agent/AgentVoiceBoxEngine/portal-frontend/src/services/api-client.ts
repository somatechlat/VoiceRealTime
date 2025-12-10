/**
 * API Client with retry logic and exponential backoff
 * Implements Requirements 20.4: Retry with exponential backoff (3 attempts, 1s/2s/4s delays)
 */

export interface RequestOptions {
  retry?: boolean;
  maxRetries?: number;
  timeout?: number;
  headers?: Record<string, string>;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  headers: Headers;
}

export interface ApiError {
  message: string;
  status: number;
  code?: string;
  retryable: boolean;
}

const DEFAULT_RETRY_DELAYS = [1000, 2000, 4000]; // 1s, 2s, 4s
const DEFAULT_TIMEOUT = 30000; // 30 seconds

// Status codes that should trigger a retry
const RETRYABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504];

/**
 * Sleep for a specified duration
 */
const sleep = (ms: number): Promise<void> => 
  new Promise(resolve => setTimeout(resolve, ms));

/**
 * Check if an error is retryable
 */
const isRetryable = (status: number): boolean => 
  RETRYABLE_STATUS_CODES.includes(status);

/**
 * Create an API error from a response
 */
const createApiError = async (response: Response): Promise<ApiError> => {
  let message = `HTTP ${response.status}: ${response.statusText}`;
  let code: string | undefined;

  try {
    const body = await response.json();
    message = body.message || body.error || message;
    code = body.code;
  } catch {
    // Response body is not JSON
  }

  return {
    message,
    status: response.status,
    code,
    retryable: isRetryable(response.status),
  };
};

/**
 * Execute a fetch request with retry logic
 */
async function fetchWithRetry<T>(
  url: string,
  options: RequestInit & RequestOptions
): Promise<ApiResponse<T>> {
  const { retry = true, maxRetries = 3, timeout = DEFAULT_TIMEOUT, ...fetchOptions } = options;
  const retryDelays = DEFAULT_RETRY_DELAYS.slice(0, maxRetries);
  
  let lastError: ApiError | null = null;
  let attempts = 0;

  while (attempts <= (retry ? retryDelays.length : 0)) {
    try {
      // Create abort controller for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(url, {
        ...fetchOptions,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await createApiError(response);
        
        // If not retryable or retry disabled, throw immediately
        if (!retry || !error.retryable || attempts >= retryDelays.length) {
          throw error;
        }
        
        lastError = error;
        attempts++;
        await sleep(retryDelays[attempts - 1]);
        continue;
      }

      const data = await response.json() as T;
      return {
        data,
        status: response.status,
        headers: response.headers,
      };
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        lastError = {
          message: 'Request timeout',
          status: 408,
          code: 'TIMEOUT',
          retryable: true,
        };
      } else if ((error as ApiError).status) {
        lastError = error as ApiError;
      } else {
        lastError = {
          message: (error as Error).message || 'Network error',
          status: 0,
          code: 'NETWORK_ERROR',
          retryable: true,
        };
      }

      // If retry disabled or max retries reached, throw
      if (!retry || attempts >= retryDelays.length) {
        throw lastError;
      }

      attempts++;
      await sleep(retryDelays[attempts - 1]);
    }
  }

  throw lastError || { message: 'Unknown error', status: 0, retryable: false };
}

/**
 * API Client class
 */
export class ApiClient {
  private baseUrl: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseUrl: string = '', defaultHeaders: Record<string, string> = {}) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...defaultHeaders,
    };
  }

  private buildUrl(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  private buildHeaders(customHeaders?: Record<string, string>): Record<string, string> {
    return {
      ...this.defaultHeaders,
      ...customHeaders,
    };
  }

  async get<T>(path: string, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    return fetchWithRetry<T>(this.buildUrl(path), {
      method: 'GET',
      headers: this.buildHeaders(options.headers),
      ...options,
    });
  }

  async post<T>(path: string, data: unknown, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    return fetchWithRetry<T>(this.buildUrl(path), {
      method: 'POST',
      headers: this.buildHeaders(options.headers),
      body: JSON.stringify(data),
      ...options,
    });
  }

  async put<T>(path: string, data: unknown, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    return fetchWithRetry<T>(this.buildUrl(path), {
      method: 'PUT',
      headers: this.buildHeaders(options.headers),
      body: JSON.stringify(data),
      ...options,
    });
  }

  async patch<T>(path: string, data: unknown, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    return fetchWithRetry<T>(this.buildUrl(path), {
      method: 'PATCH',
      headers: this.buildHeaders(options.headers),
      body: JSON.stringify(data),
      ...options,
    });
  }

  async delete<T>(path: string, options: RequestOptions = {}): Promise<ApiResponse<T>> {
    return fetchWithRetry<T>(this.buildUrl(path), {
      method: 'DELETE',
      headers: this.buildHeaders(options.headers),
      ...options,
    });
  }

  /**
   * Set authorization header
   */
  setAuthToken(token: string): void {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`;
  }

  /**
   * Remove authorization header
   */
  clearAuthToken(): void {
    delete this.defaultHeaders['Authorization'];
  }
}

// Export singleton instance
export const apiClient = new ApiClient(process.env.NEXT_PUBLIC_API_URL || '');

// Export retry delays for testing
export const RETRY_DELAYS = DEFAULT_RETRY_DELAYS;
