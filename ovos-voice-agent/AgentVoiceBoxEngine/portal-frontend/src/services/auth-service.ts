/**
 * Authentication Service
 * Connects to Keycloak for OIDC authentication
 * Implements Requirements 2.1, 2.2, 2.3, 2.4
 */

import { apiClient } from './api-client';

// Types matching backend UserContext
export interface User {
  id: string;
  tenantId: string;
  email: string;
  username: string;
  roles: string[];
  permissions: string[];
  mfaEnabled?: boolean;
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface AuthResult {
  success: boolean;
  user?: User;
  accessToken?: string;
  refreshToken?: string;
  expiresAt?: number;
  error?: string;
  requiresMfa?: boolean;
  mfaToken?: string;
}

export interface MfaVerifyRequest {
  mfaToken: string;
  code: string;
}

export interface TokenPair {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

// Customer portal roles
export type CustomerRole = 'owner' | 'admin' | 'developer' | 'billing' | 'viewer';

// Admin portal roles
export type AdminRole = 'super_admin' | 'tenant_admin' | 'support_agent' | 'billing_admin' | 'viewer';

// Permission types
export type Permission =
  | 'team:manage' | 'team:view'
  | 'api_keys:create' | 'api_keys:rotate' | 'api_keys:revoke' | 'api_keys:view'
  | 'billing:manage' | 'billing:view'
  | 'usage:view'
  | 'settings:manage'
  | 'tenant:manage' | 'tenant:view' | 'tenant:delete'
  | 'impersonate:user'
  | 'system:configure';

// Role to permissions mapping
const ROLE_PERMISSIONS: Record<string, Permission[]> = {
  owner: [
    'team:manage', 'team:view',
    'api_keys:create', 'api_keys:rotate', 'api_keys:revoke', 'api_keys:view',
    'billing:manage', 'billing:view',
    'usage:view',
    'settings:manage',
  ],
  admin: [
    'team:manage', 'team:view',
    'api_keys:create', 'api_keys:rotate', 'api_keys:revoke', 'api_keys:view',
    'billing:view',
    'usage:view',
  ],
  developer: [
    'api_keys:create', 'api_keys:rotate', 'api_keys:view',
    'usage:view',
  ],
  billing: [
    'billing:manage', 'billing:view',
  ],
  viewer: [
    'usage:view',
  ],
  super_admin: [
    'tenant:manage', 'tenant:view', 'tenant:delete',
    'impersonate:user',
    'system:configure',
    'billing:manage', 'billing:view',
  ],
  tenant_admin: [
    'tenant:manage', 'tenant:view',
  ],
  support_agent: [
    'tenant:view',
    'impersonate:user',
  ],
  billing_admin: [
    'billing:manage', 'billing:view',
  ],
};

// Storage keys
const ACCESS_TOKEN_KEY = 'agentvoicebox_access_token';
const REFRESH_TOKEN_KEY = 'agentvoicebox_refresh_token';
const USER_KEY = 'agentvoicebox_user';
const EXPIRES_AT_KEY = 'agentvoicebox_expires_at';

class AuthService {
  private user: User | null = null;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;
  private expiresAt: number | null = null;
  private refreshPromise: Promise<boolean> | null = null;

  constructor() {
    // Initialize from storage on client side
    if (typeof window !== 'undefined') {
      this.loadFromStorage();
    }
  }

  private loadFromStorage(): void {
    try {
      const accessToken = localStorage.getItem(ACCESS_TOKEN_KEY);
      const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
      const userJson = localStorage.getItem(USER_KEY);
      const expiresAt = localStorage.getItem(EXPIRES_AT_KEY);

      if (accessToken && userJson) {
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
        this.user = JSON.parse(userJson);
        this.expiresAt = expiresAt ? parseInt(expiresAt, 10) : null;

        // Set token in API client
        apiClient.setAuthToken(accessToken);
      }
    } catch (error) {
      console.error('Failed to load auth from storage:', error);
      this.clearStorage();
    }
  }

  private saveToStorage(): void {
    if (typeof window === 'undefined') return;

    try {
      if (this.accessToken) {
        localStorage.setItem(ACCESS_TOKEN_KEY, this.accessToken);
      }
      if (this.refreshToken) {
        localStorage.setItem(REFRESH_TOKEN_KEY, this.refreshToken);
      }
      if (this.user) {
        localStorage.setItem(USER_KEY, JSON.stringify(this.user));
      }
      if (this.expiresAt) {
        localStorage.setItem(EXPIRES_AT_KEY, this.expiresAt.toString());
      }
    } catch (error) {
      console.error('Failed to save auth to storage:', error);
    }
  }

  private clearStorage(): void {
    if (typeof window === 'undefined') return;

    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem(EXPIRES_AT_KEY);
  }

  /**
   * Login with email and password
   */
  async login(credentials: LoginCredentials): Promise<AuthResult> {
    try {
      const response = await apiClient.post<{
        access_token: string;
        refresh_token: string;
        expires_in: number;
        user: User;
        requires_mfa?: boolean;
        mfa_token?: string;
      }>('/api/auth/login', {
        email: credentials.email,
        password: credentials.password,
        remember_me: credentials.rememberMe,
      }, { retry: false });

      const data = response.data;

      // Check if MFA is required
      if (data.requires_mfa) {
        return {
          success: false,
          requiresMfa: true,
          mfaToken: data.mfa_token,
        };
      }

      // Store tokens and user
      this.accessToken = data.access_token;
      this.refreshToken = data.refresh_token;
      this.user = data.user;
      this.expiresAt = Date.now() + (data.expires_in * 1000);

      // Set token in API client
      apiClient.setAuthToken(this.accessToken);

      // Save to storage
      this.saveToStorage();

      return {
        success: true,
        user: this.user,
        accessToken: this.accessToken,
        refreshToken: this.refreshToken,
        expiresAt: this.expiresAt,
      };
    } catch (error) {
      const message = (error as { message?: string }).message || 'Login failed';
      return {
        success: false,
        error: message,
      };
    }
  }

  /**
   * Verify MFA code
   */
  async verifyMfa(request: MfaVerifyRequest): Promise<AuthResult> {
    try {
      const response = await apiClient.post<{
        access_token: string;
        refresh_token: string;
        expires_in: number;
        user: User;
      }>('/api/auth/mfa/verify', {
        mfa_token: request.mfaToken,
        code: request.code,
      }, { retry: false });

      const data = response.data;

      // Store tokens and user
      this.accessToken = data.access_token;
      this.refreshToken = data.refresh_token;
      this.user = data.user;
      this.expiresAt = Date.now() + (data.expires_in * 1000);

      // Set token in API client
      apiClient.setAuthToken(this.accessToken);

      // Save to storage
      this.saveToStorage();

      return {
        success: true,
        user: this.user,
        accessToken: this.accessToken,
        refreshToken: this.refreshToken,
        expiresAt: this.expiresAt,
      };
    } catch (error) {
      const message = (error as { message?: string }).message || 'MFA verification failed';
      return {
        success: false,
        error: message,
      };
    }
  }

  /**
   * Logout and clear session
   */
  async logout(): Promise<void> {
    try {
      if (this.accessToken) {
        await apiClient.post('/api/auth/logout', {}, { retry: false });
      }
    } catch (error) {
      console.error('Logout API call failed:', error);
    } finally {
      this.accessToken = null;
      this.refreshToken = null;
      this.user = null;
      this.expiresAt = null;
      apiClient.clearAuthToken();
      this.clearStorage();
    }
  }

  /**
   * Refresh access token
   */
  async refreshAccessToken(): Promise<boolean> {
    // Prevent concurrent refresh calls
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    if (!this.refreshToken) {
      return false;
    }

    this.refreshPromise = this._doRefresh();
    const result = await this.refreshPromise;
    this.refreshPromise = null;
    return result;
  }

  private async _doRefresh(): Promise<boolean> {
    try {
      const response = await apiClient.post<{
        access_token: string;
        refresh_token: string;
        expires_in: number;
      }>('/api/auth/refresh', {
        refresh_token: this.refreshToken,
      }, { retry: false });

      const data = response.data;

      this.accessToken = data.access_token;
      this.refreshToken = data.refresh_token;
      this.expiresAt = Date.now() + (data.expires_in * 1000);

      apiClient.setAuthToken(this.accessToken);
      this.saveToStorage();

      return true;
    } catch (error) {
      console.error('Token refresh failed:', error);
      await this.logout();
      return false;
    }
  }

  /**
   * Get current user
   */
  getCurrentUser(): User | null {
    return this.user;
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!this.accessToken && !!this.user;
  }

  /**
   * Check if token is expired or about to expire
   */
  isTokenExpired(bufferMs: number = 60000): boolean {
    if (!this.expiresAt) return true;
    return Date.now() >= (this.expiresAt - bufferMs);
  }

  /**
   * Check if user has a specific role
   */
  hasRole(role: string): boolean {
    return this.user?.roles.includes(role) ?? false;
  }

  /**
   * Check if user has a specific permission
   */
  hasPermission(permission: Permission): boolean {
    if (!this.user) return false;

    // Check direct permissions
    if (this.user.permissions.includes(permission)) {
      return true;
    }

    // Check role-based permissions
    for (const role of this.user.roles) {
      const rolePermissions = ROLE_PERMISSIONS[role];
      if (rolePermissions?.includes(permission)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Get all effective permissions for current user
   */
  getEffectivePermissions(): Permission[] {
    if (!this.user) return [];

    const permissions = new Set<Permission>(this.user.permissions as Permission[]);

    // Add role-based permissions
    for (const role of this.user.roles) {
      const rolePermissions = ROLE_PERMISSIONS[role];
      if (rolePermissions) {
        rolePermissions.forEach(p => permissions.add(p));
      }
    }

    return Array.from(permissions);
  }

  /**
   * Check if user is admin (customer portal)
   */
  isAdmin(): boolean {
    return this.hasRole('owner') || this.hasRole('admin');
  }

  /**
   * Check if user is platform admin (admin portal)
   */
  isPlatformAdmin(): boolean {
    return this.hasRole('super_admin') || this.hasRole('tenant_admin');
  }

  /**
   * Get access token (for manual API calls)
   */
  getAccessToken(): string | null {
    return this.accessToken;
  }
}

// Export singleton instance
export const authService = new AuthService();
