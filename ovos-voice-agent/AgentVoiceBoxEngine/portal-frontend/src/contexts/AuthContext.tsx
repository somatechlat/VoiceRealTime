"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { authService, User, LoginCredentials, AuthResult, Permission } from "@/services/auth-service";

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (credentials: LoginCredentials) => Promise<AuthResult>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<boolean>;
  hasPermission: (permission: Permission) => boolean;
  hasRole: (role: string) => boolean;
  isAdmin: () => boolean;
  isPlatformAdmin: () => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Session timeout constants (from Requirements 2.5)
const IDLE_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
const MAX_SESSION_MS = 8 * 60 * 60 * 1000; // 8 hours
const REFRESH_CHECK_INTERVAL_MS = 60 * 1000; // 1 minute

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [sessionStart, setSessionStart] = useState<number | null>(null);
  const [lastActivity, setLastActivity] = useState<number>(Date.now());

  // Initialize auth state from service
  const initializeAuth = useCallback(async () => {
    const currentUser = authService.getCurrentUser();
    
    if (currentUser) {
      // Check if token needs refresh
      if (authService.isTokenExpired()) {
        const refreshed = await authService.refreshAccessToken();
        if (refreshed) {
          setUser(authService.getCurrentUser());
          setSessionStart(Date.now());
        } else {
          setUser(null);
        }
      } else {
        setUser(currentUser);
        setSessionStart(Date.now());
      }
    }
    
    setIsLoading(false);
  }, []);

  useEffect(() => {
    initializeAuth();
  }, [initializeAuth]);

  // Track user activity for idle timeout
  useEffect(() => {
    if (!user) return;

    const updateActivity = () => {
      setLastActivity(Date.now());
    };

    // Track user interactions
    const events = ['mousedown', 'keydown', 'scroll', 'touchstart'];
    events.forEach(event => {
      window.addEventListener(event, updateActivity, { passive: true });
    });

    return () => {
      events.forEach(event => {
        window.removeEventListener(event, updateActivity);
      });
    };
  }, [user]);

  // Session timeout and token refresh
  useEffect(() => {
    if (!user || !sessionStart) return;

    const checkSession = async () => {
      const now = Date.now();

      // Check max session duration (8 hours)
      if (now - sessionStart >= MAX_SESSION_MS) {
        console.log('Max session duration exceeded, logging out');
        await logout();
        return;
      }

      // Check idle timeout (30 minutes)
      if (now - lastActivity >= IDLE_TIMEOUT_MS) {
        console.log('Idle timeout exceeded, logging out');
        await logout();
        return;
      }

      // Refresh token if needed (1 minute before expiry)
      if (authService.isTokenExpired(60000)) {
        const refreshed = await authService.refreshAccessToken();
        if (!refreshed) {
          await logout();
        }
      }
    };

    const interval = setInterval(checkSession, REFRESH_CHECK_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [user, sessionStart, lastActivity]);

  const login = useCallback(async (credentials: LoginCredentials): Promise<AuthResult> => {
    const result = await authService.login(credentials);
    
    if (result.success && result.user) {
      setUser(result.user);
      setSessionStart(Date.now());
      setLastActivity(Date.now());
    }
    
    return result;
  }, []);

  const logout = useCallback(async () => {
    await authService.logout();
    setUser(null);
    setSessionStart(null);
  }, []);

  const refreshSession = useCallback(async (): Promise<boolean> => {
    const success = await authService.refreshAccessToken();
    if (success) {
      setUser(authService.getCurrentUser());
      setLastActivity(Date.now());
    } else {
      setUser(null);
      setSessionStart(null);
    }
    return success;
  }, []);

  const hasPermission = useCallback((permission: Permission): boolean => {
    return authService.hasPermission(permission);
  }, []);

  const hasRole = useCallback((role: string): boolean => {
    return authService.hasRole(role);
  }, []);

  const isAdmin = useCallback((): boolean => {
    return authService.isAdmin();
  }, []);

  const isPlatformAdmin = useCallback((): boolean => {
    return authService.isPlatformAdmin();
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        logout,
        refreshSession,
        hasPermission,
        hasRole,
        isAdmin,
        isPlatformAdmin,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

// Re-export types for convenience
export type { User, LoginCredentials, AuthResult, Permission };
