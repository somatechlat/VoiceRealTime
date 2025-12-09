"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import {
  User,
  extractUserFromToken,
  getStoredTokens,
  storeTokens,
  clearTokens,
  isTokenExpired,
  refreshAccessToken,
  getLoginUrl,
  getLogoutUrl,
} from "@/lib/auth";

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (returnUrl?: string) => void;
  logout: () => void;
  refreshSession: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const initializeAuth = useCallback(async () => {
    const { accessToken, refreshToken } = getStoredTokens();

    if (!accessToken) {
      setIsLoading(false);
      return;
    }

    // Check if token is expired
    if (isTokenExpired(accessToken)) {
      if (refreshToken && !isTokenExpired(refreshToken, 300)) {
        try {
          const tokens = await refreshAccessToken(refreshToken);
          storeTokens(tokens);
          const extractedUser = extractUserFromToken(tokens.access_token);
          setUser(extractedUser);
        } catch {
          clearTokens();
        }
      } else {
        clearTokens();
      }
    } else {
      const extractedUser = extractUserFromToken(accessToken);
      setUser(extractedUser);
    }

    setIsLoading(false);
  }, []);

  useEffect(() => {
    initializeAuth();
  }, [initializeAuth]);

  // Set up token refresh interval
  useEffect(() => {
    if (!user) return;

    const interval = setInterval(async () => {
      const { accessToken, refreshToken } = getStoredTokens();
      if (accessToken && isTokenExpired(accessToken, 120) && refreshToken) {
        try {
          const tokens = await refreshAccessToken(refreshToken);
          storeTokens(tokens);
          const extractedUser = extractUserFromToken(tokens.access_token);
          setUser(extractedUser);
        } catch {
          clearTokens();
          setUser(null);
        }
      }
    }, 60000); // Check every minute

    return () => clearInterval(interval);
  }, [user]);

  const login = useCallback((returnUrl?: string) => {
    const state = returnUrl ? btoa(JSON.stringify({ returnUrl })) : undefined;
    window.location.href = getLoginUrl(state);
  }, []);

  const logout = useCallback(() => {
    clearTokens();
    setUser(null);
    window.location.href = getLogoutUrl();
  }, []);

  const refreshSession = useCallback(async (): Promise<boolean> => {
    const { refreshToken } = getStoredTokens();
    if (!refreshToken) return false;

    try {
      const tokens = await refreshAccessToken(refreshToken);
      storeTokens(tokens);
      const extractedUser = extractUserFromToken(tokens.access_token);
      setUser(extractedUser);
      return true;
    } catch {
      clearTokens();
      setUser(null);
      return false;
    }
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
