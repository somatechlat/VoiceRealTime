/**
 * Login Page Unit Tests
 * Tests for form validation, error display, and loading states
 * Implements Requirements 2.1-2.3
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

// Mock next/navigation
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
  }),
  useSearchParams: () => ({
    get: vi.fn().mockReturnValue(null),
  }),
}));

// Mock auth context
const mockLogin = vi.fn();
vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => ({
    login: mockLogin,
    isLoading: false,
  }),
}));

// Mock auth service
vi.mock('@/services/auth-service', () => ({
  authService: {
    verifyMfa: vi.fn(),
  },
}));

// Mock theme context
vi.mock('@/contexts/ThemeContext', () => ({
  useTheme: () => ({
    theme: 'dark',
    resolvedTheme: 'dark',
    setTheme: vi.fn(),
  }),
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Import after mocks
import LoginPage from '@/app/login/page';

// Helper to get password input specifically
const getPasswordInput = () => screen.getByPlaceholderText('••••••••');
const getEmailInput = () => screen.getByPlaceholderText('you@company.com');

describe('Login Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLogin.mockResolvedValue({ success: true });
  });

  describe('Form Rendering', () => {
    it('should render email input', () => {
      render(<LoginPage />);
      expect(getEmailInput()).toBeInTheDocument();
    });

    it('should render password input', () => {
      render(<LoginPage />);
      expect(getPasswordInput()).toBeInTheDocument();
    });

    it('should render remember me checkbox', () => {
      render(<LoginPage />);
      expect(screen.getByRole('checkbox')).toBeInTheDocument();
    });

    it('should render sign in button', () => {
      render(<LoginPage />);
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    });

    it('should render forgot password link', () => {
      render(<LoginPage />);
      expect(screen.getByText(/forgot password/i)).toBeInTheDocument();
    });

    it('should render social login buttons', () => {
      render(<LoginPage />);
      expect(screen.getByRole('button', { name: /google/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /github/i })).toBeInTheDocument();
    });

    it('should render sign up link', () => {
      render(<LoginPage />);
      expect(screen.getByText(/sign up/i)).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('should require email field', () => {
      render(<LoginPage />);
      expect(getEmailInput()).toHaveAttribute('required');
    });

    it('should require password field', () => {
      render(<LoginPage />);
      expect(getPasswordInput()).toHaveAttribute('required');
    });

    it('should have email type on email input', () => {
      render(<LoginPage />);
      expect(getEmailInput()).toHaveAttribute('type', 'email');
    });
  });

  describe('Password Toggle', () => {
    it('should toggle password visibility', async () => {
      const user = userEvent.setup();
      render(<LoginPage />);
      
      const passwordInput = getPasswordInput();
      const toggleButton = screen.getByRole('button', { name: /show password/i });
      
      // Initially password type
      expect(passwordInput).toHaveAttribute('type', 'password');
      
      // Click to show
      await user.click(toggleButton);
      expect(passwordInput).toHaveAttribute('type', 'text');
      
      // Click to hide
      await user.click(screen.getByRole('button', { name: /hide password/i }));
      expect(passwordInput).toHaveAttribute('type', 'password');
    });
  });

  describe('Form Submission', () => {
    it('should call login with credentials on submit', async () => {
      const user = userEvent.setup();
      render(<LoginPage />);
      
      await user.type(getEmailInput(), 'test@example.com');
      await user.type(getPasswordInput(), 'password123');
      await user.click(screen.getByRole('button', { name: /sign in/i }));
      
      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith({
          email: 'test@example.com',
          password: 'password123',
          rememberMe: false,
        });
      });
    });

    it('should include rememberMe when checked', async () => {
      const user = userEvent.setup();
      render(<LoginPage />);
      
      await user.type(getEmailInput(), 'test@example.com');
      await user.type(getPasswordInput(), 'password123');
      await user.click(screen.getByRole('checkbox'));
      await user.click(screen.getByRole('button', { name: /sign in/i }));
      
      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith({
          email: 'test@example.com',
          password: 'password123',
          rememberMe: true,
        });
      });
    });
  });

  describe('Loading State', () => {
    it('should show loading state during submission', async () => {
      const user = userEvent.setup();
      mockLogin.mockImplementation(() => new Promise(resolve => setTimeout(() => resolve({ success: true }), 100)));
      
      render(<LoginPage />);
      
      await user.type(getEmailInput(), 'test@example.com');
      await user.type(getPasswordInput(), 'password123');
      await user.click(screen.getByRole('button', { name: /sign in/i }));
      
      expect(screen.getByText(/signing in/i)).toBeInTheDocument();
    });

    it('should disable button during submission', async () => {
      const user = userEvent.setup();
      mockLogin.mockImplementation(() => new Promise(resolve => setTimeout(() => resolve({ success: true }), 100)));
      
      render(<LoginPage />);
      
      await user.type(getEmailInput(), 'test@example.com');
      await user.type(getPasswordInput(), 'password123');
      
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);
      
      expect(submitButton).toBeDisabled();
    });
  });

  describe('Error Handling', () => {
    it('should display error message on login failure', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue({ success: false, error: 'Invalid credentials' });
      
      render(<LoginPage />);
      
      await user.type(getEmailInput(), 'test@example.com');
      await user.type(getPasswordInput(), 'wrongpassword');
      await user.click(screen.getByRole('button', { name: /sign in/i }));
      
      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
      });
    });

    it('should display generic error on exception', async () => {
      const user = userEvent.setup();
      mockLogin.mockRejectedValue(new Error('Network error'));
      
      render(<LoginPage />);
      
      await user.type(getEmailInput(), 'test@example.com');
      await user.type(getPasswordInput(), 'password123');
      await user.click(screen.getByRole('button', { name: /sign in/i }));
      
      await waitFor(() => {
        expect(screen.getByText(/an error occurred/i)).toBeInTheDocument();
      });
    });
  });

  describe('MFA Flow', () => {
    it('should show MFA verification when required', async () => {
      const user = userEvent.setup();
      mockLogin.mockResolvedValue({ 
        success: false, 
        requiresMfa: true, 
        mfaToken: 'test-mfa-token' 
      });
      
      render(<LoginPage />);
      
      await user.type(getEmailInput(), 'test@example.com');
      await user.type(getPasswordInput(), 'password123');
      await user.click(screen.getByRole('button', { name: /sign in/i }));
      
      await waitFor(() => {
        expect(screen.getByText(/two-factor authentication/i)).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('should have proper autocomplete attributes', () => {
      render(<LoginPage />);
      
      expect(getEmailInput()).toHaveAttribute('autocomplete', 'email');
      expect(getPasswordInput()).toHaveAttribute('autocomplete', 'current-password');
    });

    it('should have accessible password toggle button', () => {
      render(<LoginPage />);
      
      const toggleButton = screen.getByRole('button', { name: /show password/i });
      expect(toggleButton).toHaveAttribute('aria-label');
    });
  });
});
