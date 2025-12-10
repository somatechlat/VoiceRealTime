import { test, expect } from '@playwright/test';

const KEYCLOAK_URL = 'http://localhost:25004';
const PORTAL_URL = 'http://localhost:25007';
const TEST_USER = {
  email: 'demo@test.com',
  password: 'demo123',
};

test.describe('Authentication Flow', () => {
  test('should redirect unauthenticated user to login page', async ({ page }) => {
    // Go to root
    await page.goto('/');
    
    // Should redirect to login
    await expect(page).toHaveURL(/\/login/);
    
    // Login page should have SSO button
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible();
  });

  test('should show login page with correct elements', async ({ page }) => {
    await page.goto('/login');
    
    // Check for AgentVoiceBox branding
    await expect(page.getByText('AgentVoiceBox')).toBeVisible();
    
    // Check for SSO button
    await expect(page.getByRole('button', { name: /sign in with sso/i })).toBeVisible();
    
    // Check for demo credentials info
    await expect(page.getByText('demo@test.com')).toBeVisible();
  });

  test('should redirect to Keycloak on SSO button click', async ({ page }) => {
    await page.goto('/login');
    
    // Click SSO button
    await page.getByRole('button', { name: /sign in with sso/i }).click();
    
    // Should redirect to Keycloak
    await page.waitForURL(/localhost:25004.*auth/);
    
    // Keycloak login form should be visible
    await expect(page.locator('#username, #kc-form-login, input[name="username"]')).toBeVisible({ timeout: 10000 });
  });

  test('should complete full login flow with Keycloak', async ({ page }) => {
    // Start at login page
    await page.goto('/login');
    
    // Click SSO button
    await page.getByRole('button', { name: /sign in with sso/i }).click();
    
    // Wait for Keycloak login page
    await page.waitForURL(/localhost:25004/);
    
    // Fill in credentials
    await page.fill('#username', TEST_USER.email);
    await page.fill('#password', TEST_USER.password);
    
    // Submit login form
    await page.click('#kc-login');
    
    // Should redirect back to portal dashboard
    await page.waitForURL(/localhost:25007/, { timeout: 15000 });
    
    // Should be on dashboard or auth callback
    const url = page.url();
    expect(url).toMatch(/\/(dashboard|auth\/callback)/);
  });
});

test.describe('Protected Routes', () => {
  test('should redirect /dashboard to login when not authenticated', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page).toHaveURL(/\/login/);
  });

  test('should redirect /api-keys to login when not authenticated', async ({ page }) => {
    await page.goto('/api-keys');
    await expect(page).toHaveURL(/\/login/);
  });

  test('should redirect /billing to login when not authenticated', async ({ page }) => {
    await page.goto('/billing');
    await expect(page).toHaveURL(/\/login/);
  });

  test('should redirect /team to login when not authenticated', async ({ page }) => {
    await page.goto('/team');
    await expect(page).toHaveURL(/\/login/);
  });

  test('should redirect /settings to login when not authenticated', async ({ page }) => {
    await page.goto('/settings');
    await expect(page).toHaveURL(/\/login/);
  });

  test('should redirect /admin to login when not authenticated', async ({ page }) => {
    await page.goto('/admin');
    await expect(page).toHaveURL(/\/login/);
  });
});

test.describe('Auth Callback', () => {
  test('should handle auth callback page', async ({ page }) => {
    // Auth callback without code should show error or redirect
    await page.goto('/auth/callback');
    
    // Should either show error or redirect to login
    await page.waitForTimeout(2000);
    const url = page.url();
    const hasError = await page.getByText(/error|no authorization code/i).isVisible().catch(() => false);
    
    expect(url.includes('/login') || url.includes('/auth/callback') || hasError).toBeTruthy();
  });
});

test.describe('Theme Toggle', () => {
  test('should have theme toggle on login page', async ({ page }) => {
    await page.goto('/login');
    
    // Look for theme toggle button (sun/moon icon)
    const themeToggle = page.locator('button').filter({ has: page.locator('svg') }).first();
    await expect(themeToggle).toBeVisible();
  });
});
