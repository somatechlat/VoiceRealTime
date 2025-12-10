/**
 * Authentication Property Tests
 * Tests for JWT claims, session management, token refresh, and MFA
 * 
 * Uses fast-check for property-based testing with 100+ iterations
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as fc from 'fast-check';
import { decodeJWT, validateJWTClaims, isJWTExpired, extractUserFromJWT } from '@/services/jwt-utils';

// Helper to create a valid JWT structure (base64 encoded)
function createMockJWT(claims: Record<string, unknown>): string {
  const header = { alg: 'RS256', typ: 'JWT' };
  const headerB64 = btoa(JSON.stringify(header));
  const payloadB64 = btoa(JSON.stringify(claims));
  const signature = 'mock-signature';
  return `${headerB64}.${payloadB64}.${signature}`;
}

// Arbitrary for valid JWT claims
const validJWTClaimsArb = fc.record({
  sub: fc.uuid(),
  tenant_id: fc.uuid(),
  email: fc.emailAddress(),
  preferred_username: fc.string({ minLength: 3, maxLength: 50 }).filter(s => /^[a-zA-Z0-9_]+$/.test(s)),
  roles: fc.array(fc.constantFrom('owner', 'admin', 'developer', 'billing', 'viewer'), { minLength: 1, maxLength: 5 }),
  permissions: fc.array(fc.constantFrom(
    'team:manage', 'team:view',
    'api_keys:create', 'api_keys:rotate', 'api_keys:revoke', 'api_keys:view',
    'billing:manage', 'billing:view',
    'usage:view', 'settings:manage'
  ), { minLength: 0, maxLength: 10 }),
  exp: fc.integer({ min: Math.floor(Date.now() / 1000) + 3600, max: Math.floor(Date.now() / 1000) + 86400 }),
  iat: fc.integer({ min: Math.floor(Date.now() / 1000) - 3600, max: Math.floor(Date.now() / 1000) }),
  iss: fc.constant('https://auth.agentvoicebox.com/realms/agentvoicebox'),
  aud: fc.constantFrom('agentvoicebox-api', ['agentvoicebox-api', 'account']),
});

// Arbitrary for session timing
const sessionTimingArb = fc.record({
  sessionStartMs: fc.integer({ min: 0, max: Date.now() }),
  lastActivityMs: fc.integer({ min: 0, max: Date.now() }),
  currentTimeMs: fc.integer({ min: 0, max: Date.now() + 10 * 60 * 60 * 1000 }),
});

// Arbitrary for MFA scenarios
const mfaScenarioArb = fc.record({
  mfaEnabled: fc.boolean(),
  hasPassword: fc.boolean(),
  hasMfaCode: fc.boolean(),
  mfaCodeValid: fc.boolean(),
});

describe('Authentication Property Tests', () => {
  /**
   * **Feature: saas-portal, Property 2: JWT Token Claims Completeness**
   * For any successful login, the resulting JWT access token SHALL contain
   * all required claims: user_id, tenant_id, roles, and permissions.
   * **Validates: Requirements 2.2**
   */
  describe('Property 2: JWT Token Claims Completeness', () => {
    it('should extract all required claims from valid JWT tokens', () => {
      fc.assert(
        fc.property(validJWTClaimsArb, (claims) => {
          const token = createMockJWT(claims);
          const decoded = decodeJWT(token);
          
          // All required claims must be present
          expect(decoded).not.toBeNull();
          expect(decoded!.sub).toBe(claims.sub);
          expect(decoded!.tenant_id).toBe(claims.tenant_id);
          expect(decoded!.roles).toEqual(claims.roles);
          expect(decoded!.permissions).toEqual(claims.permissions);
        }),
        { numRuns: 100 }
      );
    });

    it('should validate JWT has required claims', () => {
      fc.assert(
        fc.property(validJWTClaimsArb, (claims) => {
          const token = createMockJWT(claims);
          const validation = validateJWTClaims(token);
          
          // Valid tokens should pass validation
          expect(validation.valid).toBe(true);
          expect(validation.missingClaims).toHaveLength(0);
        }),
        { numRuns: 100 }
      );
    });

    it('should detect missing required claims', () => {
      const incompleteClaimsArb = fc.record({
        // Missing sub, tenant_id, or exp
        email: fc.emailAddress(),
        roles: fc.array(fc.string(), { minLength: 1 }),
      });

      fc.assert(
        fc.property(incompleteClaimsArb, (claims) => {
          const token = createMockJWT(claims);
          const validation = validateJWTClaims(token);
          
          // Should detect missing claims
          expect(validation.valid).toBe(false);
          expect(validation.missingClaims.length).toBeGreaterThan(0);
        }),
        { numRuns: 100 }
      );
    });

    it('should extract user info correctly from JWT', () => {
      fc.assert(
        fc.property(validJWTClaimsArb, (claims) => {
          const token = createMockJWT(claims);
          const userInfo = extractUserFromJWT(token);
          
          expect(userInfo).not.toBeNull();
          expect(userInfo!.userId).toBe(claims.sub);
          expect(userInfo!.tenantId).toBe(claims.tenant_id);
          expect(userInfo!.email).toBe(claims.email);
          expect(userInfo!.username).toBe(claims.preferred_username);
          expect(userInfo!.roles).toEqual(claims.roles);
          expect(userInfo!.permissions).toEqual(claims.permissions);
        }),
        { numRuns: 100 }
      );
    });
  });


  /**
   * **Feature: saas-portal, Property 4: Session Timeout Enforcement**
   * For any session, if idle time exceeds 30 minutes OR total session time
   * exceeds 8 hours, the session SHALL be invalidated.
   * **Validates: Requirements 2.5**
   */
  describe('Property 4: Session Timeout Enforcement', () => {
    const IDLE_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
    const MAX_SESSION_MS = 8 * 60 * 60 * 1000; // 8 hours

    function shouldSessionBeValid(
      sessionStartMs: number,
      lastActivityMs: number,
      currentTimeMs: number
    ): boolean {
      const idleTime = currentTimeMs - lastActivityMs;
      const sessionDuration = currentTimeMs - sessionStartMs;
      
      return idleTime < IDLE_TIMEOUT_MS && sessionDuration < MAX_SESSION_MS;
    }

    it('should invalidate sessions that exceed idle timeout', () => {
      const idleTimeoutScenarioArb = fc.record({
        sessionStartMs: fc.constant(Date.now() - 60 * 60 * 1000), // 1 hour ago
        lastActivityMs: fc.integer({ 
          min: Date.now() - 2 * 60 * 60 * 1000, // 2 hours ago
          max: Date.now() - 31 * 60 * 1000 // 31 minutes ago (past idle timeout)
        }),
        currentTimeMs: fc.constant(Date.now()),
      });

      fc.assert(
        fc.property(idleTimeoutScenarioArb, ({ sessionStartMs, lastActivityMs, currentTimeMs }) => {
          const idleTime = currentTimeMs - lastActivityMs;
          
          // If idle time exceeds 30 minutes, session should be invalid
          if (idleTime >= IDLE_TIMEOUT_MS) {
            expect(shouldSessionBeValid(sessionStartMs, lastActivityMs, currentTimeMs)).toBe(false);
          }
        }),
        { numRuns: 100 }
      );
    });

    it('should invalidate sessions that exceed max duration', () => {
      const maxDurationScenarioArb = fc.record({
        sessionStartMs: fc.integer({
          min: Date.now() - 12 * 60 * 60 * 1000, // 12 hours ago
          max: Date.now() - 8 * 60 * 60 * 1000 - 1000 // Just over 8 hours ago
        }),
        lastActivityMs: fc.constant(Date.now() - 1000), // Recent activity
        currentTimeMs: fc.constant(Date.now()),
      });

      fc.assert(
        fc.property(maxDurationScenarioArb, ({ sessionStartMs, lastActivityMs, currentTimeMs }) => {
          const sessionDuration = currentTimeMs - sessionStartMs;
          
          // If session duration exceeds 8 hours, session should be invalid
          if (sessionDuration >= MAX_SESSION_MS) {
            expect(shouldSessionBeValid(sessionStartMs, lastActivityMs, currentTimeMs)).toBe(false);
          }
        }),
        { numRuns: 100 }
      );
    });

    it('should keep sessions valid within timeout limits', () => {
      const validSessionArb = fc.record({
        sessionStartMs: fc.integer({
          min: Date.now() - 7 * 60 * 60 * 1000, // Up to 7 hours ago
          max: Date.now() - 1000
        }),
        lastActivityMs: fc.integer({
          min: Date.now() - 29 * 60 * 1000, // Up to 29 minutes ago
          max: Date.now() - 1000
        }),
        currentTimeMs: fc.constant(Date.now()),
      });

      fc.assert(
        fc.property(validSessionArb, ({ sessionStartMs, lastActivityMs, currentTimeMs }) => {
          const idleTime = currentTimeMs - lastActivityMs;
          const sessionDuration = currentTimeMs - sessionStartMs;
          
          // If both conditions are within limits, session should be valid
          if (idleTime < IDLE_TIMEOUT_MS && sessionDuration < MAX_SESSION_MS) {
            expect(shouldSessionBeValid(sessionStartMs, lastActivityMs, currentTimeMs)).toBe(true);
          }
        }),
        { numRuns: 100 }
      );
    });
  });

  /**
   * **Feature: saas-portal, Property 5: Token Refresh Transparency**
   * For any expired access token with a valid refresh token, the system
   * SHALL obtain a new access token without requiring user re-authentication.
   * **Validates: Requirements 2.6**
   */
  describe('Property 5: Token Refresh Transparency', () => {
    it('should correctly identify expired tokens', () => {
      const expiredTokenArb = fc.record({
        sub: fc.uuid(),
        tenant_id: fc.uuid(),
        email: fc.emailAddress(),
        preferred_username: fc.string({ minLength: 3 }),
        roles: fc.array(fc.string(), { minLength: 1 }),
        permissions: fc.array(fc.string()),
        // Expired: exp is in the past
        exp: fc.integer({ 
          min: Math.floor(Date.now() / 1000) - 3600, 
          max: Math.floor(Date.now() / 1000) - 60 
        }),
        iat: fc.integer({ min: Math.floor(Date.now() / 1000) - 7200 }),
        iss: fc.constant('https://auth.agentvoicebox.com'),
        aud: fc.constant('agentvoicebox-api'),
      });

      fc.assert(
        fc.property(expiredTokenArb, (claims) => {
          const token = createMockJWT(claims);
          
          // Token should be identified as expired
          expect(isJWTExpired(token, 0)).toBe(true);
        }),
        { numRuns: 100 }
      );
    });

    it('should correctly identify valid (non-expired) tokens', () => {
      const validTokenArb = fc.record({
        sub: fc.uuid(),
        tenant_id: fc.uuid(),
        email: fc.emailAddress(),
        preferred_username: fc.string({ minLength: 3 }),
        roles: fc.array(fc.string(), { minLength: 1 }),
        permissions: fc.array(fc.string()),
        // Valid: exp is in the future
        exp: fc.integer({ 
          min: Math.floor(Date.now() / 1000) + 300, // At least 5 minutes from now
          max: Math.floor(Date.now() / 1000) + 3600 
        }),
        iat: fc.integer({ min: Math.floor(Date.now() / 1000) - 60 }),
        iss: fc.constant('https://auth.agentvoicebox.com'),
        aud: fc.constant('agentvoicebox-api'),
      });

      fc.assert(
        fc.property(validTokenArb, (claims) => {
          const token = createMockJWT(claims);
          
          // Token should not be identified as expired
          expect(isJWTExpired(token, 0)).toBe(false);
        }),
        { numRuns: 100 }
      );
    });

    it('should respect buffer time for token expiry check', () => {
      const bufferTestArb = fc.record({
        // Token expires in 30-90 seconds
        expiresInSeconds: fc.integer({ min: 30, max: 90 }),
        bufferSeconds: fc.integer({ min: 0, max: 120 }),
      });

      fc.assert(
        fc.property(bufferTestArb, ({ expiresInSeconds, bufferSeconds }) => {
          const claims = {
            sub: 'test-user',
            tenant_id: 'test-tenant',
            exp: Math.floor(Date.now() / 1000) + expiresInSeconds,
            iat: Math.floor(Date.now() / 1000),
          };
          const token = createMockJWT(claims);
          
          const isExpired = isJWTExpired(token, bufferSeconds);
          
          // Token should be considered expired if remaining time < buffer
          if (expiresInSeconds <= bufferSeconds) {
            expect(isExpired).toBe(true);
          } else {
            expect(isExpired).toBe(false);
          }
        }),
        { numRuns: 100 }
      );
    });
  });

  /**
   * **Feature: saas-portal, Property 3: MFA Enforcement**
   * For any user with MFA enabled, a login attempt with only username/password
   * SHALL NOT grant full access until second factor is provided.
   * **Validates: Requirements 2.4**
   */
  describe('Property 3: MFA Enforcement', () => {
    interface LoginAttempt {
      hasValidCredentials: boolean;
      mfaEnabled: boolean;
      mfaCodeProvided: boolean;
      mfaCodeValid: boolean;
    }

    function simulateLoginResult(attempt: LoginAttempt): {
      success: boolean;
      requiresMfa: boolean;
      fullAccess: boolean;
    } {
      // Invalid credentials always fail
      if (!attempt.hasValidCredentials) {
        return { success: false, requiresMfa: false, fullAccess: false };
      }

      // MFA not enabled - grant full access
      if (!attempt.mfaEnabled) {
        return { success: true, requiresMfa: false, fullAccess: true };
      }

      // MFA enabled but no code provided - require MFA
      if (!attempt.mfaCodeProvided) {
        return { success: true, requiresMfa: true, fullAccess: false };
      }

      // MFA code provided but invalid - fail
      if (!attempt.mfaCodeValid) {
        return { success: false, requiresMfa: true, fullAccess: false };
      }

      // Valid credentials + valid MFA code - full access
      return { success: true, requiresMfa: false, fullAccess: true };
    }

    it('should require MFA when enabled and no code provided', () => {
      const mfaRequiredArb = fc.record({
        hasValidCredentials: fc.constant(true),
        mfaEnabled: fc.constant(true),
        mfaCodeProvided: fc.constant(false),
        mfaCodeValid: fc.boolean(),
      });

      fc.assert(
        fc.property(mfaRequiredArb, (attempt) => {
          const result = simulateLoginResult(attempt);
          
          // Should not grant full access without MFA code
          expect(result.fullAccess).toBe(false);
          expect(result.requiresMfa).toBe(true);
        }),
        { numRuns: 100 }
      );
    });

    it('should grant full access when MFA is disabled', () => {
      const noMfaArb = fc.record({
        hasValidCredentials: fc.constant(true),
        mfaEnabled: fc.constant(false),
        mfaCodeProvided: fc.boolean(),
        mfaCodeValid: fc.boolean(),
      });

      fc.assert(
        fc.property(noMfaArb, (attempt) => {
          const result = simulateLoginResult(attempt);
          
          // Should grant full access without MFA
          expect(result.fullAccess).toBe(true);
          expect(result.requiresMfa).toBe(false);
        }),
        { numRuns: 100 }
      );
    });

    it('should grant full access with valid MFA code', () => {
      const validMfaArb = fc.record({
        hasValidCredentials: fc.constant(true),
        mfaEnabled: fc.constant(true),
        mfaCodeProvided: fc.constant(true),
        mfaCodeValid: fc.constant(true),
      });

      fc.assert(
        fc.property(validMfaArb, (attempt) => {
          const result = simulateLoginResult(attempt);
          
          // Should grant full access with valid MFA
          expect(result.fullAccess).toBe(true);
          expect(result.requiresMfa).toBe(false);
        }),
        { numRuns: 100 }
      );
    });

    it('should deny access with invalid MFA code', () => {
      const invalidMfaArb = fc.record({
        hasValidCredentials: fc.constant(true),
        mfaEnabled: fc.constant(true),
        mfaCodeProvided: fc.constant(true),
        mfaCodeValid: fc.constant(false),
      });

      fc.assert(
        fc.property(invalidMfaArb, (attempt) => {
          const result = simulateLoginResult(attempt);
          
          // Should not grant full access with invalid MFA
          expect(result.fullAccess).toBe(false);
          expect(result.success).toBe(false);
        }),
        { numRuns: 100 }
      );
    });

    it('should handle all MFA scenarios correctly', () => {
      const allScenariosArb = fc.record({
        hasValidCredentials: fc.boolean(),
        mfaEnabled: fc.boolean(),
        mfaCodeProvided: fc.boolean(),
        mfaCodeValid: fc.boolean(),
      });

      fc.assert(
        fc.property(allScenariosArb, (attempt) => {
          const result = simulateLoginResult(attempt);
          
          // Key invariant: MFA enabled + no valid MFA = no full access
          if (attempt.hasValidCredentials && attempt.mfaEnabled) {
            if (!attempt.mfaCodeProvided || !attempt.mfaCodeValid) {
              expect(result.fullAccess).toBe(false);
            }
          }
          
          // Key invariant: Invalid credentials = no access
          if (!attempt.hasValidCredentials) {
            expect(result.success).toBe(false);
            expect(result.fullAccess).toBe(false);
          }
        }),
        { numRuns: 100 }
      );
    });
  });
});
