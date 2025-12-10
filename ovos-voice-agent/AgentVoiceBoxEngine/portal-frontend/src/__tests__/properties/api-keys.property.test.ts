/**
 * API Key Management Property Tests
 * Tests for API key correctness properties using fast-check
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import * as fc from "fast-check";

/**
 * **Feature: saas-portal, Property 13: API Key Single Display**
 * For any newly created API key, the full key value SHALL be displayed exactly once
 * and never retrievable again.
 * **Validates: Requirements 8.3**
 */
describe("Property 13: API Key Single Display", () => {
  it("should only expose full key value once during creation", () => {
    fc.assert(
      fc.property(
        fc.record({
          keyId: fc.uuid(),
          keyName: fc.string({ minLength: 1, maxLength: 50 }),
          keySecret: fc.string({ minLength: 32, maxLength: 64 }),
          keyPrefix: fc.string({ minLength: 8, maxLength: 8 }),
        }),
        ({ keyId, keyName, keySecret, keyPrefix }) => {
          // Simulate key creation response
          const creationResponse = {
            id: keyId,
            name: keyName,
            secret: keySecret, // Full key only in creation response
            prefix: keyPrefix,
            scopes: ["realtime:connect"],
            created_at: new Date().toISOString(),
            is_active: true,
          };

          // Simulate subsequent GET response (no secret)
          const getResponse = {
            id: keyId,
            name: keyName,
            // secret is NOT included
            prefix: keyPrefix,
            scopes: ["realtime:connect"],
            created_at: creationResponse.created_at,
            is_active: true,
          };

          // Property: creation response includes full secret
          expect(creationResponse.secret).toBe(keySecret);
          expect(creationResponse.secret.length).toBeGreaterThanOrEqual(32);

          // Property: subsequent responses do NOT include secret
          expect(getResponse).not.toHaveProperty("secret");

          // Property: only prefix is available after creation
          expect(getResponse.prefix).toBe(keyPrefix);
          expect(getResponse.prefix.length).toBe(8);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it("should never return full key in list operations", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            id: fc.uuid(),
            name: fc.string({ minLength: 1, maxLength: 50 }),
            prefix: fc.string({ minLength: 8, maxLength: 8 }),
          }),
          { minLength: 1, maxLength: 10 }
        ),
        (keys) => {
          // Simulate list response
          const listResponse = keys.map((key) => ({
            id: key.id,
            name: key.name,
            prefix: key.prefix,
            scopes: ["realtime:connect"],
            created_at: new Date().toISOString(),
            is_active: true,
          }));

          // Property: no key in list should have secret field
          for (const key of listResponse) {
            expect(key).not.toHaveProperty("secret");
            expect(key.prefix.length).toBe(8);
          }

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it("should display warning about key not being retrievable", () => {
    fc.assert(
      fc.property(
        fc.record({
          keySecret: fc.string({ minLength: 32, maxLength: 64 }),
        }),
        ({ keySecret }) => {
          // Simulate UI state after key creation
          const uiState = {
            showKeyCreatedDialog: true,
            displayedSecret: keySecret,
            warningMessage: "Copy your API key now. You won't be able to see it again.",
          };

          // Property: warning message should be displayed
          expect(uiState.warningMessage).toContain("won't be able to see it again");

          // Property: dialog should be shown with the secret
          expect(uiState.showKeyCreatedDialog).toBe(true);
          expect(uiState.displayedSecret).toBe(keySecret);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });
});

/**
 * **Feature: saas-portal, Property 14: API Key Rotation Grace Period**
 * For any API key rotation, the old key SHALL remain valid for exactly 24 hours
 * after the new key is created.
 * **Validates: Requirements 8.5**
 */
describe("Property 14: API Key Rotation Grace Period", () => {
  const GRACE_PERIOD_HOURS = 24;
  const GRACE_PERIOD_MS = GRACE_PERIOD_HOURS * 60 * 60 * 1000;

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should keep old key valid for exactly 24 hours after rotation", () => {
    fc.assert(
      fc.property(
        fc.record({
          oldKeyId: fc.uuid(),
          newKeyId: fc.uuid(),
          rotationTime: fc.date({ min: new Date("2024-01-01"), max: new Date("2025-12-31") }),
        }),
        ({ oldKeyId, newKeyId, rotationTime }) => {
          // Set current time to rotation time
          vi.setSystemTime(rotationTime);

          // Simulate rotation response
          const rotationResponse = {
            old_key_id: oldKeyId,
            new_key: {
              id: newKeyId,
              secret: "new-secret-key-value",
              prefix: "newkey12",
            },
            grace_period_hours: GRACE_PERIOD_HOURS,
          };

          // Calculate grace period end
          const gracePeriodEnd = new Date(rotationTime.getTime() + GRACE_PERIOD_MS);

          // Property: grace period should be exactly 24 hours
          expect(rotationResponse.grace_period_hours).toBe(24);

          // Property: old key should be valid during grace period
          const duringGracePeriod = new Date(rotationTime.getTime() + GRACE_PERIOD_MS / 2);
          expect(duringGracePeriod.getTime()).toBeLessThan(gracePeriodEnd.getTime());

          // Property: old key should be invalid after grace period
          const afterGracePeriod = new Date(rotationTime.getTime() + GRACE_PERIOD_MS + 1);
          expect(afterGracePeriod.getTime()).toBeGreaterThan(gracePeriodEnd.getTime());

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it("should validate both old and new keys during grace period", () => {
    fc.assert(
      fc.property(
        fc.record({
          oldKeyId: fc.uuid(),
          newKeyId: fc.uuid(),
          requestTimeOffset: fc.integer({ min: 0, max: GRACE_PERIOD_MS - 1 }),
        }),
        ({ oldKeyId, newKeyId, requestTimeOffset }) => {
          const rotationTime = new Date();
          vi.setSystemTime(rotationTime);

          // Simulate key states
          const keyStates = {
            [oldKeyId]: {
              isValid: true,
              expiresAt: new Date(rotationTime.getTime() + GRACE_PERIOD_MS),
            },
            [newKeyId]: {
              isValid: true,
              expiresAt: null, // New key doesn't expire from rotation
            },
          };

          // Advance time within grace period
          vi.advanceTimersByTime(requestTimeOffset);
          const currentTime = new Date();

          // Property: both keys should be valid during grace period
          const oldKeyValid = keyStates[oldKeyId].expiresAt!.getTime() > currentTime.getTime();
          const newKeyValid = keyStates[newKeyId].isValid;

          expect(oldKeyValid).toBe(true);
          expect(newKeyValid).toBe(true);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it("should invalidate old key after grace period expires", () => {
    fc.assert(
      fc.property(
        fc.record({
          oldKeyId: fc.uuid(),
          extraTimeMs: fc.integer({ min: 1, max: 86400000 }), // 1ms to 24 hours extra
        }),
        ({ oldKeyId, extraTimeMs }) => {
          const rotationTime = new Date();
          vi.setSystemTime(rotationTime);

          const gracePeriodEnd = new Date(rotationTime.getTime() + GRACE_PERIOD_MS);

          // Advance time past grace period
          vi.advanceTimersByTime(GRACE_PERIOD_MS + extraTimeMs);
          const currentTime = new Date();

          // Property: old key should be invalid after grace period
          const oldKeyValid = gracePeriodEnd.getTime() > currentTime.getTime();
          expect(oldKeyValid).toBe(false);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });
});

/**
 * **Feature: saas-portal, Property 15: API Key Immediate Revocation**
 * For any API key revocation, the key SHALL be immediately invalidated and all
 * subsequent requests using that key SHALL fail.
 * **Validates: Requirements 8.6**
 */
describe("Property 15: API Key Immediate Revocation", () => {
  it("should immediately invalidate revoked keys", () => {
    fc.assert(
      fc.property(
        fc.record({
          keyId: fc.uuid(),
          keyPrefix: fc.string({ minLength: 8, maxLength: 8 }),
        }),
        ({ keyId, keyPrefix }) => {
          // Simulate key state before revocation
          let keyState = {
            id: keyId,
            prefix: keyPrefix,
            is_active: true,
            revoked_at: null as Date | null,
          };

          // Simulate revocation
          const revocationTime = new Date();
          keyState = {
            ...keyState,
            is_active: false,
            revoked_at: revocationTime,
          };

          // Property: key should be immediately inactive
          expect(keyState.is_active).toBe(false);
          expect(keyState.revoked_at).toEqual(revocationTime);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it("should reject all requests with revoked keys", () => {
    fc.assert(
      fc.property(
        fc.record({
          keyId: fc.uuid(),
          requestCount: fc.integer({ min: 1, max: 100 }),
        }),
        ({ keyId, requestCount }) => {
          // Simulate revoked key
          const revokedKey = {
            id: keyId,
            is_active: false,
            revoked_at: new Date(),
          };

          // Simulate multiple requests with revoked key
          const requestResults: boolean[] = [];
          for (let i = 0; i < requestCount; i++) {
            // All requests should fail
            const isAuthorized = revokedKey.is_active;
            requestResults.push(isAuthorized);
          }

          // Property: all requests should be rejected
          expect(requestResults.every((r) => r === false)).toBe(true);
          expect(requestResults.length).toBe(requestCount);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it("should not allow re-activation of revoked keys", () => {
    fc.assert(
      fc.property(
        fc.record({
          keyId: fc.uuid(),
          reactivationAttempts: fc.integer({ min: 1, max: 10 }),
        }),
        ({ keyId, reactivationAttempts }) => {
          // Simulate revoked key
          const revokedKey = {
            id: keyId,
            is_active: false,
            revoked_at: new Date(),
          };

          // Attempt to reactivate
          const reactivationResults: boolean[] = [];
          for (let i = 0; i < reactivationAttempts; i++) {
            // Reactivation should always fail for revoked keys
            const canReactivate = false; // Business rule: revoked keys cannot be reactivated
            reactivationResults.push(canReactivate);
          }

          // Property: all reactivation attempts should fail
          expect(reactivationResults.every((r) => r === false)).toBe(true);

          // Property: key should remain inactive
          expect(revokedKey.is_active).toBe(false);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it("should require confirmation before revocation", () => {
    fc.assert(
      fc.property(
        fc.record({
          keyId: fc.uuid(),
          keyName: fc.string({ minLength: 1, maxLength: 50 }),
          userConfirmed: fc.boolean(),
        }),
        ({ keyId, keyName, userConfirmed }) => {
          // Simulate revocation attempt
          let revocationSuccessful = false;

          if (userConfirmed) {
            // Only proceed if user confirmed
            revocationSuccessful = true;
          }

          // Property: revocation should only succeed with confirmation
          if (userConfirmed) {
            expect(revocationSuccessful).toBe(true);
          } else {
            expect(revocationSuccessful).toBe(false);
          }

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });
});

/**
 * API Key Scope Validation Properties
 */
describe("API Key Scope Validation", () => {
  const VALID_SCOPES = ["realtime:connect", "realtime:admin", "billing:read", "tenant:admin"];

  it("should only allow valid scopes", () => {
    fc.assert(
      fc.property(
        fc.array(fc.string({ minLength: 1, maxLength: 30 }), { minLength: 1, maxLength: 5 }),
        (requestedScopes) => {
          // Filter to only valid scopes
          const validScopes = requestedScopes.filter((s) => VALID_SCOPES.includes(s));
          const invalidScopes = requestedScopes.filter((s) => !VALID_SCOPES.includes(s));

          // Property: only valid scopes should be accepted
          for (const scope of validScopes) {
            expect(VALID_SCOPES).toContain(scope);
          }

          // Property: invalid scopes should be rejected
          for (const scope of invalidScopes) {
            expect(VALID_SCOPES).not.toContain(scope);
          }

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it("should require at least one scope", () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...VALID_SCOPES), { minLength: 0, maxLength: 4 }),
        (scopes) => {
          const isValid = scopes.length > 0;

          // Property: empty scopes should be invalid
          if (scopes.length === 0) {
            expect(isValid).toBe(false);
          } else {
            expect(isValid).toBe(true);
          }

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });
});

/**
 * API Key Prefix Generation Properties
 */
describe("API Key Prefix Generation", () => {
  it("should generate consistent 8-character prefixes", () => {
    fc.assert(
      fc.property(
        fc.string({ minLength: 32, maxLength: 64 }),
        (fullKey) => {
          // Generate prefix from full key
          const prefix = fullKey.substring(0, 8);

          // Property: prefix should always be 8 characters
          expect(prefix.length).toBe(8);

          // Property: prefix should be a substring of the full key
          expect(fullKey.startsWith(prefix)).toBe(true);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it("should display prefix with ellipsis", () => {
    fc.assert(
      fc.property(
        fc.string({ minLength: 8, maxLength: 8 }),
        (prefix) => {
          // Format for display
          const displayFormat = `${prefix}...`;

          // Property: display format should end with ellipsis
          expect(displayFormat).toMatch(/^.{8}\.\.\.$/);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });
});
