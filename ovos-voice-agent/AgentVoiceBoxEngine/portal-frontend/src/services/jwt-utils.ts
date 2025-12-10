/**
 * JWT Token Utilities
 * Implements Requirements 2.2: JWT token handling with claims extraction
 */

export interface JWTClaims {
  sub: string;           // User ID
  tenant_id: string;     // Tenant ID
  email: string;
  preferred_username: string;
  roles: string[];
  permissions: string[];
  exp: number;           // Expiration timestamp
  iat: number;           // Issued at timestamp
  iss: string;           // Issuer
  aud: string | string[]; // Audience
}

/**
 * Decode a JWT token without verification
 * Note: This is for client-side use only. Server validates tokens.
 */
export function decodeJWT(token: string): JWTClaims | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) {
      return null;
    }

    const payload = parts[1];
    const decoded = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(decoded) as JWTClaims;
  } catch (error) {
    console.error('Failed to decode JWT:', error);
    return null;
  }
}

/**
 * Check if a JWT token is expired
 */
export function isJWTExpired(token: string, bufferSeconds: number = 60): boolean {
  const claims = decodeJWT(token);
  if (!claims) return true;

  const now = Math.floor(Date.now() / 1000);
  return claims.exp <= (now + bufferSeconds);
}

/**
 * Get expiration time from JWT
 */
export function getJWTExpiration(token: string): Date | null {
  const claims = decodeJWT(token);
  if (!claims) return null;

  return new Date(claims.exp * 1000);
}

/**
 * Get time until JWT expires in milliseconds
 */
export function getJWTTimeToExpiry(token: string): number {
  const claims = decodeJWT(token);
  if (!claims) return 0;

  const expiresAt = claims.exp * 1000;
  return Math.max(0, expiresAt - Date.now());
}

/**
 * Extract user info from JWT claims
 */
export function extractUserFromJWT(token: string): {
  userId: string;
  tenantId: string;
  email: string;
  username: string;
  roles: string[];
  permissions: string[];
} | null {
  const claims = decodeJWT(token);
  if (!claims) return null;

  return {
    userId: claims.sub,
    tenantId: claims.tenant_id,
    email: claims.email,
    username: claims.preferred_username,
    roles: claims.roles || [],
    permissions: claims.permissions || [],
  };
}

/**
 * Validate JWT has required claims
 */
export function validateJWTClaims(token: string): {
  valid: boolean;
  missingClaims: string[];
} {
  const claims = decodeJWT(token);
  if (!claims) {
    return { valid: false, missingClaims: ['all'] };
  }

  const requiredClaims = ['sub', 'tenant_id', 'exp'];
  const missingClaims = requiredClaims.filter(claim => !(claim in claims));

  return {
    valid: missingClaims.length === 0,
    missingClaims,
  };
}
