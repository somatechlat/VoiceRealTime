/**
 * Authorization Property Tests
 * Tests for RBAC, route separation, and permission union
 * 
 * Uses fast-check for property-based testing with 100+ iterations
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
  Role,
  CustomerRole,
  AdminRole,
  Permission,
  getRolePermissions,
  getPermissionUnion,
  hasPermission,
  canAccessRoute,
  isCustomerUser,
  isAdminUser,
  CUSTOMER_ROLE_PERMISSIONS,
  ADMIN_ROLE_PERMISSIONS,
} from '@/services/permissions';

// Arbitraries for roles
const customerRoleArb = fc.constantFrom<CustomerRole>('owner', 'admin', 'developer', 'billing', 'viewer');
const adminRoleArb = fc.constantFrom<AdminRole>('super_admin', 'tenant_admin', 'support_agent', 'billing_admin', 'viewer');
const roleArb = fc.oneof(customerRoleArb, adminRoleArb) as fc.Arbitrary<Role>;

// Arbitraries for permissions
const permissionArb = fc.constantFrom<Permission>(
  'team:manage', 'team:view',
  'api_keys:create', 'api_keys:rotate', 'api_keys:revoke', 'api_keys:view',
  'billing:manage', 'billing:view',
  'usage:view', 'settings:manage',
  'tenant:manage', 'tenant:view', 'tenant:delete',
  'impersonate:user', 'system:configure'
);

// Arbitraries for routes
const adminRouteArb = fc.constantFrom(
  '/admin',
  '/admin/dashboard',
  '/admin/tenants',
  '/admin/billing',
  '/admin/plans',
  '/admin/monitoring',
  '/admin/audit'
);

const customerRouteArb = fc.constantFrom(
  '/dashboard',
  '/api-keys',
  '/billing',
  '/team',
  '/settings'
);

const publicRouteArb = fc.constantFrom(
  '/login',
  '/register',
  '/forgot-password'
);

describe('Authorization Property Tests', () => {
  /**
   * **Feature: saas-portal, Property 1: Portal Route Separation**
   * For any user with only customer portal roles, attempting to access
   * any admin portal route SHALL result in a 403 Forbidden response.
   * **Validates: Requirements 1.4, 2.8**
   */
  describe('Property 1: Portal Route Separation', () => {
    it('should deny customer-only users access to admin routes', () => {
      const customerOnlyRolesArb = fc.array(customerRoleArb, { minLength: 1, maxLength: 3 });

      fc.assert(
        fc.property(customerOnlyRolesArb, adminRouteArb, (roles, route) => {
          // Customer-only users should not access admin routes
          const canAccess = canAccessRoute(roles, route);
          expect(canAccess).toBe(false);
        }),
        { numRuns: 100 }
      );
    });

    it('should allow admin users access to admin routes', () => {
      const adminRolesArb = fc.array(adminRoleArb, { minLength: 1, maxLength: 3 })
        .filter(roles => roles.some(r => r !== 'viewer')); // Exclude viewer-only

      fc.assert(
        fc.property(adminRolesArb, adminRouteArb, (roles, route) => {
          // Admin users should access admin routes
          const canAccess = canAccessRoute(roles, route);
          expect(canAccess).toBe(true);
        }),
        { numRuns: 100 }
      );
    });

    it('should allow customer users access to customer routes', () => {
      const customerRolesArb = fc.array(customerRoleArb, { minLength: 1, maxLength: 3 });

      fc.assert(
        fc.property(customerRolesArb, customerRouteArb, (roles, route) => {
          // Customer users should access customer routes
          const canAccess = canAccessRoute(roles, route);
          expect(canAccess).toBe(true);
        }),
        { numRuns: 100 }
      );
    });

    it('should allow all users access to public routes', () => {
      const anyRolesArb = fc.array(roleArb, { minLength: 0, maxLength: 5 });

      fc.assert(
        fc.property(anyRolesArb, publicRouteArb, (roles, route) => {
          // All users should access public routes
          const canAccess = canAccessRoute(roles, route);
          expect(canAccess).toBe(true);
        }),
        { numRuns: 100 }
      );
    });
  });

  /**
   * **Feature: saas-portal, Property 6: RBAC Permission Check**
   * For any API request, the system SHALL verify the requesting user
   * has the required permission before processing the request.
   * **Validates: Requirements 2.7**
   */
  describe('Property 6: RBAC Permission Check', () => {
    it('should correctly identify permissions for each role', () => {
      fc.assert(
        fc.property(roleArb, (role) => {
          const permissions = getRolePermissions(role);
          
          // Each role should have defined permissions
          expect(Array.isArray(permissions)).toBe(true);
          
          // Permissions should be valid
          permissions.forEach(p => {
            expect(typeof p).toBe('string');
            expect(p.includes(':')).toBe(true);
          });
        }),
        { numRuns: 100 }
      );
    });

    it('should correctly check if role has permission', () => {
      fc.assert(
        fc.property(roleArb, permissionArb, (role, permission) => {
          const rolePermissions = getRolePermissions(role);
          const has = hasPermission([role], permission);
          
          // hasPermission should match direct lookup
          expect(has).toBe(rolePermissions.includes(permission));
        }),
        { numRuns: 100 }
      );
    });

    it('should enforce owner has all customer permissions', () => {
      const allCustomerPermissions: Permission[] = [
        'team:manage', 'team:view',
        'api_keys:create', 'api_keys:rotate', 'api_keys:revoke', 'api_keys:view',
        'billing:manage', 'billing:view',
        'usage:view', 'settings:manage',
      ];

      fc.assert(
        fc.property(fc.constantFrom(...allCustomerPermissions), (permission) => {
          const ownerPermissions = getRolePermissions('owner');
          expect(ownerPermissions).toContain(permission);
        }),
        { numRuns: 100 }
      );
    });

    it('should enforce super_admin has all admin permissions', () => {
      const allAdminPermissions: Permission[] = [
        'tenant:manage', 'tenant:view', 'tenant:delete',
        'impersonate:user', 'system:configure',
        'billing:manage', 'billing:view',
      ];

      fc.assert(
        fc.property(fc.constantFrom(...allAdminPermissions), (permission) => {
          const superAdminPermissions = getRolePermissions('super_admin');
          expect(superAdminPermissions).toContain(permission);
        }),
        { numRuns: 100 }
      );
    });
  });

  /**
   * **Feature: saas-portal, Property 7: Role Permission Union**
   * For any user with multiple roles, the effective permissions SHALL be
   * the union of all permissions from all assigned roles.
   * **Validates: Requirements 3.7**
   */
  describe('Property 7: Role Permission Union', () => {
    it('should compute union of permissions for multiple roles', () => {
      const multipleRolesArb = fc.array(roleArb, { minLength: 2, maxLength: 5 });

      fc.assert(
        fc.property(multipleRolesArb, (roles) => {
          const unionPermissions = getPermissionUnion(roles);
          
          // Union should contain all permissions from each role
          for (const role of roles) {
            const rolePermissions = getRolePermissions(role);
            for (const permission of rolePermissions) {
              expect(unionPermissions).toContain(permission);
            }
          }
        }),
        { numRuns: 100 }
      );
    });

    it('should not have duplicates in permission union', () => {
      const multipleRolesArb = fc.array(roleArb, { minLength: 2, maxLength: 5 });

      fc.assert(
        fc.property(multipleRolesArb, (roles) => {
          const unionPermissions = getPermissionUnion(roles);
          const uniquePermissions = new Set(unionPermissions);
          
          // No duplicates
          expect(unionPermissions.length).toBe(uniquePermissions.size);
        }),
        { numRuns: 100 }
      );
    });

    it('should be commutative - order of roles should not matter', () => {
      const twoRolesArb = fc.tuple(roleArb, roleArb);

      fc.assert(
        fc.property(twoRolesArb, ([role1, role2]) => {
          const union1 = getPermissionUnion([role1, role2]).sort();
          const union2 = getPermissionUnion([role2, role1]).sort();
          
          expect(union1).toEqual(union2);
        }),
        { numRuns: 100 }
      );
    });

    it('should be idempotent - adding same role twice should not change result', () => {
      fc.assert(
        fc.property(roleArb, (role) => {
          const single = getPermissionUnion([role]).sort();
          const double = getPermissionUnion([role, role]).sort();
          
          expect(single).toEqual(double);
        }),
        { numRuns: 100 }
      );
    });

    it('should correctly identify user type from roles', () => {
      fc.assert(
        fc.property(fc.array(customerRoleArb, { minLength: 1, maxLength: 3 }), (roles) => {
          expect(isCustomerUser(roles)).toBe(true);
        }),
        { numRuns: 100 }
      );

      fc.assert(
        fc.property(
          fc.array(adminRoleArb, { minLength: 1, maxLength: 3 })
            .filter(roles => roles.some(r => r !== 'viewer')),
          (roles) => {
            expect(isAdminUser(roles)).toBe(true);
          }
        ),
        { numRuns: 100 }
      );
    });
  });

  /**
   * Additional authorization invariants
   */
  describe('Authorization Invariants', () => {
    it('should ensure viewer role has minimal permissions', () => {
      const viewerPermissions = getRolePermissions('viewer');
      
      // Viewer should only have view permissions
      viewerPermissions.forEach(p => {
        expect(p.endsWith(':view') || p === 'usage:view').toBe(true);
      });
    });

    it('should ensure no customer role has admin-only permissions', () => {
      const adminOnlyPermissions: Permission[] = [
        'tenant:manage', 'tenant:view', 'tenant:delete',
        'impersonate:user', 'system:configure',
      ];

      fc.assert(
        fc.property(customerRoleArb, (role) => {
          const permissions = CUSTOMER_ROLE_PERMISSIONS[role];
          
          for (const adminPerm of adminOnlyPermissions) {
            expect(permissions).not.toContain(adminPerm);
          }
        }),
        { numRuns: 100 }
      );
    });

    it('should ensure permission hierarchy is respected', () => {
      // Owner should have all admin permissions
      const ownerPerms = new Set(CUSTOMER_ROLE_PERMISSIONS.owner);
      const adminPerms = new Set(CUSTOMER_ROLE_PERMISSIONS.admin);
      
      // Admin permissions should be subset of owner
      for (const perm of adminPerms) {
        expect(ownerPerms.has(perm)).toBe(true);
      }
    });
  });
});
