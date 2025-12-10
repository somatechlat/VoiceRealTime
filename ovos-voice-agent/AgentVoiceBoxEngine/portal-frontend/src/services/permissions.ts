/**
 * Permission System
 * Implements Requirements 3.1-3.6, 4.1-4.6: Role-based access control
 */

// Customer portal roles
export type CustomerRole = 'owner' | 'admin' | 'developer' | 'billing' | 'viewer';

// Admin portal roles
export type AdminRole = 'super_admin' | 'tenant_admin' | 'support_agent' | 'billing_admin' | 'viewer';

// All roles
export type Role = CustomerRole | AdminRole;

// Permission types
export type Permission =
  // Team permissions
  | 'team:manage'
  | 'team:view'
  // API key permissions
  | 'api_keys:create'
  | 'api_keys:rotate'
  | 'api_keys:revoke'
  | 'api_keys:view'
  // Billing permissions
  | 'billing:manage'
  | 'billing:view'
  // Usage permissions
  | 'usage:view'
  // Settings permissions
  | 'settings:manage'
  // Tenant permissions (admin portal)
  | 'tenant:manage'
  | 'tenant:view'
  | 'tenant:delete'
  // Impersonation (admin portal)
  | 'impersonate:user'
  // System configuration (admin portal)
  | 'system:configure';

// Customer role permissions mapping (Requirements 3.1-3.6)
export const CUSTOMER_ROLE_PERMISSIONS: Record<CustomerRole, Permission[]> = {
  owner: [
    'team:manage',
    'team:view',
    'api_keys:create',
    'api_keys:rotate',
    'api_keys:revoke',
    'api_keys:view',
    'billing:manage',
    'billing:view',
    'usage:view',
    'settings:manage',
  ],
  admin: [
    'team:manage',
    'team:view',
    'api_keys:create',
    'api_keys:rotate',
    'api_keys:revoke',
    'api_keys:view',
    'billing:view',
    'usage:view',
  ],
  developer: [
    'api_keys:create',
    'api_keys:rotate',
    'api_keys:view',
    'usage:view',
  ],
  billing: [
    'billing:manage',
    'billing:view',
  ],
  viewer: [
    'usage:view',
  ],
};

// Admin role permissions mapping (Requirements 4.1-4.6)
export const ADMIN_ROLE_PERMISSIONS: Record<AdminRole, Permission[]> = {
  super_admin: [
    'tenant:manage',
    'tenant:view',
    'tenant:delete',
    'impersonate:user',
    'system:configure',
    'billing:manage',
    'billing:view',
  ],
  tenant_admin: [
    'tenant:manage',
    'tenant:view',
  ],
  support_agent: [
    'tenant:view',
    'impersonate:user',
  ],
  billing_admin: [
    'billing:manage',
    'billing:view',
  ],
  viewer: [
    'tenant:view',
  ],
};

// Combined role permissions
export const ROLE_PERMISSIONS: Record<string, Permission[]> = {
  ...CUSTOMER_ROLE_PERMISSIONS,
  ...ADMIN_ROLE_PERMISSIONS,
};

/**
 * Get permissions for a single role
 */
export function getRolePermissions(role: Role): Permission[] {
  return ROLE_PERMISSIONS[role] || [];
}

/**
 * Get union of permissions for multiple roles (Property 7)
 */
export function getPermissionUnion(roles: Role[]): Permission[] {
  const permissionSet = new Set<Permission>();
  
  for (const role of roles) {
    const rolePermissions = ROLE_PERMISSIONS[role];
    if (rolePermissions) {
      rolePermissions.forEach(p => permissionSet.add(p));
    }
  }
  
  return Array.from(permissionSet);
}

/**
 * Check if a role has a specific permission
 */
export function roleHasPermission(role: Role, permission: Permission): boolean {
  const permissions = ROLE_PERMISSIONS[role];
  return permissions?.includes(permission) ?? false;
}

/**
 * Check if any of the roles has a specific permission
 */
export function hasPermission(roles: Role[], permission: Permission): boolean {
  return roles.some(role => roleHasPermission(role, permission));
}

/**
 * Check if roles include any customer portal role
 */
export function isCustomerUser(roles: string[]): boolean {
  const customerRoles: CustomerRole[] = ['owner', 'admin', 'developer', 'billing', 'viewer'];
  return roles.some(r => customerRoles.includes(r as CustomerRole));
}

/**
 * Check if roles include any admin portal role
 */
export function isAdminUser(roles: string[]): boolean {
  const adminRoles: AdminRole[] = ['super_admin', 'tenant_admin', 'support_agent', 'billing_admin'];
  return roles.some(r => adminRoles.includes(r as AdminRole));
}

/**
 * Get portal type based on roles
 */
export function getPortalType(roles: string[]): 'customer' | 'admin' | 'none' {
  if (isAdminUser(roles)) return 'admin';
  if (isCustomerUser(roles)) return 'customer';
  return 'none';
}

/**
 * Validate route access based on roles (Property 1)
 */
export function canAccessRoute(roles: string[], route: string): boolean {
  // Auth routes are public - always accessible
  const publicRoutes = ['/login', '/register', '/forgot-password', '/reset-password'];
  if (publicRoutes.some(r => route.startsWith(r))) {
    return true;
  }
  
  const portalType = getPortalType(roles);
  
  // Admin routes - only admin users
  if (route.startsWith('/admin')) {
    return portalType === 'admin';
  }
  
  // Customer routes - customer or admin users
  const customerRoutes = ['/dashboard', '/api-keys', '/billing', '/team', '/settings', '/customer'];
  if (customerRoutes.some(r => route.startsWith(r))) {
    return portalType === 'customer' || portalType === 'admin';
  }
  
  return false;
}
