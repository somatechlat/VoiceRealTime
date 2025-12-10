/**
 * Next.js Middleware for Route Protection
 * Implements Requirements 2.7, 2.8: Route guards and portal separation
 */

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Public routes that don't require authentication
const PUBLIC_ROUTES = [
  '/login',
  '/register',
  '/forgot-password',
  '/reset-password',
  '/api/auth/login',
  '/api/auth/register',
  '/api/auth/refresh',
  '/api/auth/mfa/verify',
];

// Admin-only routes
const ADMIN_ROUTES = [
  '/admin',
];

// Customer routes
const CUSTOMER_ROUTES = [
  '/dashboard',
  '/api-keys',
  '/billing',
  '/team',
  '/settings',
];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Allow public routes
  if (PUBLIC_ROUTES.some(route => pathname.startsWith(route))) {
    return NextResponse.next();
  }

  // Allow static files and API routes (API routes handle their own auth)
  if (
    pathname.startsWith('/_next') ||
    pathname.startsWith('/static') ||
    pathname.includes('.')
  ) {
    return NextResponse.next();
  }

  // Check for auth token
  const token = request.cookies.get('agentvoicebox_access_token')?.value;
  
  if (!token) {
    // Redirect to login if no token
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('redirect', pathname);
    return NextResponse.redirect(loginUrl);
  }

  // Decode token to check roles (basic check - full validation on server)
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const roles: string[] = payload.roles || [];
    
    // Check admin route access
    if (ADMIN_ROUTES.some(route => pathname.startsWith(route))) {
      const adminRoles = ['super_admin', 'tenant_admin', 'support_agent', 'billing_admin'];
      const hasAdminRole = roles.some(r => adminRoles.includes(r));
      
      if (!hasAdminRole) {
        // Return 403 for unauthorized admin access
        return new NextResponse('Forbidden', { status: 403 });
      }
    }

    // Check customer route access
    if (CUSTOMER_ROUTES.some(route => pathname.startsWith(route))) {
      const customerRoles = ['owner', 'admin', 'developer', 'billing', 'viewer'];
      const adminRoles = ['super_admin', 'tenant_admin', 'support_agent', 'billing_admin'];
      const hasAccess = roles.some(r => 
        customerRoles.includes(r) || adminRoles.includes(r)
      );
      
      if (!hasAccess) {
        return new NextResponse('Forbidden', { status: 403 });
      }
    }

    return NextResponse.next();
  } catch {
    // Invalid token - redirect to login
    const loginUrl = new URL('/login', request.url);
    return NextResponse.redirect(loginUrl);
  }
}

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};
