# AgentVoiceBox SaaS Portal - Technical Design Document

## Overview

This document describes the technical architecture and design for the AgentVoiceBox SaaS Portal System - a dual-portal architecture providing Customer Portal (tenant self-service) and Admin Portal (platform operator administration).

### Goals
- Deliver enterprise-grade security with Keycloak integration
- Provide intuitive, accessible UI with dark/light theme support
- Enable real-time data synchronization and responsive design
- Maintain clear separation between customer and admin functionality

### Technology Stack
- **Frontend**: Next.js 14 (App Router), React 18, TypeScript
- **Styling**: Tailwind CSS, CSS Custom Properties
- **State Management**: React Context, SWR for data fetching
- **Authentication**: Keycloak (OIDC), NextAuth.js
- **Icons**: Lucide React
- **Testing**: Vitest, React Testing Library, fast-check (property-based testing)

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Next.js Application                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   Customer Portal   │    │    Admin Portal     │            │
│  │   /customer/*       │    │    /admin/*         │            │
│  └─────────────────────┘    └─────────────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                    Shared Components Layer                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  Theme   │ │   UI     │ │  Forms   │ │  Charts  │          │
│  │ Context  │ │Components│ │          │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                      Services Layer                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │   Auth   │ │   API    │ │ Billing  │ │  Audit   │          │
│  │ Service  │ │  Client  │ │ Service  │ │ Service  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                    External Services                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Keycloak │ │   Lago   │ │  Stripe  │ │ Backend  │          │
│  │  (Auth)  │ │(Billing) │ │(Payments)│ │   API    │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Route Structure

```
src/app/
├── (auth)/
│   ├── login/page.tsx
│   ├── register/page.tsx
│   └── forgot-password/page.tsx
├── (customer)/
│   ├── layout.tsx              # Customer portal layout
│   ├── dashboard/page.tsx
│   ├── api-keys/page.tsx
│   ├── billing/page.tsx
│   ├── team/page.tsx
│   └── settings/page.tsx
├── (admin)/
│   ├── layout.tsx              # Admin portal layout
│   ├── dashboard/page.tsx
│   ├── tenants/page.tsx
│   ├── billing/page.tsx
│   ├── plans/page.tsx
│   ├── monitoring/page.tsx
│   └── audit/page.tsx
├── layout.tsx                  # Root layout with providers
├── providers.tsx               # Context providers
└── globals.css                 # Global styles & CSS variables
```

---

## Components and Interfaces

### Theme System

```typescript
// src/contexts/ThemeContext.tsx
type ThemeMode = 'light' | 'dark' | 'system';

interface ThemeContextValue {
  theme: ThemeMode;
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: ThemeMode) => void;
}
```

### Authentication

```typescript
// src/services/auth.ts
interface AuthService {
  login(credentials: LoginCredentials): Promise<AuthResult>;
  logout(): Promise<void>;
  refreshToken(): Promise<TokenPair>;
  getCurrentUser(): Promise<User | null>;
  hasPermission(permission: string): boolean;
}

interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

interface AuthResult {
  success: boolean;
  user?: User;
  tokens?: TokenPair;
  error?: AuthError;
  requiresMfa?: boolean;
}

interface TokenPair {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}
```

### API Client

```typescript
// src/services/api-client.ts
interface ApiClient {
  get<T>(url: string, options?: RequestOptions): Promise<ApiResponse<T>>;
  post<T>(url: string, data: unknown, options?: RequestOptions): Promise<ApiResponse<T>>;
  put<T>(url: string, data: unknown, options?: RequestOptions): Promise<ApiResponse<T>>;
  delete<T>(url: string, options?: RequestOptions): Promise<ApiResponse<T>>;
}

interface ApiResponse<T> {
  data: T;
  status: number;
  headers: Headers;
}

interface RequestOptions {
  retry?: boolean;
  cache?: boolean;
  timeout?: number;
}
```

### UI Components

```typescript
// src/components/ui/
interface ButtonProps {
  variant: 'primary' | 'secondary' | 'ghost' | 'danger';
  size: 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
  children: React.ReactNode;
  onClick?: () => void;
}

interface CardProps {
  variant?: 'default' | 'accent';
  padding?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
}

interface MetricCardProps {
  label: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  accent?: boolean;
  onClick?: () => void;
}

interface InputProps {
  type: 'text' | 'email' | 'password' | 'number';
  label?: string;
  error?: string;
  icon?: React.ReactNode;
  showPasswordToggle?: boolean;
}
```

---

## Data Models

### User & Authentication

```typescript
interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  tenantId: string;
  roles: Role[];
  permissions: Permission[];
  mfaEnabled: boolean;
  createdAt: string; // ISO 8601
  lastLoginAt: string; // ISO 8601
}

type CustomerRole = 'owner' | 'admin' | 'developer' | 'billing' | 'viewer';
type AdminRole = 'super_admin' | 'tenant_admin' | 'support_agent' | 'billing_admin' | 'viewer';
type Role = CustomerRole | AdminRole;

type Permission = 
  | 'team:manage' | 'team:view'
  | 'api_keys:create' | 'api_keys:rotate' | 'api_keys:revoke' | 'api_keys:view'
  | 'billing:manage' | 'billing:view'
  | 'usage:view'
  | 'settings:manage'
  | 'tenant:manage' | 'tenant:view' | 'tenant:delete'
  | 'impersonate:user'
  | 'system:configure';
```

### Tenant & API Keys

```typescript
interface Tenant {
  id: string;
  name: string;
  email: string;
  plan: Plan;
  status: 'active' | 'suspended' | 'trial';
  mrr: number;
  createdAt: string;
  lastActivityAt: string;
}

interface ApiKey {
  id: string;
  name: string;
  prefix: string; // First 8 chars
  scopes: ApiKeyScope[];
  createdAt: string;
  lastUsedAt?: string;
  expiresAt?: string;
  usageToday: number;
  usageThisMonth: number;
}

type ApiKeyScope = 'realtime:connect' | 'realtime:admin' | 'billing:read' | 'tenant:admin';
```

### Billing

```typescript
interface Plan {
  id: string;
  name: string;
  price: number;
  interval: 'monthly' | 'yearly';
  limits: PlanLimits;
  features: string[];
}

interface PlanLimits {
  apiRequests: number;
  audioMinutes: number;
  llmTokens: number;
  teamMembers: number;
}

interface Invoice {
  id: string;
  tenantId: string;
  amount: number;
  currency: string;
  status: 'paid' | 'pending' | 'failed' | 'refunded';
  createdAt: string;
  paidAt?: string;
  pdfUrl?: string;
}
```

### Audit Log

```typescript
interface AuditLogEntry {
  id: string;
  timestamp: string; // ISO 8601
  actorId: string;
  actorType: 'user' | 'admin' | 'system';
  action: AuditAction;
  target: string;
  targetType: 'tenant' | 'user' | 'api_key' | 'billing' | 'config';
  details: Record<string, unknown>;
  ipAddress: string;
  userAgent?: string;
}

type AuditAction = 
  | 'create' | 'update' | 'delete'
  | 'login' | 'logout' | 'mfa_enable' | 'mfa_disable'
  | 'impersonate' | 'suspend' | 'unsuspend'
  | 'refund' | 'credit_apply';
```

---


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the prework analysis, the following correctness properties have been identified:

### Property 1: Portal Route Separation
*For any* user with only customer portal roles, attempting to access any admin portal route SHALL result in a 403 Forbidden response.
**Validates: Requirements 1.4, 2.8**

### Property 2: JWT Token Claims Completeness
*For any* successful login, the resulting JWT access token SHALL contain all required claims: user_id, tenant_id, roles, and permissions.
**Validates: Requirements 2.2**

### Property 3: MFA Enforcement
*For any* user with MFA enabled, a login attempt with only username/password SHALL NOT grant full access until second factor is provided.
**Validates: Requirements 2.4**

### Property 4: Session Timeout Enforcement
*For any* session, if idle time exceeds 30 minutes OR total session time exceeds 8 hours, the session SHALL be invalidated.
**Validates: Requirements 2.5**

### Property 5: Token Refresh Transparency
*For any* expired access token with a valid refresh token, the system SHALL obtain a new access token without requiring user re-authentication.
**Validates: Requirements 2.6**

### Property 6: RBAC Permission Check
*For any* API request, the system SHALL verify the requesting user has the required permission before processing the request.
**Validates: Requirements 2.7**

### Property 7: Role Permission Union
*For any* user with multiple roles, the effective permissions SHALL be the union of all permissions from all assigned roles.
**Validates: Requirements 3.7**

### Property 8: Theme Persistence
*For any* theme selection (light, dark, system), the preference SHALL be persisted to localStorage and applied immediately without page reload.
**Validates: Requirements 5.2**

### Property 9: System Theme Detection
*For any* user with "system" theme mode selected, the resolved theme SHALL match the OS preference and update when OS preference changes.
**Validates: Requirements 5.7**

### Property 10: WCAG Contrast Compliance
*For any* text/background color combination in both themes, the contrast ratio SHALL meet WCAG 2.1 AA standards (4.5:1 for normal text, 3:1 for large text and UI components).
**Validates: Requirements 5.6, 18.6**

### Property 11: Dashboard Default Landing
*For any* successful customer portal login, the user SHALL be redirected to the dashboard page.
**Validates: Requirements 7.1**

### Property 12: Dashboard Auto-Refresh
*For any* dashboard view, data SHALL be refreshed every 60 seconds without full page reload.
**Validates: Requirements 7.8**

### Property 13: API Key Single Display
*For any* newly created API key, the full key value SHALL be displayed exactly once and never retrievable again.
**Validates: Requirements 8.3**

### Property 14: API Key Rotation Grace Period
*For any* API key rotation, the old key SHALL remain valid for exactly 24 hours after the new key is created.
**Validates: Requirements 8.5**

### Property 15: API Key Immediate Revocation
*For any* API key revocation, the key SHALL be immediately invalidated and all subsequent requests using that key SHALL fail.
**Validates: Requirements 8.6**

### Property 16: Team Invite Expiration
*For any* team invitation, the invite link SHALL expire after exactly 7 days.
**Validates: Requirements 10.2**

### Property 17: Team Size Limit Enforcement
*For any* team, the number of members SHALL NOT exceed the plan limit (Free: 3, Pro: 10, Enterprise: unlimited).
**Validates: Requirements 10.8**

### Property 18: Impersonation Audit Trail
*For any* user impersonation action, the system SHALL create an audit log entry containing admin_id, target_user_id, timestamp, and reason.
**Validates: Requirements 4.7, 13.8**

### Property 19: Refund Approval Threshold
*For any* refund request exceeding $100, the system SHALL require approval before processing.
**Validates: Requirements 14.3**

### Property 20: Plan Price Grandfathering
*For any* plan price change, existing subscribers SHALL continue at their original price until they change plans.
**Validates: Requirements 15.3**

### Property 21: Lago Sync Timing
*For any* plan change in the portal, the change SHALL be synchronized to Lago within 60 seconds.
**Validates: Requirements 15.8**

### Property 22: Audit Log Completeness
*For any* admin action (tenant changes, billing operations, user management, config changes), an audit log entry SHALL be created with timestamp, actor, action, target, details, and IP address.
**Validates: Requirements 17.1, 17.2**

### Property 23: JSON Serialization Round-Trip
*For any* data object sent to or received from the API, serializing to JSON and deserializing back SHALL produce an equivalent object.
**Validates: Requirements 20.1**

### Property 24: ISO 8601 Date Format
*For any* date value in API requests or responses, the format SHALL be ISO 8601 with timezone.
**Validates: Requirements 20.2**

### Property 25: Retry with Exponential Backoff
*For any* failed API request with retry enabled, the system SHALL retry up to 3 times with delays of 1s, 2s, and 4s.
**Validates: Requirements 20.4**

### Property 26: Currency Locale Formatting
*For any* currency value displayed to the user, the format SHALL match the user's locale settings.
**Validates: Requirements 20.6**

---

## Error Handling

### Error Types

```typescript
type ErrorCode = 
  | 'AUTH_INVALID_CREDENTIALS'
  | 'AUTH_MFA_REQUIRED'
  | 'AUTH_SESSION_EXPIRED'
  | 'AUTH_FORBIDDEN'
  | 'VALIDATION_ERROR'
  | 'NOT_FOUND'
  | 'RATE_LIMITED'
  | 'NETWORK_ERROR'
  | 'SERVER_ERROR';

interface AppError {
  code: ErrorCode;
  message: string;
  details?: Record<string, string>;
  retryable: boolean;
}
```

### Error Display Strategy

1. **Toast Notifications**: For transient errors (network issues, rate limits)
2. **Inline Validation**: For form field errors
3. **Error Pages**: For 404, 403, 500 errors
4. **Retry UI**: For retryable errors with exponential backoff

### Offline Handling

- Detect network status via `navigator.onLine`
- Display offline indicator in header
- Queue mutations for retry when online
- Show cached data with "stale" indicator

---

## Testing Strategy

### Dual Testing Approach

This project uses both unit tests and property-based tests for comprehensive coverage:

- **Unit tests** verify specific examples, edge cases, and integration points
- **Property-based tests** verify universal properties that should hold across all inputs

### Testing Framework

- **Test Runner**: Vitest
- **Component Testing**: React Testing Library
- **Property-Based Testing**: fast-check
- **Minimum Iterations**: 100 per property test

### Test Organization

```
src/
├── __tests__/
│   ├── unit/
│   │   ├── components/
│   │   ├── services/
│   │   └── utils/
│   └── properties/
│       ├── auth.property.test.ts
│       ├── theme.property.test.ts
│       ├── api-client.property.test.ts
│       └── serialization.property.test.ts
```

### Property Test Annotation Format

Each property-based test MUST be annotated with:
```typescript
/**
 * **Feature: saas-portal, Property 23: JSON Serialization Round-Trip**
 * For any data object, serializing to JSON and deserializing back
 * SHALL produce an equivalent object.
 * **Validates: Requirements 20.1**
 */
```

### Key Test Categories

1. **Authentication Properties**: Token claims, MFA enforcement, session timeout
2. **Authorization Properties**: Route separation, RBAC checks, permission union
3. **Theme Properties**: Persistence, system detection, contrast compliance
4. **Data Properties**: JSON round-trip, date formatting, currency locale
5. **Business Logic Properties**: Team limits, invite expiry, refund approval

---

## Security Considerations

### Authentication Security
- All tokens stored in httpOnly cookies (not localStorage)
- CSRF protection via SameSite cookies
- Token refresh happens server-side

### Authorization Security
- Permission checks on both client and server
- Route guards with middleware
- API routes validate JWT on every request

### Data Security
- No sensitive data in client-side storage
- API keys displayed once, never stored
- Audit logging for all sensitive operations

### Input Validation
- Zod schemas for all API inputs
- Sanitization of user-generated content
- Rate limiting on authentication endpoints

---

## Performance Considerations

### Caching Strategy
- SWR for data fetching with stale-while-revalidate
- Static generation for public pages
- ISR for semi-dynamic content

### Bundle Optimization
- Code splitting by route
- Dynamic imports for heavy components
- Tree shaking for unused code

### Rendering Strategy
- Server components for data fetching
- Client components only where interactivity needed
- Streaming for large data sets

---

## Accessibility

### WCAG 2.1 AA Compliance
- Color contrast ratios verified
- Keyboard navigation for all interactive elements
- Screen reader support with ARIA labels
- Focus indicators visible in both themes

### Reduced Motion
- Respect `prefers-reduced-motion` media query
- Disable animations when preference set
- Provide static alternatives

### Skip Links
- Skip to main content link
- Skip navigation for keyboard users
