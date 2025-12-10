# Requirements Document

## Introduction

This document specifies the complete requirements for **AgentVoiceBox SaaS Portal System** - a dual-portal architecture providing both Customer Portal (tenant self-service) and Admin Portal (platform operator administration). The system delivers enterprise-grade security, role-based access control, and a minimalist dark/light theme UI.

The portal system is designed as a **complete SaaS administration layer** with:
- Customer Portal for tenant self-service (API keys, billing, usage, team management)
- Admin Portal for platform operators (tenant management, system health, billing administration)
- Full-stack security with Keycloak integration (RBAC, SSO, MFA)
- Dark/light theme with elegant minimalist design
- Real-time data synchronization and responsive UI

## Glossary

- **Customer Portal**: Web application for tenant administrators and developers to manage their account, API keys, billing, and team
- **Admin Portal**: Web application for platform operators to manage all tenants, monitor system health, and administer billing
- **Tenant**: A paying customer organization with isolated resources
- **Platform Operator**: AgentVoiceBox staff with administrative access to manage the SaaS platform
- **RBAC**: Role-Based Access Control - permission system based on user roles
- **Theme**: Visual appearance mode (dark or light) affecting colors and contrast
- **Keycloak**: Identity and access management system providing authentication and authorization
- **Lago**: Billing engine for usage-based pricing and subscription management
- **JWT**: JSON Web Token - secure token format for authentication
- **MFA**: Multi-Factor Authentication - additional security layer beyond password

---

## Requirements

---

### Requirement 1: Portal Architecture & Separation

**User Story:** As a SaaS operator, I want separate Customer and Admin portals, so that tenant users cannot access platform administration functions.

#### Acceptance Criteria

1. THE system SHALL provide two distinct web applications: Customer Portal and Admin Portal
2. WHEN a user accesses Customer Portal THEN the system SHALL authenticate against tenant-specific Keycloak realm
3. WHEN a user accesses Admin Portal THEN the system SHALL authenticate against platform Keycloak realm with admin roles
4. THE system SHALL enforce complete separation: Customer Portal users SHALL NOT access Admin Portal routes
5. THE system SHALL share common UI components (theme, design system) between both portals
6. WHEN deploying THEN the system SHALL support both portals from single codebase with route-based separation
7. THE system SHALL use Next.js 14 App Router with server components for optimal performance

---

### Requirement 2: Authentication & Authorization

**User Story:** As a security administrator, I want robust authentication with role-based permissions, so that users can only access authorized functions.

#### Acceptance Criteria

1. THE system SHALL integrate with Keycloak for all authentication using OIDC protocol
2. WHEN a user logs in THEN the system SHALL obtain JWT access token with claims: user_id, tenant_id, roles, permissions
3. THE system SHALL support authentication methods: username/password, social login (Google, GitHub), SAML 2.0 SSO
4. WHEN MFA is enabled THEN the system SHALL require second factor (TOTP, WebAuthn) before granting access
5. THE system SHALL enforce session timeout: 30 minutes idle, 8 hours maximum
6. WHEN access token expires THEN the system SHALL silently refresh using refresh token without user interruption
7. THE system SHALL implement RBAC with permissions checked on every API call and UI component render
8. IF authorization fails THEN the system SHALL return 403 Forbidden and display appropriate error message

---

### Requirement 3: Customer Portal - Role Definitions

**User Story:** As a tenant administrator, I want to assign roles to team members, so that I can control who can perform which actions.

#### Acceptance Criteria

1. THE Customer Portal SHALL support roles: Owner, Admin, Developer, Billing, Viewer
2. THE Owner role SHALL have all permissions and ability to transfer ownership
3. THE Admin role SHALL have permissions: manage team, manage API keys, view billing, view usage
4. THE Developer role SHALL have permissions: create/rotate API keys, view usage, view documentation
5. THE Billing role SHALL have permissions: view/manage billing, view invoices, manage payment methods
6. THE Viewer role SHALL have permissions: view dashboard, view usage (read-only access)
7. WHEN a user has multiple roles THEN the system SHALL grant union of all role permissions
8. THE system SHALL display UI elements based on user permissions (hide unauthorized actions)

---

### Requirement 4: Admin Portal - Role Definitions

**User Story:** As a platform operator, I want granular admin roles, so that support staff have limited access compared to super admins.

#### Acceptance Criteria

1. THE Admin Portal SHALL support roles: Super Admin, Tenant Admin, Support Agent, Billing Admin, Viewer
2. THE Super Admin role SHALL have all permissions including system configuration
3. THE Tenant Admin role SHALL have permissions: create/suspend/delete tenants, manage tenant settings
4. THE Support Agent role SHALL have permissions: view tenant details, view sessions, impersonate users (with audit)
5. THE Billing Admin role SHALL have permissions: manage plans, process refunds, adjust credits, view all invoices
6. THE Viewer role SHALL have permissions: view dashboards, view metrics (read-only access)
7. WHEN impersonating a user THEN the system SHALL log action with admin_id, target_user_id, timestamp, reason
8. THE system SHALL require Super Admin approval for destructive actions (delete tenant, bulk operations)

---

### Requirement 5: Theme System (Dark/Light Mode)

**User Story:** As a user, I want to choose between dark and light themes, so that I can use the portal comfortably in any lighting condition.

#### Acceptance Criteria

1. THE system SHALL support three theme modes: Light, Dark, System (follows OS preference)
2. WHEN user selects theme THEN the system SHALL persist preference in localStorage and apply immediately
3. THE system SHALL use CSS custom properties for all theme colors enabling instant switching
4. THE Dark theme SHALL use colors: background #0a0a0f, card #111118, text #e4e4e7, primary #3b82f6
5. THE Light theme SHALL use colors: background #ffffff, card #f4f4f5, text #18181b, primary #2563eb
6. THE system SHALL ensure WCAG 2.1 AA contrast ratios in both themes (4.5:1 for text, 3:1 for UI)
7. WHEN System mode is selected THEN the system SHALL detect OS preference and update on OS theme change
8. THE theme toggle SHALL be accessible via keyboard and screen readers

---

### Requirement 6: Design System - Minimalist Aesthetic

**User Story:** As a user, I want a clean, uncluttered interface, so that I can focus on tasks without visual distraction.

#### Acceptance Criteria

1. THE system SHALL use minimalist design: generous whitespace, clear typography, subtle shadows
2. THE system SHALL use Inter font family for all text with sizes: 12px (small), 14px (body), 16px (large), 24px (heading)
3. THE system SHALL use consistent spacing scale: 4px, 8px, 12px, 16px, 24px, 32px, 48px
4. THE system SHALL use subtle animations: 150ms transitions, ease-out timing, no jarring movements
5. THE system SHALL use card-based layout with 8px border radius and subtle shadows
6. THE system SHALL display loading states with skeleton placeholders matching content shape
7. THE system SHALL use icon library (Lucide) with consistent 20px size and 1.5px stroke
8. THE system SHALL avoid visual clutter: no unnecessary borders, minimal color usage, clear hierarchy

---

### Requirement 7: Customer Portal - Dashboard

**User Story:** As a tenant user, I want a dashboard showing key metrics, so that I can quickly understand my account status.

#### Acceptance Criteria

1. WHEN user logs in THEN the system SHALL display dashboard as default landing page
2. THE dashboard SHALL display usage summary: API requests, audio minutes, LLM tokens (current period)
3. THE dashboard SHALL display billing summary: current plan, amount due, next billing date
4. THE dashboard SHALL display system health: API status (operational/degraded/down), latency metrics
5. THE dashboard SHALL display recent activity: last 10 API calls, key events, alerts
6. THE dashboard SHALL display usage chart: line graph of API calls over last 7 days
7. WHEN clicking metric card THEN the system SHALL navigate to detailed view
8. THE dashboard SHALL auto-refresh data every 60 seconds without full page reload

---

### Requirement 8: Customer Portal - API Key Management

**User Story:** As a developer, I want to manage API keys, so that I can securely integrate with the AgentVoiceBox API.

#### Acceptance Criteria

1. THE system SHALL display list of API keys with: name, prefix (first 8 chars), scopes, created date, last used
2. WHEN creating API key THEN the system SHALL require: name, scope selection, optional expiration
3. WHEN API key is created THEN the system SHALL display full key ONCE with copy button and warning
4. THE system SHALL support key scopes: realtime:connect, realtime:admin, billing:read, tenant:admin
5. WHEN rotating key THEN the system SHALL create new key and allow 24-hour grace period for old key
6. WHEN revoking key THEN the system SHALL require confirmation and immediately invalidate
7. THE system SHALL display per-key usage: requests today, requests this month, last used timestamp
8. THE system SHALL support bulk operations: revoke multiple keys, export key list (without secrets)

---

### Requirement 9: Customer Portal - Billing & Payments

**User Story:** As a billing administrator, I want to manage subscription and payments, so that I can control costs and maintain service.

#### Acceptance Criteria

1. THE billing page SHALL display: current plan details, usage vs limits, projected cost
2. THE system SHALL display plan comparison: Free, Pro, Enterprise with feature matrix
3. WHEN upgrading plan THEN the system SHALL show prorated cost and apply immediately
4. WHEN downgrading plan THEN the system SHALL apply at end of billing period with confirmation
5. THE system SHALL display invoice history: date, amount, status, download PDF link
6. THE system SHALL display payment methods: card (last 4, expiry), PayPal, with default indicator
7. WHEN adding payment method THEN the system SHALL use Stripe Elements (PCI compliant)
8. THE system SHALL display usage breakdown: API calls, audio minutes, tokens with unit costs

---

### Requirement 10: Customer Portal - Team Management

**User Story:** As a tenant administrator, I want to manage team members, so that I can control access to our account.

#### Acceptance Criteria

1. THE team page SHALL display members: name, email, role, status, last login
2. WHEN inviting member THEN the system SHALL send email with secure invite link (expires 7 days)
3. THE system SHALL support role assignment: select from Owner, Admin, Developer, Billing, Viewer
4. WHEN changing member role THEN the system SHALL apply immediately and log action
5. WHEN removing member THEN the system SHALL require confirmation and revoke all sessions
6. THE system SHALL support ownership transfer: requires current owner confirmation and new owner acceptance
7. THE system SHALL display pending invites with option to resend or cancel
8. THE system SHALL enforce maximum team size based on plan (Free: 3, Pro: 10, Enterprise: unlimited)

---

### Requirement 11: Customer Portal - Settings

**User Story:** As a tenant administrator, I want to configure account settings, so that I can customize notifications and integrations.

#### Acceptance Criteria

1. THE settings page SHALL have sections: Profile, Notifications, Webhooks, Security
2. THE Profile section SHALL allow editing: organization name, email, timezone, logo upload
3. THE Notifications section SHALL allow toggling: billing alerts, usage alerts, security alerts, product updates
4. THE Webhooks section SHALL allow: create webhook (URL, events), test webhook, view delivery history
5. THE Security section SHALL display: active sessions, login history, MFA status
6. WHEN enabling MFA THEN the system SHALL guide through TOTP setup with QR code and backup codes
7. THE system SHALL allow terminating other sessions (force logout)
8. THE system SHALL display audit log: last 100 actions with timestamp, action, IP address

---

### Requirement 12: Admin Portal - Dashboard

**User Story:** As a platform operator, I want a system-wide dashboard, so that I can monitor platform health and key metrics.

#### Acceptance Criteria

1. THE admin dashboard SHALL display: total tenants, active sessions, API requests/min, error rate
2. THE dashboard SHALL display system health: all services status, database connections, queue depth
3. THE dashboard SHALL display revenue metrics: MRR, ARR, churn rate, new signups (today/week/month)
4. THE dashboard SHALL display usage heatmap: requests by hour over last 7 days
5. THE dashboard SHALL display alerts: critical issues, tenants approaching limits, failed payments
6. THE dashboard SHALL display top tenants: by usage, by revenue, by growth
7. WHEN clicking alert THEN the system SHALL navigate to relevant detail view
8. THE dashboard SHALL support date range selection for metrics (today, 7d, 30d, 90d, custom)

---

### Requirement 13: Admin Portal - Tenant Management

**User Story:** As a platform operator, I want to manage all tenants, so that I can onboard, support, and administer customer accounts.

#### Acceptance Criteria

1. THE tenant list SHALL display: name, plan, status, MRR, created date, last activity
2. THE system SHALL support search: by name, email, tenant_id with instant results
3. THE system SHALL support filters: by plan, by status (active/suspended/trial), by date range
4. WHEN viewing tenant detail THEN the system SHALL display: profile, usage, billing, team, API keys, audit log
5. THE system SHALL allow: suspend tenant (with reason), unsuspend, delete (with confirmation)
6. THE system SHALL allow: change plan, apply credits, extend trial, adjust limits
7. WHEN suspending tenant THEN the system SHALL send notification email and close active sessions
8. THE system SHALL support impersonation: login as tenant user for support (with audit trail)

---

### Requirement 14: Admin Portal - Billing Administration

**User Story:** As a billing administrator, I want to manage all billing operations, so that I can handle refunds, credits, and disputes.

#### Acceptance Criteria

1. THE billing admin page SHALL display: total revenue, pending payments, failed payments, refunds
2. THE system SHALL display all invoices with filters: status, date range, tenant, amount
3. WHEN processing refund THEN the system SHALL require: amount, reason, approval (for amounts > $100)
4. THE system SHALL allow applying credits: amount, reason, expiration (optional)
5. THE system SHALL display payment failures with: tenant, amount, failure reason, retry count
6. WHEN retrying payment THEN the system SHALL attempt charge and update status
7. THE system SHALL support manual invoice creation for custom billing arrangements
8. THE system SHALL display revenue reports: by plan, by period, by payment method

---

### Requirement 15: Admin Portal - Plan Management

**User Story:** As a platform operator, I want to manage subscription plans, so that I can adjust pricing and features.

#### Acceptance Criteria

1. THE plan management page SHALL display all plans: name, price, limits, subscriber count
2. THE system SHALL allow editing plan: name, description, price, limits, features
3. WHEN changing plan price THEN the system SHALL apply to new subscribers only (grandfather existing)
4. THE system SHALL allow creating new plans with: name, price, billing interval, limits, features
5. THE system SHALL allow deprecating plans: hide from new signups, migrate existing subscribers
6. THE system SHALL display plan analytics: subscribers, revenue, churn, upgrades/downgrades
7. WHEN creating promotional plan THEN the system SHALL support: discount percentage, duration, coupon codes
8. THE system SHALL sync plan changes to Lago billing system within 60 seconds

---

### Requirement 16: Admin Portal - System Monitoring

**User Story:** As a platform operator, I want to monitor system health, so that I can identify and resolve issues quickly.

#### Acceptance Criteria

1. THE monitoring page SHALL display: service status grid (gateway, workers, database, cache, billing)
2. THE system SHALL display metrics: CPU, memory, disk, network for each service
3. THE system SHALL display queue metrics: depth, processing rate, error rate for STT/TTS/LLM queues
4. THE system SHALL display database metrics: connections, query latency, replication lag
5. THE system SHALL display error logs: filterable by service, severity, time range
6. WHEN service is unhealthy THEN the system SHALL display alert with details and suggested actions
7. THE system SHALL support drill-down: click service to see detailed metrics and logs
8. THE system SHALL integrate with Grafana for advanced dashboards (link out)

---

### Requirement 17: Admin Portal - Audit & Compliance

**User Story:** As a compliance officer, I want comprehensive audit logs, so that I can track all administrative actions.

#### Acceptance Criteria

1. THE audit log SHALL record: timestamp, actor (user_id), action, target, details, IP address
2. THE system SHALL log all admin actions: tenant changes, billing operations, user management, config changes
3. THE system SHALL support search: by actor, action type, target, date range
4. THE system SHALL support export: CSV, JSON formats for compliance reporting
5. THE system SHALL retain audit logs for 7 years (configurable per compliance requirements)
6. WHEN viewing sensitive data THEN the system SHALL log access with reason field
7. THE system SHALL display login history: all admin logins with IP, device, location
8. THE system SHALL alert on suspicious activity: multiple failed logins, unusual access patterns

---

### Requirement 18: Responsive Design & Accessibility

**User Story:** As a user, I want to access the portal from any device, so that I can manage my account on mobile or desktop.

#### Acceptance Criteria

1. THE system SHALL be fully responsive: desktop (1200px+), tablet (768-1199px), mobile (320-767px)
2. THE system SHALL use mobile-first CSS with progressive enhancement for larger screens
3. THE navigation SHALL collapse to hamburger menu on mobile with slide-out drawer
4. THE data tables SHALL transform to card layout on mobile for readability
5. THE system SHALL support touch interactions: swipe, tap, long-press where appropriate
6. THE system SHALL meet WCAG 2.1 AA: keyboard navigation, screen reader support, focus indicators
7. THE system SHALL support reduced motion preference for users with vestibular disorders
8. THE system SHALL provide skip links for keyboard users to bypass navigation

---

### Requirement 19: Error Handling & User Feedback

**User Story:** As a user, I want clear error messages and feedback, so that I can understand and resolve issues.

#### Acceptance Criteria

1. WHEN an error occurs THEN the system SHALL display user-friendly message with suggested action
2. THE system SHALL use toast notifications for: success, warning, error, info messages
3. THE system SHALL display form validation errors inline with specific field guidance
4. WHEN API request fails THEN the system SHALL show retry option and error details (expandable)
5. THE system SHALL display loading states: skeleton screens for initial load, spinners for actions
6. WHEN network is offline THEN the system SHALL display offline indicator and queue actions
7. THE system SHALL log client-side errors to monitoring system for debugging
8. THE system SHALL provide "Report Issue" link with pre-filled context for support tickets

---

### Requirement 20: Data Serialization & API Communication

**User Story:** As a developer, I want reliable API communication, so that the portal displays accurate real-time data.

#### Acceptance Criteria

1. THE system SHALL use JSON for all API request/response serialization
2. WHEN serializing dates THEN the system SHALL use ISO 8601 format with timezone
3. WHEN deserializing API response THEN the system SHALL validate against TypeScript types
4. THE system SHALL implement request retry with exponential backoff (3 attempts, 1s/2s/4s delays)
5. THE system SHALL cache GET responses with SWR pattern (stale-while-revalidate)
6. WHEN displaying currency THEN the system SHALL format according to user locale
7. THE system SHALL handle pagination: cursor-based for lists, with infinite scroll option
8. THE system SHALL implement optimistic updates for better perceived performance

---

## Design References

Design guidelines are documented in `.kiro/specs/saas-portal/design-guidelines.md` based on:
- Fish Audio SaaS Dashboard (dark theme reference)
- Twisty Dashboard (light theme reference)
- Modern minimalist SaaS UI patterns

Key design principles:
- Simple, elegant, human-friendly
- Clear icons and readable typography
- Generous whitespace
- Card-based layouts with subtle shadows
- Smooth 150ms transitions
- WCAG 2.1 AA accessibility compliance

