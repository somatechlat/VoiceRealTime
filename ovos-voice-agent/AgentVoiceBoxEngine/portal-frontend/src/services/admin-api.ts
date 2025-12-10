/**
 * Admin API Service - Connects to Portal API & Lago
 * Endpoints: Users, Billing, Audit, Monitoring, System Config
 * Portal API URL: http://localhost:25001
 * Lago URL: http://localhost:25005
 */

import { apiClient, ApiResponse } from './api-client';

// API URLs
const PORTAL_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:25001';
const LAGO_API_URL = process.env.NEXT_PUBLIC_LAGO_URL || 'http://localhost:25005';
const PROMETHEUS_URL = process.env.NEXT_PUBLIC_PROMETHEUS_URL || 'http://localhost:25008';
const KEYCLOAK_URL = process.env.NEXT_PUBLIC_KEYCLOAK_URL || 'http://localhost:25004';

// Create API clients
const portalClient = new (apiClient.constructor as typeof import('./api-client').ApiClient)(PORTAL_API_URL);

// ============================================================================
// Types - Users
// ============================================================================

export interface AdminUser {
  id: string;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  enabled: boolean;
  email_verified: boolean;
  tenant_id: string;
  roles: string[];
  created_at: number;
}

export interface CreateUserRequest {
  username: string;
  email: string;
  password: string;
  first_name?: string;
  last_name?: string;
  tenant_id?: string;
  roles?: string[];
  email_verified?: boolean;
  temporary_password?: boolean;
}

// ============================================================================
// Types - Billing (Lago)
// ============================================================================

export interface LagoCustomer {
  lago_id: string;
  external_id: string;
  name: string;
  email: string;
  currency: string;
  timezone: string;
  created_at: string;
}

export interface LagoSubscription {
  lago_id: string;
  external_id: string;
  external_customer_id: string;
  plan_code: string;
  status: 'active' | 'pending' | 'terminated' | 'canceled';
  name?: string;
  started_at?: string;
  ending_at?: string;
  created_at: string;
}

export interface LagoInvoice {
  lago_id: string;
  sequential_id: number;
  number: string;
  status: 'draft' | 'finalized' | 'voided';
  payment_status: string;
  currency: string;
  total_amount_cents: number;
  taxes_amount_cents: number;
  issuing_date?: string;
  payment_due_date?: string;
  file_url?: string;
}

export interface LagoPlan {
  lago_id: string;
  name: string;
  code: string;
  description?: string;
  amount_cents: number;
  amount_currency: string;
  interval: 'monthly' | 'yearly' | 'weekly';
  pay_in_advance: boolean;
  active_subscriptions_count: number;
  created_at: string;
}

export interface UsageEvent {
  transaction_id: string;
  external_customer_id: string;
  code: string;
  timestamp: number;
  properties: Record<string, unknown>;
}

// ============================================================================
// Types - Audit
// ============================================================================

export interface AuditLogEntry {
  id: number;
  tenant_id: string;
  actor_id?: string;
  actor_type: 'user' | 'admin' | 'system';
  action: string;
  resource_type: string;
  resource_id?: string;
  details: Record<string, unknown>;
  ip_address?: string;
  user_agent?: string;
  created_at: string;
}

// ============================================================================
// Types - Monitoring
// ============================================================================

export interface ServiceStatus {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  latency_ms: number;
  last_check: string;
  details?: Record<string, unknown>;
}

export interface SystemMetrics {
  cpu_percent: number;
  memory_percent: number;
  disk_percent: number;
  active_connections: number;
  requests_per_minute: number;
  error_rate: number;
}

export interface QueueMetrics {
  name: string;
  depth: number;
  processing_rate: number;
  error_rate: number;
}

// ============================================================================
// Users API
// ============================================================================

export const usersApi = {
  async list(params?: {
    first?: number;
    max?: number;
    search?: string;
    tenant_id?: string;
  }): Promise<ApiResponse<{ users: AdminUser[]; count: number }>> {
    const query = new URLSearchParams();
    if (params?.first) query.set('first', params.first.toString());
    if (params?.max) query.set('max', params.max.toString());
    if (params?.search) query.set('search', params.search);
    if (params?.tenant_id) query.set('tenant_id', params.tenant_id);
    
    return portalClient.get(`/v1/admin/users?${query}`);
  },

  async get(userId: string): Promise<ApiResponse<AdminUser>> {
    return portalClient.get(`/v1/admin/users/${userId}`);
  },

  async create(data: CreateUserRequest): Promise<ApiResponse<AdminUser>> {
    return portalClient.post('/v1/admin/users', data);
  },

  async update(userId: string, data: Partial<AdminUser>): Promise<ApiResponse<AdminUser>> {
    return portalClient.patch(`/v1/admin/users/${userId}`, data);
  },

  async delete(userId: string): Promise<ApiResponse<void>> {
    return portalClient.delete(`/v1/admin/users/${userId}`);
  },

  async deactivate(userId: string): Promise<ApiResponse<{ user: AdminUser; message: string }>> {
    return portalClient.post(`/v1/admin/users/${userId}/deactivate`, {});
  },

  async activate(userId: string): Promise<ApiResponse<{ user: AdminUser; message: string }>> {
    return portalClient.post(`/v1/admin/users/${userId}/activate`, {});
  },

  async getRoles(userId: string): Promise<ApiResponse<{ roles: string[] }>> {
    return portalClient.get(`/v1/admin/users/${userId}/roles`);
  },

  async assignRoles(userId: string, roles: string[]): Promise<ApiResponse<{ roles: string[] }>> {
    return portalClient.post(`/v1/admin/users/${userId}/roles`, { roles });
  },

  async removeRoles(userId: string, roles: string[]): Promise<ApiResponse<{ roles: string[] }>> {
    return portalClient.delete(`/v1/admin/users/${userId}/roles`);
  },

  async resetPassword(userId: string, password: string, temporary?: boolean): Promise<ApiResponse<{ message: string }>> {
    return portalClient.post(`/v1/admin/users/${userId}/reset-password`, { password, temporary });
  },
};

// ============================================================================
// Billing API (Lago)
// ============================================================================

export const billingApi = {
  // Customers
  async listCustomers(page?: number, perPage?: number): Promise<ApiResponse<{ customers: LagoCustomer[] }>> {
    return portalClient.get(`/v1/admin/billing/customers?page=${page || 1}&per_page=${perPage || 20}`);
  },

  async getCustomer(externalId: string): Promise<ApiResponse<LagoCustomer>> {
    return portalClient.get(`/v1/admin/billing/customers/${externalId}`);
  },

  async createCustomer(data: {
    external_id: string;
    name: string;
    email: string;
    currency?: string;
  }): Promise<ApiResponse<LagoCustomer>> {
    return portalClient.post('/v1/admin/billing/customers', data);
  },

  // Subscriptions
  async listSubscriptions(params?: {
    external_customer_id?: string;
    status?: string[];
    page?: number;
  }): Promise<ApiResponse<{ subscriptions: LagoSubscription[] }>> {
    const query = new URLSearchParams();
    if (params?.external_customer_id) query.set('external_customer_id', params.external_customer_id);
    if (params?.page) query.set('page', params.page.toString());
    
    return portalClient.get(`/v1/admin/billing/subscriptions?${query}`);
  },

  async createSubscription(data: {
    external_customer_id: string;
    plan_code: string;
    name?: string;
  }): Promise<ApiResponse<LagoSubscription>> {
    return portalClient.post('/v1/admin/billing/subscriptions', data);
  },

  async terminateSubscription(externalId: string): Promise<ApiResponse<LagoSubscription>> {
    return portalClient.delete(`/v1/admin/billing/subscriptions/${externalId}`);
  },

  // Invoices
  async listInvoices(params?: {
    external_customer_id?: string;
    status?: string;
    page?: number;
  }): Promise<ApiResponse<{ invoices: LagoInvoice[] }>> {
    const query = new URLSearchParams();
    if (params?.external_customer_id) query.set('external_customer_id', params.external_customer_id);
    if (params?.status) query.set('status', params.status);
    if (params?.page) query.set('page', params.page.toString());
    
    return portalClient.get(`/v1/admin/billing/invoices?${query}`);
  },

  async getInvoice(lagoId: string): Promise<ApiResponse<LagoInvoice>> {
    return portalClient.get(`/v1/admin/billing/invoices/${lagoId}`);
  },

  async downloadInvoice(lagoId: string): Promise<ApiResponse<{ file_url: string }>> {
    return portalClient.post(`/v1/admin/billing/invoices/${lagoId}/download`, {});
  },

  // Plans
  async listPlans(): Promise<ApiResponse<{ plans: LagoPlan[] }>> {
    return portalClient.get('/v1/admin/billing/plans');
  },

  async getPlan(code: string): Promise<ApiResponse<LagoPlan>> {
    return portalClient.get(`/v1/admin/billing/plans/${code}`);
  },

  async createPlan(data: {
    name: string;
    code: string;
    amount_cents: number;
    amount_currency: string;
    interval: string;
    description?: string;
  }): Promise<ApiResponse<LagoPlan>> {
    return portalClient.post('/v1/admin/billing/plans', data);
  },

  async updatePlan(code: string, data: Partial<LagoPlan>): Promise<ApiResponse<LagoPlan>> {
    return portalClient.put(`/v1/admin/billing/plans/${code}`, data);
  },

  // Usage
  async sendUsageEvent(event: UsageEvent): Promise<ApiResponse<void>> {
    return portalClient.post('/v1/admin/billing/events', event);
  },

  // Revenue metrics
  async getRevenueMetrics(): Promise<ApiResponse<{
    mrr: number;
    arr: number;
    total_customers: number;
    active_subscriptions: number;
    pending_invoices: number;
    failed_payments: number;
  }>> {
    return portalClient.get('/v1/admin/billing/metrics');
  },

  // Refunds
  async processRefund(invoiceId: string, data: {
    amount_cents: number;
    reason: string;
  }): Promise<ApiResponse<{ refund_id: string; status: string }>> {
    return portalClient.post(`/v1/admin/billing/invoices/${invoiceId}/refund`, data);
  },

  // Credits
  async applyCredit(customerId: string, data: {
    amount_cents: number;
    reason: string;
    expires_at?: string;
  }): Promise<ApiResponse<{ credit_id: string }>> {
    return portalClient.post(`/v1/admin/billing/customers/${customerId}/credits`, data);
  },
};

// ============================================================================
// Audit API
// ============================================================================

export const auditApi = {
  async list(params?: {
    tenant_id?: string;
    actor_id?: string;
    action?: string;
    resource_type?: string;
    start_date?: string;
    end_date?: string;
    page?: number;
    per_page?: number;
  }): Promise<ApiResponse<{ logs: AuditLogEntry[]; total: number }>> {
    const query = new URLSearchParams();
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value) query.set(key, value.toString());
    });
    
    return portalClient.get(`/v1/admin/audit?${query}`);
  },

  async export(params: {
    format: 'csv' | 'json';
    start_date?: string;
    end_date?: string;
  }): Promise<Blob> {
    const query = new URLSearchParams();
    query.set('format', params.format);
    if (params.start_date) query.set('start_date', params.start_date);
    if (params.end_date) query.set('end_date', params.end_date);
    
    const response = await fetch(`${PORTAL_API_URL}/v1/admin/audit/export?${query}`);
    return response.blob();
  },
};

// ============================================================================
// Monitoring API
// ============================================================================

export const monitoringApi = {
  async getServiceStatuses(): Promise<ApiResponse<ServiceStatus[]>> {
    return portalClient.get('/v1/admin/monitoring/services');
  },

  async getSystemMetrics(): Promise<ApiResponse<SystemMetrics>> {
    return portalClient.get('/v1/admin/monitoring/metrics');
  },

  async getQueueMetrics(): Promise<ApiResponse<QueueMetrics[]>> {
    return portalClient.get('/v1/admin/monitoring/queues');
  },

  async getDatabaseMetrics(): Promise<ApiResponse<{
    connections: number;
    max_connections: number;
    query_latency_ms: number;
    replication_lag_ms?: number;
  }>> {
    return portalClient.get('/v1/admin/monitoring/database');
  },

  async getErrorLogs(params?: {
    service?: string;
    severity?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
  }): Promise<ApiResponse<Array<{
    timestamp: string;
    service: string;
    severity: string;
    message: string;
    details?: Record<string, unknown>;
  }>>> {
    const query = new URLSearchParams();
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value) query.set(key, value.toString());
    });
    
    return portalClient.get(`/v1/admin/monitoring/errors?${query}`);
  },

  // Prometheus metrics
  async getPrometheusMetrics(): Promise<string> {
    const response = await fetch(`${PROMETHEUS_URL}/api/v1/query?query=up`);
    return response.text();
  },

  // Grafana dashboard URL
  getGrafanaUrl(): string {
    return process.env.NEXT_PUBLIC_GRAFANA_URL || 'http://localhost:25009';
  },
};

// ============================================================================
// System Config API
// ============================================================================

export const configApi = {
  async get(): Promise<ApiResponse<{
    redis: { url: string; max_connections: number };
    database: { pool_size: number; max_overflow: number };
    security: { rate_limits: { requests_per_minute: number; tokens_per_minute: number } };
    observability: { log_level: string; enable_tracing: boolean };
  }>> {
    return portalClient.get('/v1/admin/config');
  },

  async update(config: Record<string, unknown>): Promise<ApiResponse<void>> {
    return portalClient.put('/v1/admin/config', config);
  },

  async getRateLimits(): Promise<ApiResponse<{
    requests_per_minute: number;
    tokens_per_minute: number;
  }>> {
    return portalClient.get('/v1/admin/config/rate-limits');
  },

  async updateRateLimits(limits: {
    requests_per_minute?: number;
    tokens_per_minute?: number;
  }): Promise<ApiResponse<void>> {
    return portalClient.put('/v1/admin/config/rate-limits', limits);
  },
};

export { portalClient };
