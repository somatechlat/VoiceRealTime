"use client";

/**
 * Admin Dashboard Page
 * Implements Requirements 12.1-12.8: Admin dashboard with metrics, alerts, and tenant overview
 */

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  DollarSign,
  TrendingUp,
  Users,
  Zap,
} from "lucide-react";
import { AdminLayout } from "@/components/layout/AdminLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { apiRequest } from "@/lib/api";
import { formatNumber, formatCurrency, getStatusColor } from "@/lib/utils";

// Admin dashboard API types
interface AdminDashboardMetrics {
  total_tenants: number;
  active_tenants: number;
  total_mrr_cents: number;
  total_api_requests_today: number;
  total_audio_minutes_today: number;
  new_tenants_this_month: number;
  churn_rate_percent: number;
}

interface SystemHealth {
  overall: string;
  services: Record<string, string>;
  alerts: Alert[];
}

interface Alert {
  id: string;
  severity: "critical" | "warning" | "info";
  message: string;
  timestamp: string;
  tenant_id?: string;
}

interface TopTenant {
  id: string;
  name: string;
  plan: string;
  mrr_cents: number;
  api_requests_today: number;
}

interface AdminDashboardResponse {
  metrics: AdminDashboardMetrics;
  health: SystemHealth;
  top_tenants: TopTenant[];
}

// API functions
const adminDashboardApi = {
  getAll: (period: string) =>
    apiRequest<AdminDashboardResponse>(`/api/v1/admin/dashboard?period=${period}`),
};

function MetricCard({
  title,
  value,
  description,
  icon: Icon,
  trend,
}: {
  title: string;
  value: string | number;
  description?: string;
  icon: React.ElementType;
  trend?: { value: number; label: string };
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && (
          <p className="text-xs text-muted-foreground">{description}</p>
        )}
        {trend && (
          <p className={`text-xs ${trend.value >= 0 ? "text-green-600" : "text-red-600"}`}>
            {trend.value >= 0 ? "+" : ""}{trend.value}% {trend.label}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function AlertItem({ alert }: { alert: Alert }) {
  const severityColors = {
    critical: "bg-red-500",
    warning: "bg-yellow-500",
    info: "bg-blue-500",
  };

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg border">
      <div className={`w-2 h-2 rounded-full mt-2 ${severityColors[alert.severity]}`} />
      <div className="flex-1">
        <p className="text-sm font-medium">{alert.message}</p>
        <p className="text-xs text-muted-foreground">
          {new Date(alert.timestamp).toLocaleString()}
        </p>
      </div>
      <Badge variant={alert.severity === "critical" ? "destructive" : "secondary"}>
        {alert.severity}
      </Badge>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i}>
            <CardHeader className="pb-2">
              <Skeleton className="h-4 w-24" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-8 w-16 mb-1" />
              <Skeleton className="h-3 w-32" />
            </CardContent>
          </Card>
        ))}
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <Skeleton className="h-5 w-32" />
          </CardHeader>
          <CardContent className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-5 w-32" />
          </CardHeader>
          <CardContent className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function AdminDashboardPage() {
  const [period, setPeriod] = useState("7d");

  const { data, isLoading, error } = useQuery<AdminDashboardResponse>({
    queryKey: ["admin-dashboard", period],
    queryFn: () => adminDashboardApi.getAll(period),
    refetchInterval: 60000, // Refresh every 60 seconds
  });

  if (isLoading) {
    return (
      <AdminLayout title="Admin Dashboard" description="Platform overview">
        <DashboardSkeleton />
      </AdminLayout>
    );
  }

  if (error || !data) {
    return (
      <AdminLayout title="Admin Dashboard" description="Platform overview">
        <Card>
          <CardContent className="py-8 text-center">
            <p className="text-muted-foreground">Failed to load dashboard data</p>
            <Button variant="outline" className="mt-4" onClick={() => window.location.reload()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      </AdminLayout>
    );
  }

  const { metrics, health, top_tenants } = data;

  return (
    <AdminLayout title="Admin Dashboard" description="Platform overview">
      <div className="space-y-6">
        {/* Date Range Selector */}
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Overview</h2>
          <Select value={period} onValueChange={setPeriod}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="today">Today</SelectItem>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 90 days</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Key Metrics */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Total Tenants"
            value={formatNumber(metrics.total_tenants)}
            description={`${metrics.active_tenants} active`}
            icon={Users}
            trend={{ value: metrics.new_tenants_this_month, label: "new this month" }}
          />
          <MetricCard
            title="Monthly Revenue"
            value={formatCurrency(metrics.total_mrr_cents)}
            description="MRR"
            icon={DollarSign}
          />
          <MetricCard
            title="API Requests"
            value={formatNumber(metrics.total_api_requests_today)}
            description="Today"
            icon={Zap}
          />
          <MetricCard
            title="Audio Minutes"
            value={formatNumber(metrics.total_audio_minutes_today)}
            description="Today"
            icon={Activity}
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* System Health & Alerts */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" aria-hidden="true" />
                System Health
              </CardTitle>
              <CardDescription>
                Current status and alerts
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <span className="font-medium">Overall Status</span>
                <Badge className={getStatusColor(health.overall)}>
                  {health.overall}
                </Badge>
              </div>

              {Object.entries(health.services).map(([service, status]) => (
                <div key={service} className="flex items-center justify-between text-sm">
                  <span className="capitalize">{service}</span>
                  <Badge variant="outline" className={getStatusColor(status)}>
                    {status}
                  </Badge>
                </div>
              ))}

              {health.alerts.length > 0 && (
                <div className="pt-4 border-t space-y-2">
                  <h4 className="text-sm font-medium flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    Active Alerts ({health.alerts.length})
                  </h4>
                  {health.alerts.slice(0, 3).map((alert) => (
                    <AlertItem key={alert.id} alert={alert} />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Top Tenants */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" aria-hidden="true" />
                Top Tenants
              </CardTitle>
              <CardDescription>
                By usage and revenue
              </CardDescription>
            </CardHeader>
            <CardContent>
              {top_tenants.length === 0 ? (
                <p className="text-center text-muted-foreground py-4">
                  No tenant data available
                </p>
              ) : (
                <div className="space-y-4">
                  {top_tenants.map((tenant, index) => (
                    <div
                      key={tenant.id}
                      className="flex items-center gap-4 p-3 rounded-lg border"
                    >
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-sm font-medium">
                        {index + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{tenant.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {tenant.plan} • {formatNumber(tenant.api_requests_today)} requests today
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">
                          {formatCurrency(tenant.mrr_cents)}
                        </p>
                        <p className="text-xs text-muted-foreground">MRR</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Revenue & Churn */}
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Revenue Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 grid-cols-2">
                <div>
                  <p className="text-sm text-muted-foreground">Total MRR</p>
                  <p className="text-2xl font-bold">{formatCurrency(metrics.total_mrr_cents)}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Churn Rate</p>
                  <p className="text-2xl font-bold">{metrics.churn_rate_percent.toFixed(1)}%</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Tenant Growth</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 grid-cols-2">
                <div>
                  <p className="text-sm text-muted-foreground">New This Month</p>
                  <p className="text-2xl font-bold">{metrics.new_tenants_this_month}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Active Rate</p>
                  <p className="text-2xl font-bold">
                    {metrics.total_tenants > 0 
                      ? ((metrics.active_tenants / metrics.total_tenants) * 100).toFixed(0)
                      : 0}%
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </AdminLayout>
  );
}
