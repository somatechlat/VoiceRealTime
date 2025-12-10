"use client";

/**
 * Customer Dashboard Page
 * Implements Requirements 7.1-7.8: Dashboard with usage, billing, health, and activity
 * Property 11: Dashboard Default Landing
 * Property 12: Dashboard Auto-Refresh (60 seconds)
 */

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import {
  Activity,
  AudioLines,
  CreditCard,
  Key,
  MessageSquare,
  Zap,
  RefreshCw,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { dashboardApi, DashboardResponse } from "@/lib/api";
import { formatNumber, formatCurrency, formatDateTime, getStatusColor } from "@/lib/utils";

// Auto-refresh interval: 60 seconds (Property 12)
const AUTO_REFRESH_INTERVAL_MS = 60 * 1000;

function StatCard({
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

function UsageProgress({
  label,
  current,
  limit,
  unit,
}: {
  label: string;
  current: number;
  limit: number;
  unit: string;
}) {
  const percentage = limit > 0 ? Math.min((current / limit) * 100, 100) : 0;
  const isWarning = percentage >= 80;
  const isCritical = percentage >= 95;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span className="text-muted-foreground">
          {formatNumber(current)} / {limit === -1 ? "∞" : formatNumber(limit)} {unit}
        </span>
      </div>
      <div className="h-2 rounded-full bg-muted overflow-hidden" role="progressbar" aria-valuenow={percentage} aria-valuemin={0} aria-valuemax={100}>
        <div
          className={`h-full transition-all ${
            isCritical ? "bg-red-500" : isWarning ? "bg-yellow-500" : "bg-primary"
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
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
              <Skeleton key={i} className="h-8 w-full" />
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

export default function DashboardPage() {
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [isManualRefreshing, setIsManualRefreshing] = useState(false);

  // Query with auto-refresh every 60 seconds (Property 12: Dashboard Auto-Refresh)
  const { data, isLoading, error, refetch, isFetching } = useQuery<DashboardResponse>({
    queryKey: ["dashboard"],
    queryFn: dashboardApi.getAll,
    refetchInterval: AUTO_REFRESH_INTERVAL_MS,
    refetchIntervalInBackground: false, // Only refresh when tab is active
    staleTime: 30000, // Consider data stale after 30 seconds
  });

  // Update last refresh time when data changes
  useEffect(() => {
    if (data && !isFetching) {
      setLastRefresh(new Date());
    }
  }, [data, isFetching]);

  // Manual refresh handler
  const handleManualRefresh = async () => {
    setIsManualRefreshing(true);
    await refetch();
    setIsManualRefreshing(false);
  };

  if (isLoading) {
    return (
      <DashboardLayout title="Dashboard" description="Overview of your account">
        <DashboardSkeleton />
      </DashboardLayout>
    );
  }

  if (error || !data) {
    return (
      <DashboardLayout title="Dashboard" description="Overview of your account">
        <Card>
          <CardContent className="py-8 text-center space-y-4">
            <p className="text-muted-foreground">Failed to load dashboard data</p>
            <Button variant="outline" onClick={handleManualRefresh}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Try Again
            </Button>
          </CardContent>
        </Card>
      </DashboardLayout>
    );
  }

  const { usage, billing, health, recent_activity } = data;

  return (
    <DashboardLayout title="Dashboard" description="Overview of your account">
      <div className="space-y-6">
        {/* Refresh indicator */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Last updated: {lastRefresh.toLocaleTimeString()}
          </p>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleManualRefresh}
            disabled={isManualRefreshing || isFetching}
            className="gap-2"
          >
            <RefreshCw 
              className={`h-4 w-4 ${(isManualRefreshing || isFetching) ? 'animate-spin' : ''}`} 
              aria-hidden="true" 
            />
            Refresh
          </Button>
        </div>

        {/* Stats Grid - Verve style metric cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="API Requests"
            value={formatNumber(usage.api_requests)}
            description="This billing period"
            icon={Zap}
          />
          <StatCard
            title="Audio Minutes"
            value={`${(usage.audio_minutes_input + usage.audio_minutes_output).toFixed(1)}`}
            description="Input + Output"
            icon={AudioLines}
          />
          <StatCard
            title="LLM Tokens"
            value={formatNumber(usage.llm_tokens_input + usage.llm_tokens_output)}
            description="Input + Output"
            icon={MessageSquare}
          />
          <StatCard
            title="Current Plan"
            value={billing.plan_name}
            description={formatCurrency(billing.amount_due_cents)}
            icon={CreditCard}
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Usage Card */}
          <Card>
            <CardHeader>
              <CardTitle>Usage This Period</CardTitle>
              <CardDescription>
                {new Date(usage.period_start).toLocaleDateString()} -{" "}
                {new Date(usage.period_end).toLocaleDateString()}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <UsageProgress
                label="API Requests"
                current={usage.api_requests}
                limit={10000}
                unit="requests"
              />
              <UsageProgress
                label="Audio Input"
                current={usage.audio_minutes_input}
                limit={1000}
                unit="min"
              />
              <UsageProgress
                label="Audio Output"
                current={usage.audio_minutes_output}
                limit={1000}
                unit="min"
              />
              <UsageProgress
                label="LLM Tokens"
                current={usage.llm_tokens_input + usage.llm_tokens_output}
                limit={1000000}
                unit="tokens"
              />
            </CardContent>
          </Card>

          {/* Health Status Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" aria-hidden="true" />
                System Health
              </CardTitle>
              <CardDescription>
                Status of AgentVoiceBox services
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-medium">Overall Status</span>
                  <Badge className={getStatusColor(health.overall)}>
                    {health.overall}
                  </Badge>
                </div>
                {Object.entries(health.services).map(([service, status]) => (
                  <div key={service} className="flex items-center justify-between text-sm">
                    <span className="capitalize">{service}</span>
                    <div className="flex items-center gap-2">
                      {health.latency_ms[service] > 0 && (
                        <span className="text-muted-foreground">
                          {health.latency_ms[service].toFixed(0)}ms
                        </span>
                      )}
                      <Badge variant="outline" className={getStatusColor(status)}>
                        {status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Billing Summary Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CreditCard className="h-5 w-5" aria-hidden="true" />
              Billing Summary
            </CardTitle>
            <CardDescription>
              Current plan and payment status
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <p className="text-sm text-muted-foreground">Current Plan</p>
                <p className="text-lg font-semibold">{billing.plan_name}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Amount Due</p>
                <p className="text-lg font-semibold">
                  {formatCurrency(billing.amount_due_cents)}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Next Billing</p>
                <p className="text-lg font-semibold">
                  {billing.next_billing_date 
                    ? new Date(billing.next_billing_date).toLocaleDateString()
                    : "N/A"}
                </p>
              </div>
            </div>
            <div className="mt-4 flex items-center gap-2">
              <Badge 
                variant={billing.payment_status === "current" ? "default" : "destructive"}
              >
                {billing.payment_status}
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Latest events in your account</CardDescription>
          </CardHeader>
          <CardContent>
            {recent_activity.length === 0 ? (
              <p className="text-center text-muted-foreground py-4">
                No recent activity
              </p>
            ) : (
              <div className="space-y-4">
                {recent_activity.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-start gap-4 border-b pb-4 last:border-0 last:pb-0"
                  >
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted">
                      <Key className="h-4 w-4" aria-hidden="true" />
                    </div>
                    <div className="flex-1 space-y-1">
                      <p className="text-sm font-medium">{item.description}</p>
                      <p className="text-xs text-muted-foreground">
                        {formatDateTime(item.timestamp)}
                      </p>
                    </div>
                    <Badge variant="outline">{item.type}</Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
