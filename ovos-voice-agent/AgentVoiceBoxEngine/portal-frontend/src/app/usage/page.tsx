"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from "recharts";
import { AudioLines, MessageSquare, Zap } from "lucide-react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { analyticsApi, UsageDataPoint } from "@/lib/api";
import { formatNumber } from "@/lib/utils";

type Period = "24h" | "7d" | "30d" | "90d";

function UsageChart({
  data,
  dataKey,
  color,
  title,
}: {
  data: UsageDataPoint[];
  dataKey: keyof UsageDataPoint;
  color: string;
  title: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id={`gradient-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
                }}
                className="text-xs"
              />
              <YAxis
                tickFormatter={(value) => formatNumber(value)}
                className="text-xs"
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="rounded-lg border bg-background p-2 shadow-sm">
                        <p className="text-sm font-medium">
                          {new Date(label).toLocaleDateString("en-US", {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {formatNumber(payload[0].value as number)}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Area
                type="monotone"
                dataKey={dataKey}
                stroke={color}
                fill={`url(#gradient-${dataKey})`}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

function CombinedChart({ data }: { data: UsageDataPoint[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Usage Overview</CardTitle>
        <CardDescription>All metrics over time</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
                }}
                className="text-xs"
              />
              <YAxis tickFormatter={(value) => formatNumber(value)} className="text-xs" />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="rounded-lg border bg-background p-3 shadow-sm">
                        <p className="text-sm font-medium mb-2">
                          {new Date(label).toLocaleDateString("en-US", {
                            month: "short",
                            day: "numeric",
                          })}
                        </p>
                        {payload.map((entry, index) => (
                          <p key={index} className="text-sm" style={{ color: entry.color }}>
                            {entry.name}: {formatNumber(entry.value as number)}
                          </p>
                        ))}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend />
              <Bar dataKey="api_requests" name="API Requests" fill="hsl(221.2 83.2% 53.3%)" />
              <Bar dataKey="audio_minutes" name="Audio Minutes" fill="hsl(142.1 76.2% 36.3%)" />
              <Bar dataKey="llm_tokens" name="LLM Tokens (K)" fill="hsl(24.6 95% 53.1%)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

function StatSummary({
  data,
  period,
}: {
  data: UsageDataPoint[];
  period: Period;
}) {
  const totals = data.reduce(
    (acc, point) => ({
      api_requests: acc.api_requests + point.api_requests,
      audio_minutes: acc.audio_minutes + point.audio_minutes,
      llm_tokens: acc.llm_tokens + point.llm_tokens,
    }),
    { api_requests: 0, audio_minutes: 0, llm_tokens: 0 }
  );

  const periodLabel = {
    "24h": "Last 24 Hours",
    "7d": "Last 7 Days",
    "30d": "Last 30 Days",
    "90d": "Last 90 Days",
  }[period];

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">API Requests</CardTitle>
          <Zap className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{formatNumber(totals.api_requests)}</div>
          <p className="text-xs text-muted-foreground">{periodLabel}</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Audio Minutes</CardTitle>
          <AudioLines className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{totals.audio_minutes.toFixed(1)}</div>
          <p className="text-xs text-muted-foreground">{periodLabel}</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">LLM Tokens</CardTitle>
          <MessageSquare className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{formatNumber(totals.llm_tokens)}</div>
          <p className="text-xs text-muted-foreground">{periodLabel}</p>
        </CardContent>
      </Card>
    </div>
  );
}

export default function UsagePage() {
  const [period, setPeriod] = useState<Period>("30d");

  const { data, isLoading } = useQuery<UsageDataPoint[]>({
    queryKey: ["usage-analytics", period],
    queryFn: () => analyticsApi.getUsageTimeSeries(period),
  });

  return (
    <DashboardLayout title="Usage Analytics" description="Monitor your API usage over time">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-muted-foreground">
              Track your API requests, audio processing, and token usage.
            </p>
          </div>
          <Select value={period} onValueChange={(value) => setPeriod(value as Period)}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select period" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
              <SelectItem value="90d">Last 90 Days</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {isLoading ? (
          <div className="space-y-6">
            <div className="grid gap-4 md:grid-cols-3">
              {[...Array(3)].map((_, i) => (
                <Card key={i}>
                  <CardHeader className="pb-2">
                    <Skeleton className="h-4 w-24" />
                  </CardHeader>
                  <CardContent>
                    <Skeleton className="h-8 w-16 mb-1" />
                    <Skeleton className="h-3 w-20" />
                  </CardContent>
                </Card>
              ))}
            </div>
            <Card>
              <CardHeader>
                <Skeleton className="h-5 w-32" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-[300px] w-full" />
              </CardContent>
            </Card>
          </div>
        ) : data && data.length > 0 ? (
          <>
            <StatSummary data={data} period={period} />
            <CombinedChart data={data} />
            <div className="grid gap-6 md:grid-cols-3">
              <UsageChart
                data={data}
                dataKey="api_requests"
                color="hsl(221.2 83.2% 53.3%)"
                title="API Requests"
              />
              <UsageChart
                data={data}
                dataKey="audio_minutes"
                color="hsl(142.1 76.2% 36.3%)"
                title="Audio Minutes"
              />
              <UsageChart
                data={data}
                dataKey="llm_tokens"
                color="hsl(24.6 95% 53.1%)"
                title="LLM Tokens"
              />
            </div>
          </>
        ) : (
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">No usage data available for this period</p>
              <p className="text-sm text-muted-foreground mt-2">
                Start making API calls to see your usage analytics
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
}
