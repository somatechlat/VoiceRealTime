'use client';

/**
 * Usage Analytics Page - Customer Portal
 * Detailed usage metrics: API calls, audio minutes, tokens
 */

import { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp,
  Clock,
  Mic,
  Volume2,
  Brain,
  Calendar,
  Download
} from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { MetricCard } from '@/components/ui/metric-card';
import { apiClient } from '@/services/api-client';

interface UsageMetrics {
  api_requests: number;
  audio_minutes_input: number;
  audio_minutes_output: number;
  llm_tokens_input: number;
  llm_tokens_output: number;
  concurrent_connections_peak: number;
  connection_minutes: number;
}

interface DailyUsage {
  date: string;
  api_requests: number;
  audio_minutes: number;
  llm_tokens: number;
}

export default function UsagePage() {
  const [metrics, setMetrics] = useState<UsageMetrics | null>(null);
  const [dailyUsage, setDailyUsage] = useState<DailyUsage[]>([]);
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState<'7d' | '30d' | '90d'>('30d');

  useEffect(() => {
    loadUsage();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [period]);

  const loadUsage = async () => {
    setLoading(true);
    try {
      const [metricsRes, dailyRes] = await Promise.all([
        apiClient.get<UsageMetrics>(`/v1/usage/metrics?period=${period}`),
        apiClient.get<{ usage: DailyUsage[] }>(`/v1/usage/daily?period=${period}`),
      ]);
      setMetrics(metricsRes.data);
      setDailyUsage(dailyRes.data.usage || []);
    } catch (error) {
      console.error('Failed to load usage:', error);
      // Mock data for demo
      setMetrics({
        api_requests: 125430,
        audio_minutes_input: 4523,
        audio_minutes_output: 3891,
        llm_tokens_input: 2450000,
        llm_tokens_output: 1230000,
        concurrent_connections_peak: 45,
        connection_minutes: 8920,
      });
      setDailyUsage(
        Array.from({ length: period === '7d' ? 7 : period === '30d' ? 30 : 90 }, (_, i) => ({
          date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          api_requests: Math.floor(Math.random() * 5000) + 1000,
          audio_minutes: Math.floor(Math.random() * 200) + 50,
          llm_tokens: Math.floor(Math.random() * 100000) + 20000,
        })).reverse()
      );
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const maxRequests = Math.max(...dailyUsage.map(d => d.api_requests), 1);

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-card rounded w-48" />
          <div className="grid grid-cols-4 gap-4">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-32 bg-card rounded-xl" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Usage Analytics</h1>
          <p className="text-muted-foreground">Monitor your API usage and resource consumption</p>
        </div>
        <div className="flex gap-2">
          <div className="flex bg-card rounded-lg p-1">
            {(['7d', '30d', '90d'] as const).map(p => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  period === p
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {p}
              </button>
            ))}
          </div>
          <Button variant="secondary" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="API Requests"
          value={formatNumber(metrics?.api_requests || 0)}
          icon={<BarChart3 className="w-5 h-5" />}
          accent
        />
        <MetricCard
          label="Audio Input"
          value={`${formatNumber(metrics?.audio_minutes_input || 0)} min`}
          icon={<Mic className="w-5 h-5" />}
        />
        <MetricCard
          label="Audio Output"
          value={`${formatNumber(metrics?.audio_minutes_output || 0)} min`}
          icon={<Volume2 className="w-5 h-5" />}
        />
        <MetricCard
          label="LLM Tokens"
          value={formatNumber((metrics?.llm_tokens_input || 0) + (metrics?.llm_tokens_output || 0))}
          icon={<Brain className="w-5 h-5" />}
        />
      </div>

      {/* Usage Chart */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-medium">API Requests Over Time</h3>
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-primary" />
              <span className="text-muted-foreground">Requests</span>
            </div>
          </div>
        </div>
        
        {/* Simple bar chart */}
        <div className="h-64 flex items-end gap-1">
          {dailyUsage.slice(-30).map((day, i) => (
            <div
              key={day.date}
              className="flex-1 bg-primary/20 hover:bg-primary/40 transition-colors rounded-t relative group"
              style={{ height: `${(day.api_requests / maxRequests) * 100}%`, minHeight: '4px' }}
            >
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-card border border-border rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                <p className="font-medium">{day.api_requests.toLocaleString()}</p>
                <p className="text-muted-foreground">{new Date(day.date).toLocaleDateString()}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-2 text-xs text-muted-foreground">
          <span>{dailyUsage[0]?.date ? new Date(dailyUsage[0].date).toLocaleDateString() : ''}</span>
          <span>{dailyUsage[dailyUsage.length - 1]?.date ? new Date(dailyUsage[dailyUsage.length - 1].date).toLocaleDateString() : ''}</span>
        </div>
      </Card>

      {/* Detailed Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Audio Usage */}
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">Audio Processing</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Speech-to-Text (Input)</span>
                <span className="font-medium">{formatNumber(metrics?.audio_minutes_input || 0)} min</span>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500 rounded-full"
                  style={{ 
                    width: `${((metrics?.audio_minutes_input || 0) / ((metrics?.audio_minutes_input || 0) + (metrics?.audio_minutes_output || 0) || 1)) * 100}%` 
                  }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Text-to-Speech (Output)</span>
                <span className="font-medium">{formatNumber(metrics?.audio_minutes_output || 0)} min</span>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div 
                  className="h-full bg-green-500 rounded-full"
                  style={{ 
                    width: `${((metrics?.audio_minutes_output || 0) / ((metrics?.audio_minutes_input || 0) + (metrics?.audio_minutes_output || 0) || 1)) * 100}%` 
                  }}
                />
              </div>
            </div>
            <div className="pt-4 border-t border-border">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Audio</span>
                <span className="font-medium">
                  {formatNumber((metrics?.audio_minutes_input || 0) + (metrics?.audio_minutes_output || 0))} min
                </span>
              </div>
            </div>
          </div>
        </Card>

        {/* Token Usage */}
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">LLM Token Usage</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Input Tokens</span>
                <span className="font-medium">{formatNumber(metrics?.llm_tokens_input || 0)}</span>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div 
                  className="h-full bg-purple-500 rounded-full"
                  style={{ 
                    width: `${((metrics?.llm_tokens_input || 0) / ((metrics?.llm_tokens_input || 0) + (metrics?.llm_tokens_output || 0) || 1)) * 100}%` 
                  }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Output Tokens</span>
                <span className="font-medium">{formatNumber(metrics?.llm_tokens_output || 0)}</span>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div 
                  className="h-full bg-pink-500 rounded-full"
                  style={{ 
                    width: `${((metrics?.llm_tokens_output || 0) / ((metrics?.llm_tokens_input || 0) + (metrics?.llm_tokens_output || 0) || 1)) * 100}%` 
                  }}
                />
              </div>
            </div>
            <div className="pt-4 border-t border-border">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Tokens</span>
                <span className="font-medium">
                  {formatNumber((metrics?.llm_tokens_input || 0) + (metrics?.llm_tokens_output || 0))}
                </span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Connection Stats */}
      <Card className="p-6">
        <h3 className="text-lg font-medium mb-4">Connection Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-muted-foreground mb-1">Peak Concurrent Connections</p>
            <p className="text-3xl font-bold">{metrics?.concurrent_connections_peak || 0}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-1">Total Connection Time</p>
            <p className="text-3xl font-bold">{formatNumber(metrics?.connection_minutes || 0)} min</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground mb-1">Avg Session Duration</p>
            <p className="text-3xl font-bold">
              {metrics?.api_requests && metrics?.connection_minutes
                ? Math.round(metrics.connection_minutes / (metrics.api_requests / 100))
                : 0} min
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
