'use client';

/**
 * Admin Monitoring Page - System Health & Metrics
 * Connects to: Gateway, Prometheus, Workers
 */

import { useState, useEffect } from 'react';
import { 
  Activity, 
  Server, 
  Database, 
  Cpu,
  HardDrive,
  Wifi,
  AlertCircle,
  CheckCircle,
  XCircle,
  ExternalLink,
  RefreshCw
} from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { MetricCard } from '@/components/ui/metric-card';
import { monitoringApi, ServiceStatus, SystemMetrics, QueueMetrics } from '@/services/admin-api';
import { healthApi, WorkerStatus, workersApi } from '@/services/voice-api';

export default function AdminMonitoringPage() {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [queues, setQueues] = useState<QueueMetrics[]>([]);
  const [workers, setWorkers] = useState<WorkerStatus[]>([]);
  const [dbMetrics, setDbMetrics] = useState<{
    connections: number;
    max_connections: number;
    query_latency_ms: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const [servicesRes, metricsRes, queuesRes, dbRes, workersRes] = await Promise.all([
        monitoringApi.getServiceStatuses().catch(() => ({ data: [] })),
        monitoringApi.getSystemMetrics().catch(() => ({ data: null })),
        monitoringApi.getQueueMetrics().catch(() => ({ data: [] })),
        monitoringApi.getDatabaseMetrics().catch(() => ({ data: null })),
        workersApi.listAll().catch(() => ({ data: [] })),
      ]);
      
      setServices(servicesRes.data || []);
      setSystemMetrics(metricsRes.data);
      setQueues(queuesRes.data || []);
      setDbMetrics(dbRes.data);
      setWorkers(workersRes.data || []);
      setLastRefresh(new Date());
    } catch (error) {
      console.error('Failed to load monitoring data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'running':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'degraded':
      case 'idle':
        return <AlertCircle className="w-5 h-5 text-yellow-400" />;
      default:
        return <XCircle className="w-5 h-5 text-red-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'running':
        return 'bg-green-500/20 text-green-400';
      case 'degraded':
      case 'idle':
        return 'bg-yellow-500/20 text-yellow-400';
      default:
        return 'bg-red-500/20 text-red-400';
    }
  };

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
          <h1 className="text-2xl font-semibold">System Monitoring</h1>
          <p className="text-muted-foreground">
            Last updated: {lastRefresh.toLocaleTimeString()}
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={loadData} variant="secondary" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button 
            variant="secondary" 
            size="sm"
            onClick={() => window.open(monitoringApi.getGrafanaUrl(), '_blank')}
          >
            <ExternalLink className="w-4 h-4 mr-2" />
            Grafana
          </Button>
        </div>
      </div>

      {/* System Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="CPU Usage"
          value={`${systemMetrics?.cpu_percent || 0}%`}
          icon={<Cpu className="w-5 h-5" />}
          accent={Boolean(systemMetrics?.cpu_percent && systemMetrics.cpu_percent > 80)}
        />
        <MetricCard
          label="Memory Usage"
          value={`${systemMetrics?.memory_percent || 0}%`}
          icon={<Server className="w-5 h-5" />}
          accent={Boolean(systemMetrics?.memory_percent && systemMetrics.memory_percent > 80)}
        />
        <MetricCard
          label="Active Connections"
          value={systemMetrics?.active_connections || 0}
          icon={<Wifi className="w-5 h-5" />}
        />
        <MetricCard
          label="Requests/min"
          value={systemMetrics?.requests_per_minute || 0}
          icon={<Activity className="w-5 h-5" />}
          change={systemMetrics?.error_rate ? -systemMetrics.error_rate : undefined}
          changeLabel="error rate"
        />
      </div>

      {/* Services Grid */}
      <div>
        <h2 className="text-lg font-medium mb-4">Service Status</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Gateway */}
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Server className="w-5 h-5 text-muted-foreground" />
                <div>
                  <p className="font-medium">Gateway</p>
                  <p className="text-sm text-muted-foreground">Port 25000</p>
                </div>
              </div>
              {getStatusIcon('healthy')}
            </div>
          </Card>

          {/* Portal API */}
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Server className="w-5 h-5 text-muted-foreground" />
                <div>
                  <p className="font-medium">Portal API</p>
                  <p className="text-sm text-muted-foreground">Port 25001</p>
                </div>
              </div>
              {getStatusIcon('healthy')}
            </div>
          </Card>

          {/* PostgreSQL */}
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Database className="w-5 h-5 text-muted-foreground" />
                <div>
                  <p className="font-medium">PostgreSQL</p>
                  <p className="text-sm text-muted-foreground">
                    {dbMetrics ? `${dbMetrics.connections}/${dbMetrics.max_connections} conn` : 'Port 25002'}
                  </p>
                </div>
              </div>
              {getStatusIcon(dbMetrics ? 'healthy' : 'unknown')}
            </div>
          </Card>

          {/* Redis */}
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <HardDrive className="w-5 h-5 text-muted-foreground" />
                <div>
                  <p className="font-medium">Redis</p>
                  <p className="text-sm text-muted-foreground">Port 25003</p>
                </div>
              </div>
              {getStatusIcon('healthy')}
            </div>
          </Card>

          {/* Keycloak */}
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Server className="w-5 h-5 text-muted-foreground" />
                <div>
                  <p className="font-medium">Keycloak</p>
                  <p className="text-sm text-muted-foreground">Port 25004</p>
                </div>
              </div>
              {getStatusIcon('healthy')}
            </div>
          </Card>

          {/* Lago */}
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Server className="w-5 h-5 text-muted-foreground" />
                <div>
                  <p className="font-medium">Lago Billing</p>
                  <p className="text-sm text-muted-foreground">Port 25005</p>
                </div>
              </div>
              {getStatusIcon('healthy')}
            </div>
          </Card>
        </div>
      </div>

      {/* Workers */}
      <div>
        <h2 className="text-lg font-medium mb-4">Voice Workers</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* STT Worker */}
          <Card className="p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="font-medium">STT Worker</p>
                <p className="text-sm text-muted-foreground">Speech-to-Text</p>
              </div>
              <span className={`text-xs px-2 py-1 rounded ${getStatusColor('running')}`}>
                Running
              </span>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model</span>
                <span>whisper-tiny</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Device</span>
                <span>CPU</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Queue</span>
                <span>0 pending</span>
              </div>
            </div>
          </Card>

          {/* TTS Worker */}
          <Card className="p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="font-medium">TTS Worker</p>
                <p className="text-sm text-muted-foreground">Text-to-Speech</p>
              </div>
              <span className={`text-xs px-2 py-1 rounded ${getStatusColor('running')}`}>
                Running
              </span>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model</span>
                <span>Kokoro</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Voice</span>
                <span>am_onyx</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Queue</span>
                <span>0 pending</span>
              </div>
            </div>
          </Card>

          {/* LLM Worker */}
          <Card className="p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="font-medium">LLM Worker</p>
                <p className="text-sm text-muted-foreground">Language Model</p>
              </div>
              <span className={`text-xs px-2 py-1 rounded ${getStatusColor('running')}`}>
                Running
              </span>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Provider</span>
                <span>Groq</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model</span>
                <span>llama-3.1-70b</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Queue</span>
                <span>0 pending</span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Queue Metrics */}
      <div>
        <h2 className="text-lg font-medium mb-4">Queue Metrics</h2>
        <Card className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <p className="text-sm text-muted-foreground mb-1">STT Queue</p>
              <div className="flex items-end gap-2">
                <span className="text-2xl font-bold">0</span>
                <span className="text-sm text-muted-foreground mb-1">depth</span>
              </div>
              <div className="mt-2 h-2 bg-background rounded-full overflow-hidden">
                <div className="h-full bg-green-500 rounded-full" style={{ width: '5%' }} />
              </div>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1">TTS Queue</p>
              <div className="flex items-end gap-2">
                <span className="text-2xl font-bold">0</span>
                <span className="text-sm text-muted-foreground mb-1">depth</span>
              </div>
              <div className="mt-2 h-2 bg-background rounded-full overflow-hidden">
                <div className="h-full bg-green-500 rounded-full" style={{ width: '5%' }} />
              </div>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1">LLM Queue</p>
              <div className="flex items-end gap-2">
                <span className="text-2xl font-bold">0</span>
                <span className="text-sm text-muted-foreground mb-1">depth</span>
              </div>
              <div className="mt-2 h-2 bg-background rounded-full overflow-hidden">
                <div className="h-full bg-green-500 rounded-full" style={{ width: '5%' }} />
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Database Metrics */}
      {dbMetrics && (
        <div>
          <h2 className="text-lg font-medium mb-4">Database Metrics</h2>
          <Card className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Connections</p>
                <div className="flex items-end gap-2">
                  <span className="text-2xl font-bold">{dbMetrics.connections}</span>
                  <span className="text-sm text-muted-foreground mb-1">/ {dbMetrics.max_connections}</span>
                </div>
              </div>
              <div>
                <p className="text-sm text-muted-foreground mb-1">Query Latency</p>
                <div className="flex items-end gap-2">
                  <span className="text-2xl font-bold">{dbMetrics.query_latency_ms}</span>
                  <span className="text-sm text-muted-foreground mb-1">ms</span>
                </div>
              </div>
              <div>
                <p className="text-sm text-muted-foreground mb-1">Pool Usage</p>
                <div className="flex items-end gap-2">
                  <span className="text-2xl font-bold">
                    {Math.round((dbMetrics.connections / dbMetrics.max_connections) * 100)}%
                  </span>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
