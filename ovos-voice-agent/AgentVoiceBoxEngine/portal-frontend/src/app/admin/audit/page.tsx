'use client';

/**
 * Admin Audit Logs Page
 * Tracks all administrative actions for compliance
 */

import { useState, useEffect } from 'react';
import { 
  FileText, 
  Search, 
  Download, 
  Filter,
  User,
  Calendar,
  RefreshCw
} from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { auditApi, AuditLogEntry } from '@/services/admin-api';

export default function AdminAuditPage() {
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState({
    actor_id: '',
    action: '',
    resource_type: '',
    start_date: '',
    end_date: '',
  });

  useEffect(() => {
    loadLogs();
  }, [page, filters]);

  const loadLogs = async () => {
    setLoading(true);
    try {
      const res = await auditApi.list({
        ...filters,
        page,
        per_page: 50,
      });
      setLogs(res.data.logs || []);
      setTotal(res.data.total || 0);
    } catch (error) {
      console.error('Failed to load audit logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      const blob = await auditApi.export({
        format,
        start_date: filters.start_date,
        end_date: filters.end_date,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `audit-logs-${new Date().toISOString().split('T')[0]}.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export logs:', error);
    }
  };

  const getActionColor = (action: string) => {
    if (action.includes('created')) return 'bg-green-500/20 text-green-400';
    if (action.includes('deleted') || action.includes('revoked')) return 'bg-red-500/20 text-red-400';
    if (action.includes('updated') || action.includes('changed')) return 'bg-blue-500/20 text-blue-400';
    if (action.includes('login') || action.includes('logout')) return 'bg-purple-500/20 text-purple-400';
    return 'bg-gray-500/20 text-gray-400';
  };

  const formatAction = (action: string) => {
    return action.replace(/[._]/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Audit Logs</h1>
          <p className="text-muted-foreground">Track all administrative actions</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => handleExport('csv')} variant="secondary" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Export CSV
          </Button>
          <Button onClick={() => handleExport('json')} variant="secondary" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Export JSON
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <label className="text-sm text-muted-foreground mb-1 block">Actor</label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Filter by actor ID..."
                value={filters.actor_id}
                onChange={(e) => setFilters({ ...filters, actor_id: e.target.value })}
                className="pl-10"
              />
            </div>
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="text-sm text-muted-foreground mb-1 block">Action</label>
            <select
              value={filters.action}
              onChange={(e) => setFilters({ ...filters, action: e.target.value })}
              className="w-full h-10 px-3 rounded-lg bg-background border border-border text-sm"
            >
              <option value="">All Actions</option>
              <option value="tenant.created">Tenant Created</option>
              <option value="tenant.updated">Tenant Updated</option>
              <option value="tenant.suspended">Tenant Suspended</option>
              <option value="tenant.deleted">Tenant Deleted</option>
              <option value="api_key.created">API Key Created</option>
              <option value="api_key.revoked">API Key Revoked</option>
              <option value="user.login">User Login</option>
              <option value="user.logout">User Logout</option>
              <option value="settings.changed">Settings Changed</option>
            </select>
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="text-sm text-muted-foreground mb-1 block">Resource Type</label>
            <select
              value={filters.resource_type}
              onChange={(e) => setFilters({ ...filters, resource_type: e.target.value })}
              className="w-full h-10 px-3 rounded-lg bg-background border border-border text-sm"
            >
              <option value="">All Resources</option>
              <option value="tenant">Tenant</option>
              <option value="user">User</option>
              <option value="api_key">API Key</option>
              <option value="billing">Billing</option>
              <option value="config">Config</option>
            </select>
          </div>
          <div className="flex-1 min-w-[150px]">
            <label className="text-sm text-muted-foreground mb-1 block">Start Date</label>
            <Input
              type="date"
              value={filters.start_date}
              onChange={(e) => setFilters({ ...filters, start_date: e.target.value })}
            />
          </div>
          <div className="flex-1 min-w-[150px]">
            <label className="text-sm text-muted-foreground mb-1 block">End Date</label>
            <Input
              type="date"
              value={filters.end_date}
              onChange={(e) => setFilters({ ...filters, end_date: e.target.value })}
            />
          </div>
          <div className="flex items-end">
            <Button onClick={loadLogs} variant="secondary" size="sm">
              <RefreshCw className="w-4 h-4 mr-2" />
              Apply
            </Button>
          </div>
        </div>
      </Card>

      {/* Logs Table */}
      <Card className="p-6">
        {loading ? (
          <div className="space-y-4">
            {[1, 2, 3, 4, 5].map(i => (
              <div key={i} className="h-16 bg-background rounded animate-pulse" />
            ))}
          </div>
        ) : logs.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No audit logs found</p>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Timestamp</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Actor</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Action</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Resource</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">IP Address</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Details</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map(log => (
                    <tr key={log.id} className="border-b border-border/50 hover:bg-card/50">
                      <td className="py-3 px-4 text-sm">
                        <div>
                          <p>{new Date(log.created_at).toLocaleDateString()}</p>
                          <p className="text-muted-foreground text-xs">
                            {new Date(log.created_at).toLocaleTimeString()}
                          </p>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <div>
                          <p className="font-mono text-sm">{log.actor_id || 'System'}</p>
                          <p className="text-xs text-muted-foreground">{log.actor_type}</p>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`text-xs px-2 py-1 rounded ${getActionColor(log.action)}`}>
                          {formatAction(log.action)}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <div>
                          <p className="text-sm">{log.resource_type}</p>
                          {log.resource_id && (
                            <p className="text-xs text-muted-foreground font-mono">{log.resource_id}</p>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-4 text-sm text-muted-foreground font-mono">
                        {log.ip_address || '-'}
                      </td>
                      <td className="py-3 px-4">
                        {Object.keys(log.details || {}).length > 0 && (
                          <Button variant="ghost" size="sm">
                            View
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
              <p className="text-sm text-muted-foreground">
                Showing {(page - 1) * 50 + 1} - {Math.min(page * 50, total)} of {total} logs
              </p>
              <div className="flex gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={page === 1}
                  onClick={() => setPage(p => p - 1)}
                >
                  Previous
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={page * 50 >= total}
                  onClick={() => setPage(p => p + 1)}
                >
                  Next
                </Button>
              </div>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}
