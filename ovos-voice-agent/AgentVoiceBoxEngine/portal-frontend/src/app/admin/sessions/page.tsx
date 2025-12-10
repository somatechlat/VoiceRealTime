'use client';

/**
 * Admin Sessions Page
 * View all voice sessions across all tenants
 */

import { useState, useEffect } from 'react';
import { 
  Phone, 
  PhoneOff, 
  Search,
  RefreshCw,
  Filter,
  Users
} from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { MetricCard } from '@/components/ui/metric-card';
import { sessionsApi, VoiceSession } from '@/services/voice-api';

export default function AdminSessionsPage() {
  const [sessions, setSessions] = useState<VoiceSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'active' | 'closed'>('all');
  const [search, setSearch] = useState('');

  useEffect(() => {
    loadSessions();
    const interval = setInterval(loadSessions, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [filter]);

  const loadSessions = async () => {
    try {
      const res = await sessionsApi.list({
        status: filter === 'all' ? undefined : filter,
        limit: 100,
      });
      setSessions(res.data.sessions || []);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCloseSession = async (sessionId: string) => {
    try {
      await sessionsApi.close(sessionId);
      loadSessions();
    } catch (error) {
      console.error('Failed to close session:', error);
    }
  };

  const activeSessions = sessions.filter(s => s.status === 'active');
  const filteredSessions = sessions.filter(s => 
    !search || 
    s.id.toLowerCase().includes(search.toLowerCase()) ||
    s.tenant_id?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Voice Sessions</h1>
          <p className="text-muted-foreground">Monitor all active and recent sessions</p>
        </div>
        <Button onClick={loadSessions} variant="secondary" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          label="Active Sessions"
          value={activeSessions.length}
          icon={<Phone className="w-5 h-5" />}
          accent={activeSessions.length > 0}
        />
        <MetricCard
          label="Total Sessions"
          value={sessions.length}
          icon={<Users className="w-5 h-5" />}
        />
        <MetricCard
          label="Unique Tenants"
          value={new Set(sessions.map(s => s.tenant_id)).size}
          icon={<Users className="w-5 h-5" />}
        />
        <MetricCard
          label="Closed Today"
          value={sessions.filter(s => 
            s.status === 'closed' && 
            s.closed_at && 
            new Date(s.closed_at).toDateString() === new Date().toDateString()
          ).length}
          icon={<PhoneOff className="w-5 h-5" />}
        />
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px] relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search by session ID or tenant..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-10"
            />
          </div>
          <div className="flex bg-background rounded-lg p-1">
            {(['all', 'active', 'closed'] as const).map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  filter === f
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* Sessions Table */}
      <Card className="p-6">
        {loading ? (
          <div className="space-y-4">
            {[1, 2, 3, 4, 5].map(i => (
              <div key={i} className="h-16 bg-background rounded animate-pulse" />
            ))}
          </div>
        ) : filteredSessions.length === 0 ? (
          <div className="text-center py-12">
            <Phone className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No sessions found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Session ID</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Tenant</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Status</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Model</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Persona</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Created</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredSessions.map(session => (
                  <tr key={session.id} className="border-b border-border/50 hover:bg-card/50">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        {session.status === 'active' ? (
                          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                        ) : (
                          <div className="w-2 h-2 rounded-full bg-muted" />
                        )}
                        <span className="font-mono text-sm">{session.id.slice(0, 16)}...</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="font-mono text-sm text-muted-foreground">
                        {session.tenant_id?.slice(0, 8) || '-'}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`text-xs px-2 py-1 rounded ${
                        session.status === 'active'
                          ? 'bg-green-500/20 text-green-400'
                          : session.status === 'error'
                          ? 'bg-red-500/20 text-red-400'
                          : 'bg-muted text-muted-foreground'
                      }`}>
                        {session.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm">{session.model || 'Default'}</td>
                    <td className="py-3 px-4">
                      {session.persona ? (
                        <span className="text-xs px-2 py-1 bg-primary/20 text-primary rounded">
                          {session.persona.name}
                        </span>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-sm text-muted-foreground">
                      {new Date(session.created_at).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {session.status === 'active' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleCloseSession(session.id)}
                        >
                          <PhoneOff className="w-4 h-4 text-red-400" />
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
