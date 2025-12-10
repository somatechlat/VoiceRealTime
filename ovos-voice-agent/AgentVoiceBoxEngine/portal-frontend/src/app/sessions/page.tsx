'use client';

/**
 * Voice Sessions Page - Customer Portal
 * View and manage voice sessions, conversation history
 */

import { useState, useEffect } from 'react';
import { 
  Phone, 
  PhoneOff, 
  Clock, 
  MessageSquare,
  Play,
  RefreshCw,
  ChevronRight,
  Mic,
  Volume2
} from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { MetricCard } from '@/components/ui/metric-card';
import { sessionsApi, VoiceSession, ConversationItem } from '@/services/voice-api';

export default function SessionsPage() {
  const [sessions, setSessions] = useState<VoiceSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<VoiceSession | null>(null);
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'active' | 'closed'>('all');

  useEffect(() => {
    loadSessions();
  }, [filter]);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const res = await sessionsApi.list({
        status: filter === 'all' ? undefined : filter,
        limit: 50,
      });
      setSessions(res.data.sessions || []);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadConversation = async (sessionId: string) => {
    try {
      const res = await sessionsApi.getConversation(sessionId);
      setConversation(res.data || []);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleSelectSession = async (session: VoiceSession) => {
    setSelectedSession(session);
    await loadConversation(session.id);
  };

  const handleCloseSession = async (sessionId: string) => {
    try {
      await sessionsApi.close(sessionId);
      loadSessions();
      if (selectedSession?.id === sessionId) {
        setSelectedSession(null);
        setConversation([]);
      }
    } catch (error) {
      console.error('Failed to close session:', error);
    }
  };

  const activeSessions = sessions.filter(s => s.status === 'active').length;
  const totalDuration = sessions.reduce((acc, s) => {
    if (s.closed_at && s.created_at) {
      return acc + (new Date(s.closed_at).getTime() - new Date(s.created_at).getTime());
    }
    return acc;
  }, 0);

  const formatDuration = (ms: number) => {
    const minutes = Math.floor(ms / 60000);
    const hours = Math.floor(minutes / 60);
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    return `${minutes}m`;
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Voice Sessions</h1>
          <p className="text-muted-foreground">Monitor and manage your voice conversations</p>
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
          value={activeSessions}
          icon={<Phone className="w-5 h-5" />}
          accent={activeSessions > 0}
        />
        <MetricCard
          label="Total Sessions"
          value={sessions.length}
          icon={<MessageSquare className="w-5 h-5" />}
        />
        <MetricCard
          label="Total Duration"
          value={formatDuration(totalDuration)}
          icon={<Clock className="w-5 h-5" />}
        />
        <MetricCard
          label="Avg Duration"
          value={sessions.length > 0 ? formatDuration(totalDuration / sessions.length) : '0m'}
          icon={<Clock className="w-5 h-5" />}
        />
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 border-b border-border">
        {(['all', 'active', 'closed'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setFilter(tab)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              filter === tab
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Sessions List & Detail */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sessions List */}
        <div className="lg:col-span-1 space-y-2">
          {loading ? (
            <div className="space-y-2">
              {[1, 2, 3].map(i => (
                <div key={i} className="h-20 bg-card rounded-lg animate-pulse" />
              ))}
            </div>
          ) : sessions.length === 0 ? (
            <Card className="p-8 text-center">
              <Phone className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No sessions found</p>
            </Card>
          ) : (
            sessions.map(session => (
              <Card
                key={session.id}
                className={`p-4 cursor-pointer transition-colors ${
                  selectedSession?.id === session.id ? 'border-primary' : 'hover:bg-card/80'
                }`}
                onClick={() => handleSelectSession(session)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {session.status === 'active' ? (
                      <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center">
                        <Phone className="w-5 h-5 text-green-400" />
                      </div>
                    ) : (
                      <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center">
                        <PhoneOff className="w-5 h-5 text-muted-foreground" />
                      </div>
                    )}
                    <div>
                      <p className="font-medium font-mono text-sm">{session.id.slice(0, 12)}...</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(session.created_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-muted-foreground" />
                </div>
                {session.persona && (
                  <div className="mt-2 flex items-center gap-2">
                    <span className="text-xs px-2 py-1 bg-primary/20 text-primary rounded">
                      {session.persona.name}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {session.persona.voice}
                    </span>
                  </div>
                )}
              </Card>
            ))
          )}
        </div>

        {/* Session Detail */}
        <div className="lg:col-span-2">
          {selectedSession ? (
            <Card className="p-6">
              {/* Session Header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-lg font-medium">Session Details</h3>
                  <p className="text-sm text-muted-foreground font-mono">{selectedSession.id}</p>
                </div>
                {selectedSession.status === 'active' && (
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() => handleCloseSession(selectedSession.id)}
                  >
                    <PhoneOff className="w-4 h-4 mr-2" />
                    End Session
                  </Button>
                )}
              </div>

              {/* Session Info */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <p className="text-sm text-muted-foreground">Status</p>
                  <span className={`text-sm px-2 py-1 rounded ${
                    selectedSession.status === 'active'
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-muted text-muted-foreground'
                  }`}>
                    {selectedSession.status}
                  </span>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Model</p>
                  <p className="font-medium">{selectedSession.model || 'Default'}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Created</p>
                  <p className="font-medium">{new Date(selectedSession.created_at).toLocaleString()}</p>
                </div>
                {selectedSession.closed_at && (
                  <div>
                    <p className="text-sm text-muted-foreground">Closed</p>
                    <p className="font-medium">{new Date(selectedSession.closed_at).toLocaleString()}</p>
                  </div>
                )}
              </div>

              {/* Persona Info */}
              {selectedSession.persona && (
                <div className="mb-6 p-4 bg-background rounded-lg">
                  <h4 className="text-sm font-medium mb-2">Persona</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-muted-foreground">Name:</span> {selectedSession.persona.name}
                    </div>
                    <div>
                      <span className="text-muted-foreground">Voice:</span> {selectedSession.persona.voice}
                    </div>
                    <div>
                      <span className="text-muted-foreground">Language:</span> {selectedSession.persona.language}
                    </div>
                  </div>
                </div>
              )}

              {/* Conversation */}
              <div>
                <h4 className="text-sm font-medium mb-4">Conversation</h4>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {conversation.length === 0 ? (
                    <p className="text-center text-muted-foreground py-8">No messages yet</p>
                  ) : (
                    conversation.map(item => (
                      <div
                        key={item.id}
                        className={`flex gap-3 ${
                          item.role === 'user' ? 'flex-row-reverse' : ''
                        }`}
                      >
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          item.role === 'user'
                            ? 'bg-primary/20'
                            : item.role === 'assistant'
                            ? 'bg-green-500/20'
                            : 'bg-muted'
                        }`}>
                          {item.role === 'user' ? (
                            <Mic className="w-4 h-4 text-primary" />
                          ) : (
                            <Volume2 className="w-4 h-4 text-green-400" />
                          )}
                        </div>
                        <div className={`flex-1 p-3 rounded-lg ${
                          item.role === 'user'
                            ? 'bg-primary/10 text-right'
                            : 'bg-card'
                        }`}>
                          <p className="text-sm">
                            {typeof item.content === 'string'
                              ? item.content
                              : JSON.stringify(item.content)}
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {new Date(item.created_at).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </Card>
          ) : (
            <Card className="p-12 text-center">
              <MessageSquare className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">Select a session to view details</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
