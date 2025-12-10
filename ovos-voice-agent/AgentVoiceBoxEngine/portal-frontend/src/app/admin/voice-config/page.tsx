'use client';

/**
 * Voice Configuration Page - Admin Portal
 * Configure STT, TTS, LLM workers and personas
 */

import { useState, useEffect } from 'react';
import { 
  Mic, 
  Volume2, 
  Brain,
  Settings,
  Save,
  RefreshCw,
  User,
  Plus,
  Trash2
} from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  workersApi, 
  personasApi, 
  STTConfig, 
  TTSConfig, 
  LLMConfig,
  Persona 
} from '@/services/voice-api';

export default function VoiceConfigPage() {
  const [activeTab, setActiveTab] = useState<'stt' | 'tts' | 'llm' | 'personas'>('stt');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  
  // Configs
  const [sttConfig, setSttConfig] = useState<STTConfig>({
    model: 'tiny',
    device: 'cpu',
    compute_type: 'int8',
    batch_size: 2,
  });
  const [ttsConfig, setTtsConfig] = useState<TTSConfig>({
    model_dir: '/models/kokoro',
    default_voice: 'am_onyx',
    default_speed: 1.1,
    available_voices: [],
  });
  const [llmConfig, setLlmConfig] = useState<LLMConfig>({
    default_provider: 'groq',
    model: 'llama-3.1-70b-versatile',
    max_tokens: 4096,
    temperature: 0.7,
    circuit_breaker_threshold: 5,
    circuit_breaker_timeout: 30,
  });
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [voices, setVoices] = useState<string[]>([]);

  useEffect(() => {
    loadConfigs();
  }, []);

  const loadConfigs = async () => {
    setLoading(true);
    try {
      const [sttRes, ttsRes, llmRes, personasRes, voicesRes] = await Promise.all([
        workersApi.getSTT().catch(() => ({ data: { config: sttConfig } })),
        workersApi.getTTS().catch(() => ({ data: { config: ttsConfig } })),
        workersApi.getLLM().catch(() => ({ data: { config: llmConfig } })),
        personasApi.list().catch(() => ({ data: [] })),
        workersApi.getVoices().catch(() => ({ data: [] })),
      ]);
      
      if (sttRes.data?.config) setSttConfig(sttRes.data.config);
      if (ttsRes.data?.config) setTtsConfig(ttsRes.data.config);
      if (llmRes.data?.config) setLlmConfig(llmRes.data.config);
      setPersonas(personasRes.data || []);
      setVoices(voicesRes.data || ['am_onyx', 'af_bella', 'am_adam', 'af_sarah']);
    } catch (error) {
      console.error('Failed to load configs:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSTT = async () => {
    setSaving(true);
    try {
      await workersApi.updateSTT(sttConfig);
    } catch (error) {
      console.error('Failed to save STT config:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleSaveTTS = async () => {
    setSaving(true);
    try {
      await workersApi.updateTTS(ttsConfig);
    } catch (error) {
      console.error('Failed to save TTS config:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleSaveLLM = async () => {
    setSaving(true);
    try {
      await workersApi.updateLLM(llmConfig);
    } catch (error) {
      console.error('Failed to save LLM config:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleDeletePersona = async (name: string) => {
    try {
      await personasApi.delete(name);
      setPersonas(personas.filter(p => p.name !== name));
    } catch (error) {
      console.error('Failed to delete persona:', error);
    }
  };

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-card rounded w-48" />
          <div className="h-64 bg-card rounded-xl" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Voice Configuration</h1>
          <p className="text-muted-foreground">Configure speech and language processing</p>
        </div>
        <Button onClick={loadConfigs} variant="secondary" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border">
        {[
          { id: 'stt', label: 'Speech-to-Text', icon: Mic },
          { id: 'tts', label: 'Text-to-Speech', icon: Volume2 },
          { id: 'llm', label: 'Language Model', icon: Brain },
          { id: 'personas', label: 'Personas', icon: User },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* STT Config */}
      {activeTab === 'stt' && (
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
              <Mic className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h3 className="font-medium">Speech-to-Text Configuration</h3>
              <p className="text-sm text-muted-foreground">Whisper model settings</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="text-sm font-medium mb-2 block">Model Size</label>
              <select
                value={sttConfig.model}
                onChange={(e) => setSttConfig({ ...sttConfig, model: e.target.value })}
                className="w-full h-10 px-3 rounded-lg bg-background border border-border"
              >
                <option value="tiny">Tiny (39M params)</option>
                <option value="base">Base (74M params)</option>
                <option value="small">Small (244M params)</option>
                <option value="medium">Medium (769M params)</option>
                <option value="large">Large (1.5B params)</option>
              </select>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Device</label>
              <select
                value={sttConfig.device}
                onChange={(e) => setSttConfig({ ...sttConfig, device: e.target.value as 'cpu' | 'cuda' })}
                className="w-full h-10 px-3 rounded-lg bg-background border border-border"
              >
                <option value="cpu">CPU</option>
                <option value="cuda">CUDA (GPU)</option>
              </select>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Compute Type</label>
              <select
                value={sttConfig.compute_type}
                onChange={(e) => setSttConfig({ ...sttConfig, compute_type: e.target.value })}
                className="w-full h-10 px-3 rounded-lg bg-background border border-border"
              >
                <option value="int8">INT8 (Fastest)</option>
                <option value="float16">Float16</option>
                <option value="float32">Float32 (Most Accurate)</option>
              </select>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Batch Size</label>
              <Input
                type="number"
                value={sttConfig.batch_size}
                onChange={(e) => setSttConfig({ ...sttConfig, batch_size: parseInt(e.target.value) })}
                min={1}
                max={16}
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Language (Optional)</label>
              <Input
                placeholder="Auto-detect"
                value={sttConfig.language || ''}
                onChange={(e) => setSttConfig({ ...sttConfig, language: e.target.value || undefined })}
              />
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-border flex justify-end">
            <Button onClick={handleSaveSTT} disabled={saving}>
              <Save className="w-4 h-4 mr-2" />
              {saving ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </Card>
      )}

      {/* TTS Config */}
      {activeTab === 'tts' && (
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
              <Volume2 className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <h3 className="font-medium">Text-to-Speech Configuration</h3>
              <p className="text-sm text-muted-foreground">Kokoro TTS settings</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="text-sm font-medium mb-2 block">Default Voice</label>
              <select
                value={ttsConfig.default_voice}
                onChange={(e) => setTtsConfig({ ...ttsConfig, default_voice: e.target.value })}
                className="w-full h-10 px-3 rounded-lg bg-background border border-border"
              >
                {voices.map(voice => (
                  <option key={voice} value={voice}>{voice}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Default Speed</label>
              <Input
                type="number"
                step="0.1"
                min="0.5"
                max="2.0"
                value={ttsConfig.default_speed}
                onChange={(e) => setTtsConfig({ ...ttsConfig, default_speed: parseFloat(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground mt-1">Range: 0.5 - 2.0</p>
            </div>

            <div className="md:col-span-2">
              <label className="text-sm font-medium mb-2 block">Model Directory</label>
              <Input
                value={ttsConfig.model_dir}
                onChange={(e) => setTtsConfig({ ...ttsConfig, model_dir: e.target.value })}
              />
            </div>
          </div>

          <div className="mt-6">
            <label className="text-sm font-medium mb-2 block">Available Voices</label>
            <div className="flex flex-wrap gap-2">
              {voices.map(voice => (
                <span
                  key={voice}
                  className={`px-3 py-1 rounded-full text-sm ${
                    voice === ttsConfig.default_voice
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {voice}
                </span>
              ))}
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-border flex justify-end">
            <Button onClick={handleSaveTTS} disabled={saving}>
              <Save className="w-4 h-4 mr-2" />
              {saving ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </Card>
      )}

      {/* LLM Config */}
      {activeTab === 'llm' && (
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
              <Brain className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="font-medium">Language Model Configuration</h3>
              <p className="text-sm text-muted-foreground">LLM provider and model settings</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="text-sm font-medium mb-2 block">Provider</label>
              <select
                value={llmConfig.default_provider}
                onChange={(e) => setLlmConfig({ ...llmConfig, default_provider: e.target.value as LLMConfig['default_provider'] })}
                className="w-full h-10 px-3 rounded-lg bg-background border border-border"
              >
                <option value="groq">Groq</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="local">Local</option>
              </select>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Model</label>
              <Input
                value={llmConfig.model}
                onChange={(e) => setLlmConfig({ ...llmConfig, model: e.target.value })}
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Max Tokens</label>
              <Input
                type="number"
                value={llmConfig.max_tokens}
                onChange={(e) => setLlmConfig({ ...llmConfig, max_tokens: parseInt(e.target.value) })}
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Temperature</label>
              <Input
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={llmConfig.temperature}
                onChange={(e) => setLlmConfig({ ...llmConfig, temperature: parseFloat(e.target.value) })}
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Circuit Breaker Threshold</label>
              <Input
                type="number"
                value={llmConfig.circuit_breaker_threshold}
                onChange={(e) => setLlmConfig({ ...llmConfig, circuit_breaker_threshold: parseInt(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground mt-1">Failures before circuit opens</p>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Circuit Breaker Timeout</label>
              <Input
                type="number"
                value={llmConfig.circuit_breaker_timeout}
                onChange={(e) => setLlmConfig({ ...llmConfig, circuit_breaker_timeout: parseInt(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground mt-1">Seconds before retry</p>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-border flex justify-end">
            <Button onClick={handleSaveLLM} disabled={saving}>
              <Save className="w-4 h-4 mr-2" />
              {saving ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </Card>
      )}

      {/* Personas */}
      {activeTab === 'personas' && (
        <div className="space-y-4">
          <div className="flex justify-end">
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              Create Persona
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {personas.length === 0 ? (
              <Card className="p-8 text-center col-span-full">
                <User className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No personas configured</p>
              </Card>
            ) : (
              personas.map(persona => (
                <Card key={persona.name} className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                        <User className="w-5 h-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">{persona.name}</p>
                        <p className="text-sm text-muted-foreground">{persona.voice}</p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeletePersona(persona.name)}
                    >
                      <Trash2 className="w-4 h-4 text-red-400" />
                    </Button>
                  </div>
                  <div className="mt-4 space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Language</span>
                      <span>{persona.language}</span>
                    </div>
                    {persona.personality && (
                      <div>
                        <span className="text-muted-foreground">Personality</span>
                        <p className="text-xs mt-1">{persona.personality}</p>
                      </div>
                    )}
                  </div>
                </Card>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
