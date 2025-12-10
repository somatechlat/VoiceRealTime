'use client';

/**
 * Projects Page - Customer Portal
 * Manage projects (production, staging, development environments)
 */

import { useState, useEffect } from 'react';
import { 
  FolderKanban, 
  Plus,
  Settings,
  Key,
  BarChart3,
  MoreVertical,
  Rocket,
  FlaskConical,
  Code
} from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { apiClient } from '@/services/api-client';

type ProjectEnvironment = 'production' | 'staging' | 'development';

interface Project {
  id: string;
  name: string;
  environment: ProjectEnvironment;
  settings: Record<string, unknown>;
  created_at: string;
  api_keys_count?: number;
  sessions_count?: number;
}

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newProject, setNewProject] = useState<{ name: string; environment: ProjectEnvironment }>({ name: '', environment: 'development' });

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get<{ projects: Project[] }>('/v1/projects');
      setProjects(res.data.projects || []);
    } catch (error) {
      console.error('Failed to load projects:', error);
      // Mock data for demo
      setProjects([
        {
          id: 'proj_prod_001',
          name: 'Production',
          environment: 'production',
          settings: {},
          created_at: new Date().toISOString(),
          api_keys_count: 3,
          sessions_count: 1250,
        },
        {
          id: 'proj_stag_001',
          name: 'Staging',
          environment: 'staging',
          settings: {},
          created_at: new Date().toISOString(),
          api_keys_count: 2,
          sessions_count: 45,
        },
        {
          id: 'proj_dev_001',
          name: 'Development',
          environment: 'development',
          settings: {},
          created_at: new Date().toISOString(),
          api_keys_count: 5,
          sessions_count: 320,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateProject = async () => {
    try {
      await apiClient.post('/v1/projects', newProject);
      setShowCreate(false);
      setNewProject({ name: '', environment: 'development' });
      loadProjects();
    } catch (error) {
      console.error('Failed to create project:', error);
    }
  };

  const getEnvironmentIcon = (env: string) => {
    switch (env) {
      case 'production':
        return <Rocket className="w-5 h-5 text-green-400" />;
      case 'staging':
        return <FlaskConical className="w-5 h-5 text-yellow-400" />;
      default:
        return <Code className="w-5 h-5 text-blue-400" />;
    }
  };

  const getEnvironmentColor = (env: string) => {
    switch (env) {
      case 'production':
        return 'bg-green-500/20 text-green-400';
      case 'staging':
        return 'bg-yellow-500/20 text-yellow-400';
      default:
        return 'bg-blue-500/20 text-blue-400';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Projects</h1>
          <p className="text-muted-foreground">Manage your environments and configurations</p>
        </div>
        <Button onClick={() => setShowCreate(true)}>
          <Plus className="w-4 h-4 mr-2" />
          New Project
        </Button>
      </div>

      {/* Create Project Modal */}
      {showCreate && (
        <Card className="p-6">
          <h3 className="text-lg font-medium mb-4">Create New Project</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Project Name</label>
              <Input
                placeholder="My Project"
                value={newProject.name}
                onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Environment</label>
              <select
                value={newProject.environment}
                onChange={(e) => setNewProject({ ...newProject, environment: e.target.value as ProjectEnvironment })}
                className="w-full h-10 px-3 rounded-lg bg-background border border-border"
              >
                <option value="development">Development</option>
                <option value="staging">Staging</option>
                <option value="production">Production</option>
              </select>
            </div>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="secondary" onClick={() => setShowCreate(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateProject} disabled={!newProject.name}>
              Create Project
            </Button>
          </div>
        </Card>
      )}

      {/* Projects Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-48 bg-card rounded-xl animate-pulse" />
          ))}
        </div>
      ) : projects.length === 0 ? (
        <Card className="p-12 text-center">
          <FolderKanban className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-2">No projects yet</h3>
          <p className="text-muted-foreground mb-4">Create your first project to get started</p>
          <Button onClick={() => setShowCreate(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Create Project
          </Button>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {projects.map(project => (
            <Card key={project.id} className="p-6 hover:border-primary/50 transition-colors">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    project.environment === 'production' ? 'bg-green-500/20' :
                    project.environment === 'staging' ? 'bg-yellow-500/20' :
                    'bg-blue-500/20'
                  }`}>
                    {getEnvironmentIcon(project.environment)}
                  </div>
                  <div>
                    <h3 className="font-medium">{project.name}</h3>
                    <span className={`text-xs px-2 py-0.5 rounded ${getEnvironmentColor(project.environment)}`}>
                      {project.environment}
                    </span>
                  </div>
                </div>
                <Button variant="ghost" size="sm">
                  <MoreVertical className="w-4 h-4" />
                </Button>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Key className="w-4 h-4" />
                    <span>API Keys</span>
                  </div>
                  <span className="font-medium">{project.api_keys_count || 0}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <BarChart3 className="w-4 h-4" />
                    <span>Sessions</span>
                  </div>
                  <span className="font-medium">{project.sessions_count?.toLocaleString() || 0}</span>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-border flex gap-2">
                <Button variant="secondary" size="sm" className="flex-1">
                  <Key className="w-4 h-4 mr-2" />
                  Keys
                </Button>
                <Button variant="secondary" size="sm" className="flex-1">
                  <Settings className="w-4 h-4 mr-2" />
                  Settings
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
