"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Copy, Key, MoreVertical, Plus, RefreshCw, Trash2 } from "lucide-react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { apiKeysApi, ApiKey, ApiKeyCreated } from "@/lib/api";
import { formatDateTime, truncateKey } from "@/lib/utils";

const AVAILABLE_SCOPES = [
  { id: "realtime:connect", label: "Realtime Connect", description: "Connect to WebSocket API" },
  { id: "realtime:admin", label: "Realtime Admin", description: "Manage sessions" },
  { id: "billing:read", label: "Billing Read", description: "View billing information" },
];

function CreateKeyDialog({ onCreated }: { onCreated: (key: ApiKeyCreated) => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<string[]>(["realtime:connect"]);
  const [expiresInDays, setExpiresInDays] = useState<number | undefined>(undefined);
  const queryClient = useQueryClient();

  const createMutation = useMutation({
    mutationFn: apiKeysApi.create,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
      onCreated(data);
      setOpen(false);
      setName("");
      setScopes(["realtime:connect"]);
      setExpiresInDays(undefined);
    },
  });

  const toggleScope = (scope: string) => {
    setScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope]
    );
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="mr-2 h-4 w-4" aria-hidden="true" />
          Create API Key
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create API Key</DialogTitle>
          <DialogDescription>
            Create a new API key for your application. The key will only be shown once.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="key-name">Name</Label>
            <Input
              id="key-name"
              placeholder="My Application"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label>Permissions</Label>
            <div className="space-y-2">
              {AVAILABLE_SCOPES.map((scope) => (
                <label
                  key={scope.id}
                  className="flex items-center gap-3 rounded-md border p-3 cursor-pointer hover:bg-muted"
                >
                  <input
                    type="checkbox"
                    checked={scopes.includes(scope.id)}
                    onChange={() => toggleScope(scope.id)}
                    className="h-4 w-4"
                  />
                  <div>
                    <p className="font-medium text-sm">{scope.label}</p>
                    <p className="text-xs text-muted-foreground">{scope.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="expires">Expiration (optional)</Label>
            <Input
              id="expires"
              type="number"
              placeholder="Days until expiration"
              min={1}
              max={365}
              value={expiresInDays || ""}
              onChange={(e) => setExpiresInDays(e.target.value ? parseInt(e.target.value) : undefined)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => createMutation.mutate({ name, scopes, expires_in_days: expiresInDays })}
            disabled={!name || scopes.length === 0 || createMutation.isPending}
          >
            {createMutation.isPending ? "Creating..." : "Create Key"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function KeyCreatedDialog({
  apiKey,
  onClose,
}: {
  apiKey: ApiKeyCreated | null;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    if (apiKey?.secret) {
      await navigator.clipboard.writeText(apiKey.secret);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <Dialog open={!!apiKey} onOpenChange={() => onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>API Key Created</DialogTitle>
          <DialogDescription>
            Copy your API key now. You won&apos;t be able to see it again.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="rounded-md bg-muted p-4">
            <code className="text-sm break-all">{apiKey?.secret}</code>
          </div>
          <Button onClick={copyToClipboard} className="w-full">
            <Copy className="mr-2 h-4 w-4" aria-hidden="true" />
            {copied ? "Copied!" : "Copy to Clipboard"}
          </Button>
        </div>
        <DialogFooter>
          <Button onClick={onClose}>Done</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function ApiKeysPage() {
  const [createdKey, setCreatedKey] = useState<ApiKeyCreated | null>(null);
  const queryClient = useQueryClient();

  const { data: keys, isLoading } = useQuery<ApiKey[]>({
    queryKey: ["api-keys"],
    queryFn: () => apiKeysApi.list(),
  });

  const revokeMutation = useMutation({
    mutationFn: apiKeysApi.revoke,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
    },
  });

  const rotateMutation = useMutation({
    mutationFn: apiKeysApi.rotate,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
      setCreatedKey(data.new_key);
    },
  });

  return (
    <DashboardLayout title="API Keys" description="Manage your API keys">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-muted-foreground">
              API keys are used to authenticate requests to the AgentVoiceBox API.
            </p>
          </div>
          <CreateKeyDialog onCreated={setCreatedKey} />
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Your API Keys</CardTitle>
            <CardDescription>
              Keys are shown with their prefix only. Full keys are never displayed after creation.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                {[...Array(3)].map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : !keys || keys.length === 0 ? (
              <div className="text-center py-8">
                <Key className="mx-auto h-12 w-12 text-muted-foreground mb-4" aria-hidden="true" />
                <p className="text-muted-foreground">No API keys yet</p>
                <p className="text-sm text-muted-foreground">
                  Create your first API key to get started
                </p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Key</TableHead>
                    <TableHead>Scopes</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Last Used</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="w-[100px]">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {keys.map((key) => (
                    <TableRow key={key.id}>
                      <TableCell className="font-medium">{key.name}</TableCell>
                      <TableCell>
                        <code className="text-sm bg-muted px-2 py-1 rounded">
                          {key.prefix}...
                        </code>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {key.scopes.map((scope) => (
                            <Badge key={scope} variant="secondary" className="text-xs">
                              {scope}
                            </Badge>
                          ))}
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {formatDateTime(key.created_at)}
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {key.last_used_at ? formatDateTime(key.last_used_at) : "Never"}
                      </TableCell>
                      <TableCell>
                        <Badge variant={key.is_active ? "success" : "destructive"}>
                          {key.is_active ? "Active" : "Revoked"}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => rotateMutation.mutate(key.id)}
                            disabled={!key.is_active || rotateMutation.isPending}
                            title="Rotate key"
                          >
                            <RefreshCw className="h-4 w-4" aria-hidden="true" />
                            <span className="sr-only">Rotate key</span>
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => {
                              if (confirm("Are you sure you want to revoke this key?")) {
                                revokeMutation.mutate(key.id);
                              }
                            }}
                            disabled={!key.is_active || revokeMutation.isPending}
                            title="Revoke key"
                          >
                            <Trash2 className="h-4 w-4 text-destructive" aria-hidden="true" />
                            <span className="sr-only">Revoke key</span>
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        <KeyCreatedDialog apiKey={createdKey} onClose={() => setCreatedKey(null)} />
      </div>
    </DashboardLayout>
  );
}
