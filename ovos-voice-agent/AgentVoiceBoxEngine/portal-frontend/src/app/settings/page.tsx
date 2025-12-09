"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Bell, Building, Globe, Plus, Trash2, Webhook } from "lucide-react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
import {
  settingsApi,
  OrganizationProfile,
  NotificationPreferences,
  WebhookConfig,
} from "@/lib/api";
import { formatDateTime } from "@/lib/utils";

function OrganizationSettings() {
  const queryClient = useQueryClient();
  const { data: profile, isLoading } = useQuery<OrganizationProfile>({
    queryKey: ["settings-profile"],
    queryFn: settingsApi.getProfile,
  });

  const [formData, setFormData] = useState<OrganizationProfile | null>(null);

  const updateMutation = useMutation({
    mutationFn: settingsApi.updateProfile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings-profile"] });
    },
  });

  const currentData = formData || profile;

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(5)].map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (currentData) {
          updateMutation.mutate(currentData);
        }
      }}
      className="space-y-4"
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="org-name">Organization Name</Label>
          <Input
            id="org-name"
            value={currentData?.name || ""}
            onChange={(e) =>
              setFormData({ ...currentData!, name: e.target.value })
            }
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="org-email">Contact Email</Label>
          <Input
            id="org-email"
            type="email"
            value={currentData?.email || ""}
            onChange={(e) =>
              setFormData({ ...currentData!, email: e.target.value })
            }
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="org-website">Website</Label>
          <Input
            id="org-website"
            type="url"
            placeholder="https://example.com"
            value={currentData?.website || ""}
            onChange={(e) =>
              setFormData({ ...currentData!, website: e.target.value })
            }
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="org-phone">Phone</Label>
          <Input
            id="org-phone"
            type="tel"
            value={currentData?.phone || ""}
            onChange={(e) =>
              setFormData({ ...currentData!, phone: e.target.value })
            }
          />
        </div>
        <div className="space-y-2 md:col-span-2">
          <Label htmlFor="org-address">Address</Label>
          <Input
            id="org-address"
            value={currentData?.address || ""}
            onChange={(e) =>
              setFormData({ ...currentData!, address: e.target.value })
            }
          />
        </div>
      </div>
      <Button type="submit" disabled={updateMutation.isPending}>
        {updateMutation.isPending ? "Saving..." : "Save Changes"}
      </Button>
    </form>
  );
}

function NotificationSettings() {
  const queryClient = useQueryClient();
  const { data: prefs, isLoading } = useQuery<NotificationPreferences>({
    queryKey: ["settings-notifications"],
    queryFn: settingsApi.getNotifications,
  });

  const updateMutation = useMutation({
    mutationFn: settingsApi.updateNotifications,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings-notifications"] });
    },
  });

  const togglePref = (key: keyof NotificationPreferences) => {
    if (prefs) {
      updateMutation.mutate({ ...prefs, [key]: !prefs[key] });
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(5)].map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }

  const notifications = [
    { key: "email_billing" as const, label: "Billing Notifications", description: "Invoices, payment confirmations, and billing alerts" },
    { key: "email_usage_alerts" as const, label: "Usage Alerts", description: "Notifications when approaching usage limits" },
    { key: "email_security" as const, label: "Security Alerts", description: "Login attempts, API key changes, and security events" },
    { key: "email_product_updates" as const, label: "Product Updates", description: "New features, improvements, and announcements" },
    { key: "email_weekly_summary" as const, label: "Weekly Summary", description: "Weekly usage and activity summary" },
  ];

  return (
    <div className="space-y-4">
      {notifications.map((item) => (
        <div key={item.key} className="flex items-center justify-between rounded-lg border p-4">
          <div>
            <p className="font-medium">{item.label}</p>
            <p className="text-sm text-muted-foreground">{item.description}</p>
          </div>
          <Switch
            checked={prefs?.[item.key] ?? false}
            onCheckedChange={() => togglePref(item.key)}
            disabled={updateMutation.isPending}
            aria-label={item.label}
          />
        </div>
      ))}
    </div>
  );
}

function WebhookSettings() {
  const [createOpen, setCreateOpen] = useState(false);
  const [newUrl, setNewUrl] = useState("");
  const [newEvents, setNewEvents] = useState<string[]>([]);
  const queryClient = useQueryClient();

  const { data: webhooks, isLoading } = useQuery<WebhookConfig[]>({
    queryKey: ["settings-webhooks"],
    queryFn: settingsApi.getWebhooks,
  });

  const { data: availableEvents } = useQuery<string[]>({
    queryKey: ["webhook-events"],
    queryFn: settingsApi.getWebhookEvents,
  });

  const createMutation = useMutation({
    mutationFn: settingsApi.createWebhook,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings-webhooks"] });
      setCreateOpen(false);
      setNewUrl("");
      setNewEvents([]);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: settingsApi.deleteWebhook,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings-webhooks"] });
    },
  });

  const toggleEvent = (event: string) => {
    setNewEvents((prev) =>
      prev.includes(event) ? prev.filter((e) => e !== event) : [...prev, event]
    );
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(3)].map((_, i) => (
          <Skeleton key={i} className="h-16 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" aria-hidden="true" />
              Add Webhook
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>Create Webhook</DialogTitle>
              <DialogDescription>
                Configure a webhook endpoint to receive events.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="webhook-url">Endpoint URL</Label>
                <Input
                  id="webhook-url"
                  type="url"
                  placeholder="https://your-server.com/webhook"
                  value={newUrl}
                  onChange={(e) => setNewUrl(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>Events</Label>
                <div className="max-h-48 overflow-y-auto space-y-2 border rounded-md p-2">
                  {availableEvents?.map((event) => (
                    <label key={event} className="flex items-center gap-2 text-sm cursor-pointer">
                      <input
                        type="checkbox"
                        checked={newEvents.includes(event)}
                        onChange={() => toggleEvent(event)}
                        className="h-4 w-4"
                      />
                      {event}
                    </label>
                  ))}
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={() => createMutation.mutate({ url: newUrl, events: newEvents })}
                disabled={!newUrl || newEvents.length === 0 || createMutation.isPending}
              >
                {createMutation.isPending ? "Creating..." : "Create Webhook"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {!webhooks || webhooks.length === 0 ? (
        <div className="text-center py-8 border rounded-lg">
          <Webhook className="mx-auto h-12 w-12 text-muted-foreground mb-4" aria-hidden="true" />
          <p className="text-muted-foreground">No webhooks configured</p>
          <p className="text-sm text-muted-foreground">
            Add a webhook to receive real-time event notifications
          </p>
        </div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>URL</TableHead>
              <TableHead>Events</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Last Triggered</TableHead>
              <TableHead className="w-[80px]">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {webhooks.map((webhook) => (
              <TableRow key={webhook.id}>
                <TableCell className="font-mono text-sm truncate max-w-[200px]">
                  {webhook.url}
                </TableCell>
                <TableCell>
                  <Badge variant="secondary">{webhook.events.length} events</Badge>
                </TableCell>
                <TableCell>
                  <Badge variant={webhook.is_active ? "success" : "secondary"}>
                    {webhook.is_active ? "Active" : "Inactive"}
                  </Badge>
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {webhook.last_triggered_at ? formatDateTime(webhook.last_triggered_at) : "Never"}
                </TableCell>
                <TableCell>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      if (confirm("Delete this webhook?")) {
                        deleteMutation.mutate(webhook.id);
                      }
                    }}
                    disabled={deleteMutation.isPending}
                    title="Delete webhook"
                  >
                    <Trash2 className="h-4 w-4 text-destructive" aria-hidden="true" />
                    <span className="sr-only">Delete webhook</span>
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}

export default function SettingsPage() {
  return (
    <DashboardLayout title="Settings" description="Manage your organization settings">
      <Tabs defaultValue="organization" className="space-y-6">
        <TabsList>
          <TabsTrigger value="organization">Organization</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="webhooks">Webhooks</TabsTrigger>
        </TabsList>

        <TabsContent value="organization">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Building className="h-5 w-5" aria-hidden="true" />
                Organization Profile
              </CardTitle>
              <CardDescription>
                Update your organization&apos;s information
              </CardDescription>
            </CardHeader>
            <CardContent>
              <OrganizationSettings />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5" aria-hidden="true" />
                Notification Preferences
              </CardTitle>
              <CardDescription>
                Choose which notifications you want to receive
              </CardDescription>
            </CardHeader>
            <CardContent>
              <NotificationSettings />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="webhooks">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5" aria-hidden="true" />
                Webhooks
              </CardTitle>
              <CardDescription>
                Configure webhooks to receive real-time event notifications
              </CardDescription>
            </CardHeader>
            <CardContent>
              <WebhookSettings />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </DashboardLayout>
  );
}
