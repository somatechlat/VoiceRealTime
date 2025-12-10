"use client";

/**
 * Admin Tenant Management Page
 * Implements Requirements 13.1-13.8: Tenant list, search, actions, impersonation
 */

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Search,
  Filter,
  MoreVertical,
  Eye,
  Ban,
  CheckCircle,
  Trash2,
  UserCog,
  Building,
} from "lucide-react";
import { AdminLayout } from "@/components/layout/AdminLayout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { apiRequest } from "@/lib/api";
import { formatCurrency, formatDateTime, getStatusColor } from "@/lib/utils";

// Types
interface Tenant {
  id: string;
  name: string;
  email: string;
  plan: string;
  status: "active" | "suspended" | "trial";
  mrr_cents: number;
  created_at: string;
  last_activity_at: string;
}

interface TenantListResponse {
  tenants: Tenant[];
  total: number;
  page: number;
  per_page: number;
}

// API functions
const tenantsApi = {
  list: (params: { search?: string; status?: string; plan?: string; page?: number }) =>
    apiRequest<TenantListResponse>("/api/v1/admin/tenants", {
      params: params as Record<string, string | number | boolean | undefined>,
    }),
  suspend: (tenantId: string, reason: string) =>
    apiRequest<{ message: string }>(`/api/v1/admin/tenants/${tenantId}/suspend`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    }),
  unsuspend: (tenantId: string) =>
    apiRequest<{ message: string }>(`/api/v1/admin/tenants/${tenantId}/unsuspend`, {
      method: "POST",
    }),
  impersonate: (tenantId: string, userId: string, reason: string) =>
    apiRequest<{ token: string; expires_at: string }>(`/api/v1/admin/tenants/${tenantId}/impersonate`, {
      method: "POST",
      body: JSON.stringify({ user_id: userId, reason }),
    }),
};

function TenantRow({
  tenant,
  onAction,
}: {
  tenant: Tenant;
  onAction: (action: string, tenant: Tenant) => void;
}) {
  return (
    <TableRow>
      <TableCell>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
            <Building className="h-5 w-5 text-muted-foreground" />
          </div>
          <div>
            <p className="font-medium">{tenant.name}</p>
            <p className="text-sm text-muted-foreground">{tenant.email}</p>
          </div>
        </div>
      </TableCell>
      <TableCell>
        <Badge variant="secondary">{tenant.plan}</Badge>
      </TableCell>
      <TableCell>
        <Badge className={getStatusColor(tenant.status)}>{tenant.status}</Badge>
      </TableCell>
      <TableCell className="text-right font-medium">
        {formatCurrency(tenant.mrr_cents)}
      </TableCell>
      <TableCell className="text-sm text-muted-foreground">
        {formatDateTime(tenant.created_at)}
      </TableCell>
      <TableCell className="text-sm text-muted-foreground">
        {formatDateTime(tenant.last_activity_at)}
      </TableCell>
      <TableCell>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon">
              <MoreVertical className="h-4 w-4" />
              <span className="sr-only">Actions</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onAction("view", tenant)}>
              <Eye className="mr-2 h-4 w-4" />
              View Details
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onAction("impersonate", tenant)}>
              <UserCog className="mr-2 h-4 w-4" />
              Impersonate User
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            {tenant.status === "suspended" ? (
              <DropdownMenuItem onClick={() => onAction("unsuspend", tenant)}>
                <CheckCircle className="mr-2 h-4 w-4" />
                Unsuspend
              </DropdownMenuItem>
            ) : (
              <DropdownMenuItem onClick={() => onAction("suspend", tenant)}>
                <Ban className="mr-2 h-4 w-4" />
                Suspend
              </DropdownMenuItem>
            )}
            <DropdownMenuItem
              onClick={() => onAction("delete", tenant)}
              className="text-destructive"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </TableCell>
    </TableRow>
  );
}

export default function TenantsPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [planFilter, setPlanFilter] = useState<string>("all");
  const [page, setPage] = useState(1);
  const [actionDialog, setActionDialog] = useState<{
    type: string;
    tenant: Tenant;
  } | null>(null);
  const [actionReason, setActionReason] = useState("");
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery<TenantListResponse>({
    queryKey: ["admin-tenants", search, statusFilter, planFilter, page],
    queryFn: () =>
      tenantsApi.list({
        search: search || undefined,
        status: statusFilter !== "all" ? statusFilter : undefined,
        plan: planFilter !== "all" ? planFilter : undefined,
        page,
      }),
  });

  const suspendMutation = useMutation({
    mutationFn: ({ tenantId, reason }: { tenantId: string; reason: string }) =>
      tenantsApi.suspend(tenantId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-tenants"] });
      setActionDialog(null);
      setActionReason("");
    },
  });

  const unsuspendMutation = useMutation({
    mutationFn: (tenantId: string) => tenantsApi.unsuspend(tenantId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-tenants"] });
      setActionDialog(null);
    },
  });

  const handleAction = (action: string, tenant: Tenant) => {
    if (action === "view") {
      window.location.href = `/admin/tenants/${tenant.id}`;
    } else {
      setActionDialog({ type: action, tenant });
    }
  };

  const executeAction = () => {
    if (!actionDialog) return;

    const { type, tenant } = actionDialog;

    switch (type) {
      case "suspend":
        suspendMutation.mutate({ tenantId: tenant.id, reason: actionReason });
        break;
      case "unsuspend":
        unsuspendMutation.mutate(tenant.id);
        break;
      case "impersonate":
        // Would open impersonation flow
        alert("Impersonation would start here with audit logging");
        setActionDialog(null);
        break;
      case "delete":
        if (confirm(`Are you sure you want to delete ${tenant.name}? This cannot be undone.`)) {
          // Would call delete API
          alert("Delete would happen here");
        }
        setActionDialog(null);
        break;
    }
  };

  return (
    <AdminLayout title="Tenants" description="Manage platform tenants">
      <div className="space-y-6">
        {/* Search and Filters */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col gap-4 md:flex-row md:items-center">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by name, email, or tenant ID..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-9"
                />
              </div>
              <div className="flex gap-2">
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-32">
                    <Filter className="mr-2 h-4 w-4" />
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="suspended">Suspended</SelectItem>
                    <SelectItem value="trial">Trial</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={planFilter} onValueChange={setPlanFilter}>
                  <SelectTrigger className="w-32">
                    <SelectValue placeholder="Plan" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Plans</SelectItem>
                    <SelectItem value="free">Free</SelectItem>
                    <SelectItem value="pro">Pro</SelectItem>
                    <SelectItem value="enterprise">Enterprise</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Tenants Table */}
        <Card>
          <CardHeader>
            <CardTitle>All Tenants</CardTitle>
            <CardDescription>
              {data?.total || 0} total tenants
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                {[...Array(5)].map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : !data?.tenants || data.tenants.length === 0 ? (
              <div className="text-center py-8">
                <Building className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground">No tenants found</p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Tenant</TableHead>
                    <TableHead>Plan</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">MRR</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Last Activity</TableHead>
                    <TableHead className="w-[60px]">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.tenants.map((tenant) => (
                    <TenantRow
                      key={tenant.id}
                      tenant={tenant}
                      onAction={handleAction}
                    />
                  ))}
                </TableBody>
              </Table>
            )}

            {/* Pagination */}
            {data && data.total > data.per_page && (
              <div className="flex items-center justify-between mt-4">
                <p className="text-sm text-muted-foreground">
                  Showing {(page - 1) * data.per_page + 1} to{" "}
                  {Math.min(page * data.per_page, data.total)} of {data.total}
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((p) => p + 1)}
                    disabled={page * data.per_page >= data.total}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Action Dialog */}
        <Dialog open={!!actionDialog} onOpenChange={() => setActionDialog(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {actionDialog?.type === "suspend" && "Suspend Tenant"}
                {actionDialog?.type === "unsuspend" && "Unsuspend Tenant"}
                {actionDialog?.type === "impersonate" && "Impersonate User"}
                {actionDialog?.type === "delete" && "Delete Tenant"}
              </DialogTitle>
              <DialogDescription>
                {actionDialog?.type === "suspend" && (
                  <>
                    Suspending <strong>{actionDialog.tenant.name}</strong> will
                    immediately block all API access and close active sessions.
                  </>
                )}
                {actionDialog?.type === "unsuspend" && (
                  <>
                    Unsuspending <strong>{actionDialog?.tenant.name}</strong> will
                    restore full access to the platform.
                  </>
                )}
                {actionDialog?.type === "impersonate" && (
                  <>
                    You are about to impersonate a user in{" "}
                    <strong>{actionDialog?.tenant.name}</strong>. This action will
                    be logged for audit purposes.
                  </>
                )}
              </DialogDescription>
            </DialogHeader>

            {(actionDialog?.type === "suspend" || actionDialog?.type === "impersonate") && (
              <div className="py-4">
                <Input
                  placeholder="Reason for this action..."
                  value={actionReason}
                  onChange={(e) => setActionReason(e.target.value)}
                />
              </div>
            )}

            <DialogFooter>
              <Button variant="outline" onClick={() => setActionDialog(null)}>
                Cancel
              </Button>
              <Button
                variant={actionDialog?.type === "delete" ? "destructive" : "default"}
                onClick={executeAction}
                disabled={
                  (actionDialog?.type === "suspend" && !actionReason) ||
                  (actionDialog?.type === "impersonate" && !actionReason) ||
                  suspendMutation.isPending ||
                  unsuspendMutation.isPending
                }
              >
                {suspendMutation.isPending || unsuspendMutation.isPending
                  ? "Processing..."
                  : "Confirm"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </AdminLayout>
  );
}
