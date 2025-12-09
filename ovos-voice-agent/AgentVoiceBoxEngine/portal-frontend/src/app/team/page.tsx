"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Mail, MoreVertical, Shield, Trash2, UserPlus, Users } from "lucide-react";
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
import { teamApi, TeamMember, TeamRole } from "@/lib/api";
import { formatDateTime, getStatusColor, getInitials } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";

function InviteDialog() {
  const [open, setOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [roles, setRoles] = useState<string[]>(["viewer"]);
  const [message, setMessage] = useState("");
  const queryClient = useQueryClient();

  const { data: availableRoles } = useQuery<TeamRole[]>({
    queryKey: ["team-roles"],
    queryFn: teamApi.getRoles,
  });

  const inviteMutation = useMutation({
    mutationFn: teamApi.invite,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["team-members"] });
      setOpen(false);
      setEmail("");
      setRoles(["viewer"]);
      setMessage("");
    },
  });

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <UserPlus className="mr-2 h-4 w-4" aria-hidden="true" />
          Invite Member
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Invite Team Member</DialogTitle>
          <DialogDescription>
            Send an invitation to join your organization.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="invite-email">Email Address</Label>
            <Input
              id="invite-email"
              type="email"
              placeholder="colleague@company.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label>Role</Label>
            <Select value={roles[0]} onValueChange={(value) => setRoles([value])}>
              <SelectTrigger>
                <SelectValue placeholder="Select a role" />
              </SelectTrigger>
              <SelectContent>
                {availableRoles?.map((role) => (
                  <SelectItem key={role.name} value={role.name}>
                    <div>
                      <p className="font-medium">{role.name}</p>
                      <p className="text-xs text-muted-foreground">{role.description}</p>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="invite-message">Message (optional)</Label>
            <Input
              id="invite-message"
              placeholder="Welcome to the team!"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => inviteMutation.mutate({ email, roles, message: message || undefined })}
            disabled={!email || roles.length === 0 || inviteMutation.isPending}
          >
            {inviteMutation.isPending ? "Sending..." : "Send Invitation"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function MemberRow({
  member,
  currentUserId,
  availableRoles,
}: {
  member: TeamMember;
  currentUserId: string;
  availableRoles: TeamRole[];
}) {
  const queryClient = useQueryClient();
  const isCurrentUser = member.id === currentUserId;

  const updateRolesMutation = useMutation({
    mutationFn: ({ memberId, roles }: { memberId: string; roles: string[] }) =>
      teamApi.updateRoles(memberId, roles),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["team-members"] });
    },
  });

  const removeMutation = useMutation({
    mutationFn: teamApi.remove,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["team-members"] });
    },
  });

  return (
    <TableRow>
      <TableCell>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-foreground">
            {getInitials(member.name)}
          </div>
          <div>
            <p className="font-medium">
              {member.name}
              {isCurrentUser && <span className="text-muted-foreground ml-2">(you)</span>}
            </p>
            <p className="text-sm text-muted-foreground">{member.email}</p>
          </div>
        </div>
      </TableCell>
      <TableCell>
        <div className="flex flex-wrap gap-1">
          {member.roles.map((role) => (
            <Badge key={role} variant="secondary">
              {role}
            </Badge>
          ))}
        </div>
      </TableCell>
      <TableCell>
        <Badge className={getStatusColor(member.status)}>{member.status}</Badge>
      </TableCell>
      <TableCell className="text-sm text-muted-foreground">
        {member.joined_at ? formatDateTime(member.joined_at) : "-"}
      </TableCell>
      <TableCell>
        {!isCurrentUser && (
          <div className="flex items-center gap-2">
            <Select
              value={member.roles[0]}
              onValueChange={(value) =>
                updateRolesMutation.mutate({ memberId: member.id, roles: [value] })
              }
              disabled={updateRolesMutation.isPending}
            >
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {availableRoles.map((role) => (
                  <SelectItem key={role.name} value={role.name}>
                    {role.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                if (confirm(`Remove ${member.name} from the team?`)) {
                  removeMutation.mutate(member.id);
                }
              }}
              disabled={removeMutation.isPending}
              title="Remove member"
            >
              <Trash2 className="h-4 w-4 text-destructive" aria-hidden="true" />
              <span className="sr-only">Remove member</span>
            </Button>
          </div>
        )}
      </TableCell>
    </TableRow>
  );
}

export default function TeamPage() {
  const { user } = useAuth();

  const { data: members, isLoading: membersLoading } = useQuery<TeamMember[]>({
    queryKey: ["team-members"],
    queryFn: () => teamApi.getMembers(),
  });

  const { data: roles, isLoading: rolesLoading } = useQuery<TeamRole[]>({
    queryKey: ["team-roles"],
    queryFn: teamApi.getRoles,
  });

  const isLoading = membersLoading || rolesLoading;

  return (
    <DashboardLayout title="Team" description="Manage your team members">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-muted-foreground">
              Invite team members and manage their access to your organization.
            </p>
          </div>
          <InviteDialog />
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" aria-hidden="true" />
              Team Members
            </CardTitle>
            <CardDescription>
              {members?.length || 0} member{members?.length !== 1 ? "s" : ""} in your organization
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                {[...Array(3)].map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : !members || members.length === 0 ? (
              <div className="text-center py-8">
                <Users className="mx-auto h-12 w-12 text-muted-foreground mb-4" aria-hidden="true" />
                <p className="text-muted-foreground">No team members yet</p>
                <p className="text-sm text-muted-foreground">
                  Invite your first team member to get started
                </p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Member</TableHead>
                    <TableHead>Roles</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Joined</TableHead>
                    <TableHead className="w-[180px]">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {members.map((member) => (
                    <MemberRow
                      key={member.id}
                      member={member}
                      currentUserId={user?.id || ""}
                      availableRoles={roles || []}
                    />
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        {/* Roles Reference */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" aria-hidden="true" />
              Available Roles
            </CardTitle>
            <CardDescription>
              Roles determine what actions team members can perform
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              {roles?.map((role) => (
                <div key={role.name} className="rounded-lg border p-4">
                  <h4 className="font-medium">{role.name}</h4>
                  <p className="text-sm text-muted-foreground">{role.description}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
