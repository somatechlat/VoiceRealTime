"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, CreditCard, Download, FileText } from "lucide-react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { billingApi, PlanDetails, CurrentSubscription, Invoice } from "@/lib/api";
import { formatCurrency, formatDate, getStatusColor } from "@/lib/utils";

function PlanCard({
  plan,
  isCurrentPlan,
  onSelect,
  isLoading,
}: {
  plan: PlanDetails;
  isCurrentPlan: boolean;
  onSelect: () => void;
  isLoading: boolean;
}) {
  const isEnterprise = plan.code === "enterprise";

  return (
    <Card className={isCurrentPlan ? "border-primary" : ""}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{plan.name}</CardTitle>
          {isCurrentPlan && <Badge>Current Plan</Badge>}
        </div>
        <CardDescription>{plan.description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <span className="text-3xl font-bold">
            {plan.amount_cents === 0 ? "Free" : formatCurrency(plan.amount_cents)}
          </span>
          {plan.amount_cents > 0 && (
            <span className="text-muted-foreground">/{plan.interval}</span>
          )}
        </div>
        <ul className="space-y-2">
          {plan.features.map((feature, i) => (
            <li key={i} className="flex items-center gap-2 text-sm">
              <Check className="h-4 w-4 text-green-500" aria-hidden="true" />
              {feature}
            </li>
          ))}
        </ul>
      </CardContent>
      <CardFooter>
        {isEnterprise ? (
          <Button variant="outline" className="w-full" asChild>
            <a href="mailto:sales@agentvoicebox.com">Contact Sales</a>
          </Button>
        ) : isCurrentPlan ? (
          <Button variant="outline" className="w-full" disabled>
            Current Plan
          </Button>
        ) : (
          <Button className="w-full" onClick={onSelect} disabled={isLoading}>
            {isLoading ? "Processing..." : plan.amount_cents === 0 ? "Downgrade" : "Upgrade"}
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}

function InvoicesTable({ invoices }: { invoices: Invoice[] }) {
  if (invoices.length === 0) {
    return (
      <div className="text-center py-8">
        <FileText className="mx-auto h-12 w-12 text-muted-foreground mb-4" aria-hidden="true" />
        <p className="text-muted-foreground">No invoices yet</p>
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Invoice</TableHead>
          <TableHead>Date</TableHead>
          <TableHead>Amount</TableHead>
          <TableHead>Status</TableHead>
          <TableHead className="w-[100px]">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {invoices.map((invoice) => (
          <TableRow key={invoice.id}>
            <TableCell className="font-medium">{invoice.number}</TableCell>
            <TableCell>{invoice.issuing_date ? formatDate(invoice.issuing_date) : "-"}</TableCell>
            <TableCell>{formatCurrency(invoice.total_amount_cents, invoice.currency)}</TableCell>
            <TableCell>
              <Badge className={getStatusColor(invoice.payment_status)}>
                {invoice.payment_status}
              </Badge>
            </TableCell>
            <TableCell>
              {invoice.pdf_url && (
                <Button variant="ghost" size="icon" asChild>
                  <a href={invoice.pdf_url} target="_blank" rel="noopener noreferrer" title="Download PDF">
                    <Download className="h-4 w-4" aria-hidden="true" />
                    <span className="sr-only">Download invoice</span>
                  </a>
                </Button>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function BillingPage() {
  const [changePlanDialog, setChangePlanDialog] = useState<PlanDetails | null>(null);
  const queryClient = useQueryClient();

  const { data: plans, isLoading: plansLoading } = useQuery<PlanDetails[]>({
    queryKey: ["billing-plans"],
    queryFn: billingApi.getPlans,
  });

  const { data: subscription, isLoading: subLoading } = useQuery<CurrentSubscription>({
    queryKey: ["subscription"],
    queryFn: billingApi.getSubscription,
  });

  const { data: invoices, isLoading: invoicesLoading } = useQuery<Invoice[]>({
    queryKey: ["invoices"],
    queryFn: () => billingApi.getInvoices(),
  });

  const changePlanMutation = useMutation({
    mutationFn: (planCode: string) => billingApi.changePlan(planCode),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["subscription"] });
      setChangePlanDialog(null);
    },
  });

  const cancelMutation = useMutation({
    mutationFn: billingApi.cancelSubscription,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["subscription"] });
    },
  });

  const isLoading = plansLoading || subLoading;

  return (
    <DashboardLayout title="Billing" description="Manage your subscription and invoices">
      <Tabs defaultValue="plans" className="space-y-6">
        <TabsList>
          <TabsTrigger value="plans">Plans</TabsTrigger>
          <TabsTrigger value="invoices">Invoices</TabsTrigger>
        </TabsList>

        <TabsContent value="plans" className="space-y-6">
          {/* Current Subscription */}
          {subscription && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CreditCard className="h-5 w-5" aria-hidden="true" />
                  Current Subscription
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-3">
                  <div>
                    <p className="text-sm text-muted-foreground">Plan</p>
                    <p className="font-medium">{subscription.plan.name}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Status</p>
                    <Badge className={getStatusColor(subscription.status)}>
                      {subscription.status}
                    </Badge>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Next Billing</p>
                    <p className="font-medium">
                      {subscription.current_period_end
                        ? formatDate(subscription.current_period_end)
                        : "N/A"}
                    </p>
                  </div>
                </div>
                {subscription.cancel_at_period_end && (
                  <p className="mt-4 text-sm text-yellow-600">
                    Your subscription will be canceled at the end of the current period.
                  </p>
                )}
              </CardContent>
              {subscription.plan.code !== "free" && !subscription.cancel_at_period_end && (
                <CardFooter>
                  <Button
                    variant="outline"
                    onClick={() => {
                      if (confirm("Are you sure you want to cancel your subscription?")) {
                        cancelMutation.mutate();
                      }
                    }}
                    disabled={cancelMutation.isPending}
                  >
                    {cancelMutation.isPending ? "Canceling..." : "Cancel Subscription"}
                  </Button>
                </CardFooter>
              )}
            </Card>
          )}

          {/* Plans Grid */}
          {isLoading ? (
            <div className="grid gap-6 md:grid-cols-3">
              {[...Array(3)].map((_, i) => (
                <Card key={i}>
                  <CardHeader>
                    <Skeleton className="h-6 w-24" />
                    <Skeleton className="h-4 w-48" />
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Skeleton className="h-10 w-32" />
                    <div className="space-y-2">
                      {[...Array(4)].map((_, j) => (
                        <Skeleton key={j} className="h-4 w-full" />
                      ))}
                    </div>
                  </CardContent>
                  <CardFooter>
                    <Skeleton className="h-10 w-full" />
                  </CardFooter>
                </Card>
              ))}
            </div>
          ) : (
            <div className="grid gap-6 md:grid-cols-3">
              {plans?.map((plan) => (
                <PlanCard
                  key={plan.code}
                  plan={plan}
                  isCurrentPlan={subscription?.plan.code === plan.code}
                  onSelect={() => setChangePlanDialog(plan)}
                  isLoading={changePlanMutation.isPending}
                />
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="invoices">
          <Card>
            <CardHeader>
              <CardTitle>Invoice History</CardTitle>
              <CardDescription>View and download your past invoices</CardDescription>
            </CardHeader>
            <CardContent>
              {invoicesLoading ? (
                <div className="space-y-4">
                  {[...Array(5)].map((_, i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : (
                <InvoicesTable invoices={invoices || []} />
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Change Plan Confirmation Dialog */}
      <Dialog open={!!changePlanDialog} onOpenChange={() => setChangePlanDialog(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Change Plan</DialogTitle>
            <DialogDescription>
              {changePlanDialog && (
                <>
                  You are about to change to the <strong>{changePlanDialog.name}</strong> plan.
                  {changePlanDialog.amount_cents > 0 && (
                    <> This will cost {formatCurrency(changePlanDialog.amount_cents)}/{changePlanDialog.interval}.</>
                  )}
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setChangePlanDialog(null)}>
              Cancel
            </Button>
            <Button
              onClick={() => changePlanDialog && changePlanMutation.mutate(changePlanDialog.code)}
              disabled={changePlanMutation.isPending}
            >
              {changePlanMutation.isPending ? "Processing..." : "Confirm Change"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </DashboardLayout>
  );
}
