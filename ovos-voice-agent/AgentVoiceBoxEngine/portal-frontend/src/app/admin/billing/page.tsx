'use client';

/**
 * Admin Billing Page - Lago Integration
 * Manages: Revenue metrics, Invoices, Subscriptions, Refunds, Credits
 */

import { useState, useEffect } from 'react';
import { 
  DollarSign, 
  Users, 
  FileText, 
  AlertTriangle,
  Download,
  RefreshCw,
  CreditCard,
  TrendingUp
} from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { MetricCard } from '@/components/ui/metric-card';
import { billingApi, LagoInvoice, LagoSubscription, LagoPlan } from '@/services/admin-api';

export default function AdminBillingPage() {
  const [metrics, setMetrics] = useState<{
    mrr: number;
    arr: number;
    total_customers: number;
    active_subscriptions: number;
    pending_invoices: number;
    failed_payments: number;
  } | null>(null);
  const [invoices, setInvoices] = useState<LagoInvoice[]>([]);
  const [subscriptions, setSubscriptions] = useState<LagoSubscription[]>([]);
  const [plans, setPlans] = useState<LagoPlan[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'invoices' | 'subscriptions' | 'plans'>('overview');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [metricsRes, invoicesRes, subsRes, plansRes] = await Promise.all([
        billingApi.getRevenueMetrics(),
        billingApi.listInvoices({ page: 1 }),
        billingApi.listSubscriptions({ page: 1 }),
        billingApi.listPlans(),
      ]);
      
      setMetrics(metricsRes.data);
      setInvoices(invoicesRes.data.invoices || []);
      setSubscriptions(subsRes.data.subscriptions || []);
      setPlans(plansRes.data.plans || []);
    } catch (error) {
      console.error('Failed to load billing data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (cents: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(cents / 100);
  };

  const handleDownloadInvoice = async (invoiceId: string) => {
    try {
      const res = await billingApi.downloadInvoice(invoiceId);
      if (res.data.file_url) {
        window.open(res.data.file_url, '_blank');
      }
    } catch (error) {
      console.error('Failed to download invoice:', error);
    }
  };

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-card rounded w-48" />
          <div className="grid grid-cols-4 gap-4">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-32 bg-card rounded-xl" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Billing Administration</h1>
          <p className="text-muted-foreground">Manage revenue, invoices, and subscriptions</p>
        </div>
        <Button onClick={loadData} variant="secondary" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Revenue Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Monthly Recurring Revenue"
          value={formatCurrency(metrics?.mrr || 0)}
          icon={<DollarSign className="w-5 h-5" />}
          accent
        />
        <MetricCard
          label="Annual Recurring Revenue"
          value={formatCurrency(metrics?.arr || 0)}
          icon={<TrendingUp className="w-5 h-5" />}
        />
        <MetricCard
          label="Active Subscriptions"
          value={metrics?.active_subscriptions || 0}
          icon={<Users className="w-5 h-5" />}
        />
        <MetricCard
          label="Pending Invoices"
          value={metrics?.pending_invoices || 0}
          icon={<FileText className="w-5 h-5" />}
          change={metrics?.failed_payments ? -metrics.failed_payments : undefined}
          changeLabel="failed"
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border">
        {(['overview', 'invoices', 'subscriptions', 'plans'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent Invoices */}
          <Card className="p-6">
            <h3 className="text-lg font-medium mb-4">Recent Invoices</h3>
            <div className="space-y-3">
              {invoices.slice(0, 5).map(invoice => (
                <div key={invoice.lago_id} className="flex items-center justify-between p-3 bg-background rounded-lg">
                  <div>
                    <p className="font-medium">{invoice.number}</p>
                    <p className="text-sm text-muted-foreground">
                      {invoice.issuing_date ? new Date(invoice.issuing_date).toLocaleDateString() : 'Draft'}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="font-medium">{formatCurrency(invoice.total_amount_cents)}</p>
                    <span className={`text-xs px-2 py-1 rounded ${
                      invoice.status === 'finalized' ? 'bg-green-500/20 text-green-400' :
                      invoice.status === 'draft' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {invoice.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Failed Payments Alert */}
          {(metrics?.failed_payments || 0) > 0 && (
            <Card className="p-6 border-red-500/50">
              <div className="flex items-start gap-4">
                <AlertTriangle className="w-6 h-6 text-red-400" />
                <div>
                  <h3 className="text-lg font-medium text-red-400">Failed Payments</h3>
                  <p className="text-muted-foreground mt-1">
                    {metrics?.failed_payments} payment(s) require attention
                  </p>
                  <Button variant="destructive" size="sm" className="mt-4">
                    View Failed Payments
                  </Button>
                </div>
              </div>
            </Card>
          )}
        </div>
      )}

      {activeTab === 'invoices' && (
        <Card className="p-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Invoice #</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Date</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Amount</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Status</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Payment</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {invoices.map(invoice => (
                  <tr key={invoice.lago_id} className="border-b border-border/50 hover:bg-card/50">
                    <td className="py-3 px-4 font-medium">{invoice.number}</td>
                    <td className="py-3 px-4 text-muted-foreground">
                      {invoice.issuing_date ? new Date(invoice.issuing_date).toLocaleDateString() : '-'}
                    </td>
                    <td className="py-3 px-4">{formatCurrency(invoice.total_amount_cents)}</td>
                    <td className="py-3 px-4">
                      <span className={`text-xs px-2 py-1 rounded ${
                        invoice.status === 'finalized' ? 'bg-green-500/20 text-green-400' :
                        invoice.status === 'draft' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {invoice.status}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`text-xs px-2 py-1 rounded ${
                        invoice.payment_status === 'succeeded' ? 'bg-green-500/20 text-green-400' :
                        invoice.payment_status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {invoice.payment_status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => handleDownloadInvoice(invoice.lago_id)}
                      >
                        <Download className="w-4 h-4" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {activeTab === 'subscriptions' && (
        <Card className="p-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">ID</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Customer</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Plan</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Status</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Started</th>
                </tr>
              </thead>
              <tbody>
                {subscriptions.map(sub => (
                  <tr key={sub.lago_id} className="border-b border-border/50 hover:bg-card/50">
                    <td className="py-3 px-4 font-mono text-sm">{sub.external_id}</td>
                    <td className="py-3 px-4">{sub.external_customer_id}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-1 bg-primary/20 text-primary rounded text-sm">
                        {sub.plan_code}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`text-xs px-2 py-1 rounded ${
                        sub.status === 'active' ? 'bg-green-500/20 text-green-400' :
                        sub.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {sub.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-muted-foreground">
                      {sub.started_at ? new Date(sub.started_at).toLocaleDateString() : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {activeTab === 'plans' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {plans.map(plan => (
            <Card key={plan.lago_id} className="p-6">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-medium">{plan.name}</h3>
                  <p className="text-sm text-muted-foreground">{plan.code}</p>
                </div>
                <CreditCard className="w-5 h-5 text-muted-foreground" />
              </div>
              <div className="mt-4">
                <p className="text-3xl font-bold">
                  {formatCurrency(plan.amount_cents)}
                  <span className="text-sm font-normal text-muted-foreground">/{plan.interval}</span>
                </p>
              </div>
              <div className="mt-4 pt-4 border-t border-border">
                <p className="text-sm text-muted-foreground">
                  {plan.active_subscriptions_count} active subscriptions
                </p>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
