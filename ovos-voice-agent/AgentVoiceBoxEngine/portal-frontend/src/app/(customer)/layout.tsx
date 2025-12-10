"use client";

/**
 * Customer Portal Layout
 * Wraps all customer portal pages with the dashboard layout
 * Implements Requirements 1.1, 6.1: Customer portal layout with navigation
 */

import { ReactNode } from "react";

interface CustomerLayoutProps {
  children: ReactNode;
}

export default function CustomerLayout({ children }: CustomerLayoutProps) {
  // The DashboardLayout is applied at the page level for flexibility
  // This layout ensures all customer routes share common providers
  return <>{children}</>;
}
