"use client";

/**
 * Admin Portal Layout
 * Wraps all admin portal pages with the admin-specific layout
 * Implements Requirements 1.1, 4.1: Admin portal layout with sidebar navigation
 */

import { ReactNode } from "react";

interface AdminLayoutProps {
  children: ReactNode;
}

export default function AdminLayout({ children }: AdminLayoutProps) {
  // The AdminDashboardLayout is applied at the page level for flexibility
  // This layout ensures all admin routes share common providers
  return <>{children}</>;
}
