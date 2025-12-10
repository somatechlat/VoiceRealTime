"use client";

/**
 * Admin Portal Sidebar Navigation
 * Implements Requirements 4.1-4.6: Admin portal navigation
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Users,
  CreditCard,
  Package,
  Activity,
  FileText,
  Settings,
  LogOut,
  Shield,
  ChevronLeft,
  ChevronRight,
  Mic,
  Phone,
  UserCog,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { ThemeToggleSimple } from "@/components/ui/theme-toggle";
import { Permission } from "@/services/auth-service";
import { useState } from "react";

interface NavItem {
  name: string;
  href: string;
  icon: React.ElementType;
  permission?: Permission;
}

const adminNavigation: NavItem[] = [
  { name: "Dashboard", href: "/admin/dashboard", icon: LayoutDashboard },
  { name: "Tenants", href: "/admin/tenants", icon: Users, permission: "tenant:view" },
  { name: "Users", href: "/admin/users", icon: UserCog, permission: "tenant:manage" },
  { name: "Voice Config", href: "/admin/voice-config", icon: Mic, permission: "system:configure" },
  { name: "Sessions", href: "/admin/sessions", icon: Phone, permission: "tenant:view" },
  { name: "Billing", href: "/admin/billing", icon: CreditCard, permission: "billing:view" },
  { name: "Plans", href: "/admin/plans", icon: Package, permission: "billing:manage" },
  { name: "Monitoring", href: "/admin/monitoring", icon: Activity, permission: "system:configure" },
  { name: "Audit Log", href: "/admin/audit", icon: FileText, permission: "tenant:view" },
  { name: "Settings", href: "/admin/settings", icon: Settings, permission: "system:configure" },
];

interface AdminSidebarProps {
  onClose?: () => void;
}

export function AdminSidebar({ onClose }: AdminSidebarProps) {
  const pathname = usePathname();
  const { user, logout, hasPermission } = useAuth();
  const [collapsed, setCollapsed] = useState(false);

  // Filter navigation items based on permissions
  const visibleNavigation = adminNavigation.filter(
    (item) => !item.permission || hasPermission(item.permission)
  );

  const handleNavClick = () => {
    onClose?.();
  };

  const handleLogout = async () => {
    await logout();
    onClose?.();
  };

  return (
    <aside 
      className={cn(
        "flex h-full flex-col border-r bg-card transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
      role="navigation" 
      aria-label="Admin navigation"
    >
      {/* Logo */}
      <div className="flex h-16 items-center justify-between border-b px-4">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <Shield className="h-8 w-8 text-primary" aria-hidden="true" />
            <span className="text-lg font-semibold">Admin</span>
          </div>
        )}
        {collapsed && (
          <Shield className="h-8 w-8 text-primary mx-auto" aria-hidden="true" />
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCollapsed(!collapsed)}
          className="hidden md:flex"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 space-y-1 px-2 py-4" aria-label="Admin">
        {visibleNavigation.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(`${item.href}/`);
          return (
            <Link
              key={item.name}
              href={item.href}
              onClick={handleNavClick}
              className={cn(
                "flex items-center gap-3 px-3 py-2 text-sm font-medium transition-colors rounded-md",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                collapsed && "justify-center px-2"
              )}
              aria-current={isActive ? "page" : undefined}
              title={collapsed ? item.name : undefined}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" aria-hidden="true" />
              {!collapsed && item.name}
            </Link>
          );
        })}
      </nav>

      {/* Theme Toggle */}
      {!collapsed && (
        <div className="border-t px-3 py-4">
          <div className="flex items-center gap-3 px-3 py-2">
            <ThemeToggleSimple />
            <span className="text-sm text-muted-foreground">Theme</span>
          </div>
        </div>
      )}

      {/* User Section */}
      <div className="border-t p-4">
        <div className={cn("flex items-center gap-3", collapsed && "justify-center")}>
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive text-destructive-foreground">
            {user?.username?.charAt(0).toUpperCase() || user?.email?.charAt(0).toUpperCase() || "A"}
          </div>
          {!collapsed && (
            <>
              <div className="flex-1 truncate">
                <p className="text-sm font-medium truncate">{user?.username || "Admin"}</p>
                <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleLogout}
                aria-label="Sign out"
                title="Sign out"
              >
                <LogOut className="h-4 w-4" aria-hidden="true" />
              </Button>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}
