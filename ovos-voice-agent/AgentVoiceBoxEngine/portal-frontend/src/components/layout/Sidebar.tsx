"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Key,
  BarChart3,
  CreditCard,
  Users,
  Settings,
  HelpCircle,
  LogOut,
  Mic,
  Phone,
  FolderKanban,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { ThemeToggleSimple } from "@/components/ui/theme-toggle";
import { Permission } from "@/services/auth-service";

// Navigation items with permission requirements
interface NavItem {
  name: string;
  href: string;
  icon: React.ElementType;
  permission?: Permission;
  external?: boolean;
}

const navigation: NavItem[] = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "Sessions", href: "/sessions", icon: Phone, permission: "usage:view" },
  { name: "Projects", href: "/projects", icon: FolderKanban, permission: "api_keys:view" },
  { name: "API Keys", href: "/api-keys", icon: Key, permission: "api_keys:view" },
  { name: "Usage", href: "/usage", icon: BarChart3, permission: "usage:view" },
  { name: "Billing", href: "/billing", icon: CreditCard, permission: "billing:view" },
  { name: "Team", href: "/team", icon: Users, permission: "team:view" },
  { name: "Settings", href: "/settings", icon: Settings, permission: "settings:manage" },
];

const secondaryNavigation: NavItem[] = [
  { name: "Documentation", href: "https://docs.agentvoicebox.com", icon: HelpCircle, external: true },
];

interface SidebarProps {
  onClose?: () => void;
}

export function Sidebar({ onClose }: SidebarProps) {
  const pathname = usePathname();
  const { user, logout, hasPermission } = useAuth();

  // Filter navigation items based on permissions
  const visibleNavigation = navigation.filter(
    (item) => !item.permission || hasPermission(item.permission)
  );

  const handleNavClick = () => {
    // Close mobile sidebar on navigation
    onClose?.();
  };

  const handleLogout = async () => {
    await logout();
    onClose?.();
  };

  return (
    <aside 
      className="flex h-full w-64 flex-col border-r bg-card" 
      role="navigation" 
      aria-label="Main navigation"
    >
      {/* Logo */}
      <div className="flex h-16 items-center gap-2 border-b px-6">
        <Mic className="h-8 w-8 text-primary" aria-hidden="true" />
        <span className="text-lg font-semibold">AgentVoiceBox</span>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4" aria-label="Primary">
        {visibleNavigation.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(`${item.href}/`);
          return (
            <Link
              key={item.name}
              href={item.href}
              onClick={handleNavClick}
              className={cn(
                "flex items-center gap-3 px-3 py-2 text-sm font-medium transition-colors",
                // Verve-style pill navigation for light theme, rounded for dark
                "rounded-full",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              )}
              aria-current={isActive ? "page" : undefined}
            >
              <item.icon className="h-5 w-5" aria-hidden="true" />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Secondary Navigation */}
      <nav className="border-t px-3 py-4" aria-label="Secondary">
        {secondaryNavigation.map((item) => (
          <a
            key={item.name}
            href={item.href}
            target={item.external ? "_blank" : undefined}
            rel={item.external ? "noopener noreferrer" : undefined}
            className="flex items-center gap-3 rounded-full px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            <item.icon className="h-5 w-5" aria-hidden="true" />
            {item.name}
            {item.external && <span className="sr-only">(opens in new tab)</span>}
          </a>
        ))}
        
        {/* Theme Toggle */}
        <div className="mt-2 flex items-center gap-3 px-3 py-2">
          <ThemeToggleSimple />
          <span className="text-sm text-muted-foreground">Theme</span>
        </div>
      </nav>

      {/* User Section */}
      <div className="border-t p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-foreground">
            {user?.username?.charAt(0).toUpperCase() || user?.email?.charAt(0).toUpperCase() || "U"}
          </div>
          <div className="flex-1 truncate">
            <p className="text-sm font-medium truncate">{user?.username || "User"}</p>
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
        </div>
      </div>
    </aside>
  );
}
