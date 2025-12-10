"use client";

/**
 * Mobile Navigation Component
 * Bottom navigation bar for mobile devices
 * Implements Requirements 18.1, 18.3: Responsive navigation with touch-friendly interactions
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Key,
  CreditCard,
  Settings,
  MoreHorizontal,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { Permission } from "@/services/auth-service";

interface NavItem {
  name: string;
  href: string;
  icon: React.ElementType;
  permission?: Permission;
}

// Primary mobile nav items (max 5 for bottom bar)
const mobileNavItems: NavItem[] = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "API Keys", href: "/api-keys", icon: Key, permission: "api_keys:view" },
  { name: "Billing", href: "/billing", icon: CreditCard, permission: "billing:view" },
  { name: "Settings", href: "/settings", icon: Settings, permission: "settings:manage" },
  { name: "More", href: "#more", icon: MoreHorizontal },
];

interface MobileNavProps {
  onMoreClick?: () => void;
}

export function MobileNav({ onMoreClick }: MobileNavProps) {
  const pathname = usePathname();
  const { hasPermission } = useAuth();

  // Filter items based on permissions
  const visibleItems = mobileNavItems.filter(
    (item) => !item.permission || hasPermission(item.permission)
  );

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-40 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 md:hidden"
      role="navigation"
      aria-label="Mobile navigation"
    >
      <div className="flex h-16 items-center justify-around px-2">
        {visibleItems.map((item) => {
          const isActive = item.href !== "#more" && 
            (pathname === item.href || pathname.startsWith(`${item.href}/`));
          const isMore = item.href === "#more";

          if (isMore) {
            return (
              <button
                key={item.name}
                onClick={onMoreClick}
                className={cn(
                  "flex flex-col items-center justify-center gap-1 px-3 py-2 min-w-[64px]",
                  "text-muted-foreground transition-colors",
                  "active:scale-95 touch-manipulation"
                )}
                aria-label="More options"
              >
                <item.icon className="h-5 w-5" aria-hidden="true" />
                <span className="text-xs font-medium">{item.name}</span>
              </button>
            );
          }

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex flex-col items-center justify-center gap-1 px-3 py-2 min-w-[64px]",
                "transition-colors touch-manipulation",
                isActive
                  ? "text-primary"
                  : "text-muted-foreground active:text-foreground"
              )}
              aria-current={isActive ? "page" : undefined}
            >
              <item.icon 
                className={cn(
                  "h-5 w-5 transition-transform",
                  isActive && "scale-110"
                )} 
                aria-hidden="true" 
              />
              <span className={cn(
                "text-xs font-medium",
                isActive && "font-semibold"
              )}>
                {item.name}
              </span>
              {isActive && (
                <span className="absolute -top-0.5 h-0.5 w-8 rounded-full bg-primary" />
              )}
            </Link>
          );
        })}
      </div>
      {/* Safe area padding for devices with home indicator */}
      <div className="h-safe-area-inset-bottom bg-background" />
    </nav>
  );
}
