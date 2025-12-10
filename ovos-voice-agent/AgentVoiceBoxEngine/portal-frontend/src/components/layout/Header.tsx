"use client";

import { Bell, Search, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/contexts/AuthContext";
import { Badge } from "@/components/ui/badge";

interface HeaderProps {
  title: string;
  description?: string;
  onMenuClick?: () => void;
  isMobileMenuOpen?: boolean;
}

export function Header({ title, description, onMenuClick, isMobileMenuOpen }: HeaderProps) {
  const { user } = useAuth();

  // Format current date in Verve style
  const today = new Date();
  const dateString = today.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });

  return (
    <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center gap-4 px-4 md:px-6">
        {/* Mobile menu button */}
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={onMenuClick}
          aria-label={isMobileMenuOpen ? "Close menu" : "Open menu"}
          aria-expanded={isMobileMenuOpen}
        >
          {isMobileMenuOpen ? (
            <X className="h-5 w-5" aria-hidden="true" />
          ) : (
            <Menu className="h-5 w-5" aria-hidden="true" />
          )}
        </Button>

        {/* Page title and date */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold truncate">{title}</h1>
            {/* Verve-style date display */}
            <span className="hidden text-sm text-muted-foreground lg:block">
              {dateString}
            </span>
          </div>
          {description && (
            <p className="text-sm text-muted-foreground truncate">{description}</p>
          )}
        </div>

        {/* Search - hidden on mobile */}
        <div className="hidden w-64 lg:block">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" aria-hidden="true" />
            <Input
              type="search"
              placeholder="Search..."
              className="pl-9 h-9 rounded-full bg-muted/50 border-0 focus-visible:ring-1"
              aria-label="Search"
            />
          </div>
        </div>

        {/* Notifications */}
        <Button 
          variant="ghost" 
          size="icon" 
          className="relative"
          aria-label="View notifications"
        >
          <Bell className="h-5 w-5" aria-hidden="true" />
          {/* Notification badge - show when there are unread notifications */}
          <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-primary" />
          <span className="sr-only">Notifications (1 unread)</span>
        </Button>

        {/* User info - hidden on mobile */}
        <div className="hidden items-center gap-2 md:flex">
          {user?.tenantId && (
            <Badge variant="secondary" className="rounded-full px-3">
              {user.tenantId.slice(0, 8)}...
            </Badge>
          )}
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-medium">
            {user?.username?.charAt(0).toUpperCase() || user?.email?.charAt(0).toUpperCase() || "U"}
          </div>
        </div>
      </div>
    </header>
  );
}
