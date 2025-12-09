"use client";

import { Bell, Search, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/contexts/AuthContext";

interface HeaderProps {
  title: string;
  description?: string;
  onMenuClick?: () => void;
}

export function Header({ title, description, onMenuClick }: HeaderProps) {
  const { user } = useAuth();

  return (
    <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center gap-4 px-6">
        {/* Mobile menu button */}
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={onMenuClick}
          aria-label="Open menu"
        >
          <Menu className="h-5 w-5" aria-hidden="true" />
        </Button>

        {/* Page title */}
        <div className="flex-1">
          <h1 className="text-xl font-semibold">{title}</h1>
          {description && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
        </div>

        {/* Search */}
        <div className="hidden w-64 md:block">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" aria-hidden="true" />
            <Input
              type="search"
              placeholder="Search..."
              className="pl-8"
              aria-label="Search"
            />
          </div>
        </div>

        {/* Notifications */}
        <Button variant="ghost" size="icon" aria-label="View notifications">
          <Bell className="h-5 w-5" aria-hidden="true" />
          <span className="sr-only">Notifications</span>
        </Button>

        {/* Tenant badge */}
        {user?.tenant_name && (
          <div className="hidden rounded-md bg-muted px-3 py-1 text-sm font-medium md:block">
            {user.tenant_name}
          </div>
        )}
      </div>
    </header>
  );
}
