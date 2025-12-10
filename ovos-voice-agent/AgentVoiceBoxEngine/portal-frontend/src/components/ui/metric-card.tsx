"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

export interface MetricCardProps extends React.HTMLAttributes<HTMLDivElement> {
  label: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  accent?: boolean;
  icon?: React.ReactNode;
  onClick?: () => void;
}

const MetricCard = React.forwardRef<HTMLDivElement, MetricCardProps>(
  ({ className, label, value, change, changeLabel, accent, icon, onClick, ...props }, ref) => {
    const getTrendIcon = () => {
      if (change === undefined || change === 0) {
        return <Minus className="h-3 w-3" />;
      }
      return change > 0 ? (
        <TrendingUp className="h-3 w-3" />
      ) : (
        <TrendingDown className="h-3 w-3" />
      );
    };

    const getTrendColor = () => {
      if (change === undefined || change === 0) {
        return "text-muted-foreground";
      }
      return change > 0 ? "text-success" : "text-destructive";
    };

    return (
      <div
        ref={ref}
        className={cn(
          "rounded-xl border bg-card p-6 transition-all duration-200",
          accent && "bg-accent/50 dark:bg-accent/20",
          onClick && "cursor-pointer hover:shadow-md hover:scale-[1.02]",
          className
        )}
        onClick={onClick}
        role={onClick ? "button" : undefined}
        tabIndex={onClick ? 0 : undefined}
        onKeyDown={onClick ? (e) => e.key === "Enter" && onClick() : undefined}
        {...props}
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            {label}
          </span>
          {icon && <span className="text-muted-foreground">{icon}</span>}
        </div>
        
        <div className="mt-2 flex items-baseline gap-2">
          <span className="text-3xl font-bold tracking-tight">{value}</span>
          
          {change !== undefined && (
            <div className={cn("flex items-center gap-1 text-xs font-medium", getTrendColor())}>
              {getTrendIcon()}
              <span>{Math.abs(change)}%</span>
            </div>
          )}
        </div>
        
        {changeLabel && (
          <p className="mt-1 text-xs text-muted-foreground">{changeLabel}</p>
        )}
      </div>
    );
  }
);
MetricCard.displayName = "MetricCard";

export { MetricCard };
