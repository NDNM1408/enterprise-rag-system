"use client";

import { useSidebarStore } from "@/lib/stores";
import { cn } from "@/lib/utils";

interface DashboardContentProps {
  children: React.ReactNode;
}

export function DashboardContent({ children }: DashboardContentProps) {
  const { isCollapsed } = useSidebarStore();

  return (
    <main
      className={cn(
        "pt-14 min-h-screen transition-all duration-300",
        isCollapsed ? "ml-16" : "ml-60"
      )}
    >
      <div className="container mx-auto p-6">{children}</div>
    </main>
  );
}
