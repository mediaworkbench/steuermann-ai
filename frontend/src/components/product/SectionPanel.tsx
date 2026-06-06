import * as React from "react";
import { cn } from "@/lib/utils";

interface SectionPanelProps {
  title: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  titleClassName?: string;
}

export function SectionPanel({ title, children, className, titleClassName }: SectionPanelProps) {
  return (
    <section className={cn("rounded-xl border border-border bg-surface px-6 py-6 shadow-sm", className)}>
      <h2 className={cn("mb-4 text-xl font-bold text-foreground", titleClassName)}>{title}</h2>
      {children}
    </section>
  );
}
