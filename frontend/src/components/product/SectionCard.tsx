import * as React from "react";
import { cn } from "@/lib/utils";

type SectionCardTone = "default" | "danger";

interface SectionCardProps {
  children: React.ReactNode;
  className?: string;
  tone?: SectionCardTone;
}

const toneClasses: Record<SectionCardTone, string> = {
  default: "border-border",
  danger: "border-destructive/30",
};

export function SectionCard({ children, className, tone = "default" }: SectionCardProps) {
  return (
    <section
      className={cn(
        "rounded-2xl border bg-surface p-6 shadow-sm",
        toneClasses[tone],
        className
      )}
    >
      {children}
    </section>
  );
}
