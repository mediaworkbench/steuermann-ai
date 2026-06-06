import * as React from "react";
import { SectionCard } from "@/components/product/SectionCard";
import { SectionHeader } from "@/components/product/SectionHeader";

type TitledSectionCardTone = "default" | "danger";

interface TitledSectionCardProps {
  title: React.ReactNode;
  children: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  tone?: TitledSectionCardTone;
  headerClassName?: string;
  className?: string;
}

export function TitledSectionCard({
  title,
  children,
  description,
  actions,
  tone = "default",
  headerClassName,
  className,
}: TitledSectionCardProps) {
  return (
    <SectionCard tone={tone} className={className}>
      <SectionHeader
        title={title}
        description={description}
        actions={actions}
        tone={tone}
        className={headerClassName}
      />
      {children}
    </SectionCard>
  );
}
