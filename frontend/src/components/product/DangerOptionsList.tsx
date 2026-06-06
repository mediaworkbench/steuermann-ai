import * as React from "react";
import { OptionChecklist } from "@/components/product/OptionChecklist";

interface DangerOptionItem {
  key: string;
  label: React.ReactNode;
  description: React.ReactNode;
  checked: boolean;
  onToggle: () => void;
  className?: string;
}

interface DangerOptionsListProps {
  options: DangerOptionItem[];
  className?: string;
}

export function DangerOptionsList({ options, className = "space-y-3 mb-5" }: DangerOptionsListProps) {
  return <OptionChecklist items={options} className={className} />;
}
