import * as React from "react";
import { OptionCheckboxRow } from "@/components/product/OptionCheckboxRow";

type OptionAlignment = "start" | "center";

interface OptionChecklistItem {
  key: string;
  label: React.ReactNode;
  description?: React.ReactNode;
  checked: boolean;
  onToggle: () => void;
  alignment?: OptionAlignment;
  className?: string;
  checkboxClassName?: string;
  descriptionClassName?: string;
}

interface OptionChecklistProps {
  items: OptionChecklistItem[];
  className?: string;
}

export function OptionChecklist({ items, className = "space-y-3" }: OptionChecklistProps) {
  return (
    <div className={className}>
      {items.map((item) => (
        <OptionCheckboxRow
          key={item.key}
          checked={item.checked}
          onToggle={item.onToggle}
          label={item.label}
          description={item.description}
          alignment={item.alignment}
          className={item.className}
          checkboxClassName={item.checkboxClassName}
          descriptionClassName={item.descriptionClassName}
        />
      ))}
    </div>
  );
}
