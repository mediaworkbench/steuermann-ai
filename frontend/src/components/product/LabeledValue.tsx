import * as React from "react";

interface LabeledValueProps {
  label: React.ReactNode;
  value: React.ReactNode;
}

export function LabeledValue({ label, value }: LabeledValueProps) {
  return (
    <div>
      <span className="font-semibold">{label}: </span>
      <span>{value}</span>
    </div>
  );
}
