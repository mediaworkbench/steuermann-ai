import { SectionStateText } from "@/components/product/SectionStateText";

interface PanelLoadingStateProps {
  label: string;
}

export function PanelLoadingState({ label }: PanelLoadingStateProps) {
  return (
    <div className="py-8 text-center">
      <SectionStateText>{label}</SectionStateText>
    </div>
  );
}
