import { Icon } from "../Icon";

interface EvidenceTabPlaceholderProps {
  icon: string;
  title: string;
  hint: string;
}

/**
 * Shared empty/idle state for the read-only evidence tabs (Knowledge, Memory,
 * Outputs). In R1.1 these tabs are placeholders; R1.3 wires them to the shared
 * answer-evidence source.
 */
export function EvidenceTabPlaceholder({ icon, title, hint }: EvidenceTabPlaceholderProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-4 text-center">
      <Icon name={icon} size={32} className="text-evergreen/20 mb-2" />
      <p className="text-xs font-medium text-evergreen/50 mb-1">{title}</p>
      <p className="text-xs text-evergreen/40">{hint}</p>
    </div>
  );
}
