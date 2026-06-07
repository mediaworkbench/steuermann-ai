import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";

interface MetricsDateRangeFieldProps {
  label: string;
  startDate: string;
  endDate: string;
  onStartDateChange: (value: string) => void;
  onEndDateChange: (value: string) => void;
  maxEndDate: string;
  dayCount: number;
  daysLabel: string;
  isRangeInvalid: boolean;
  invalidRangeLabel: string;
}

export function MetricsDateRangeField({
  label,
  startDate,
  endDate,
  onStartDateChange,
  onEndDateChange,
  maxEndDate,
  dayCount,
  daysLabel,
  isRangeInvalid,
  invalidRangeLabel,
}: MetricsDateRangeFieldProps) {
  return (
    <div className="grid gap-2">
      <Label className="mb-0 block">{label}</Label>
      <div className="flex flex-wrap items-center gap-2">
        <Input
          type="date"
          value={startDate}
          onChange={(event) => onStartDateChange(event.target.value)}
          max={endDate}
          className="w-auto min-w-[11rem] rounded-lg px-3 py-2"
        />
        <Input
          type="date"
          value={endDate}
          onChange={(event) => onEndDateChange(event.target.value)}
          min={startDate}
          max={maxEndDate}
          className="w-auto min-w-[11rem] rounded-lg px-3 py-2"
        />
        <span className="inline-flex items-center gap-2 text-sm text-muted-foreground">
          {dayCount} {daysLabel}
          {isRangeInvalid ? <span className="font-semibold text-destructive">{invalidRangeLabel}</span> : null}
        </span>
      </div>
    </div>
  );
}
