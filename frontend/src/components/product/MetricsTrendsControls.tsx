import { Button } from "@/components/ui/Button";
import { RefreshCw, Download } from "lucide-react";
import { MetricsDateRangeField } from "@/components/product/MetricsDateRangeField";
import { MetricsDatePresetButtons } from "@/components/product/MetricsDatePresetButtons";
import { MetricsInlineToggle } from "@/components/product/MetricsInlineToggle";
import {
  MetricsSaveStatusIndicator,
  type MetricsSaveStatus,
} from "@/components/product/MetricsSaveStatusIndicator";

interface MetricsTrendsPreset {
  key: string;
  label: string;
  days: number;
}

interface MetricsTrendsControlsProps {
  dateRangeLabel: string;
  startDate: string;
  endDate: string;
  onStartDateChange: (value: string) => void;
  onEndDateChange: (value: string) => void;
  maxEndDate: string;
  dayCount: number;
  daysLabel: string;
  isRangeInvalid: boolean;
  invalidRangeLabel: string;
  presets: MetricsTrendsPreset[];
  onSelectPreset: (days: number) => void;
  onRefresh: () => void;
  refreshDisabled: boolean;
  refreshLabel: string;
  onExport: () => void;
  exportDisabled: boolean;
  exportLabel: string;
  autoRefresh: boolean;
  onAutoRefreshChange: (checked: boolean) => void;
  autoRefreshLabel: string;
  saveStatus: MetricsSaveStatus;
  savePreferencesLabel: string;
  preferencesSavedLabel: string;
  preferencesFailedLabel: string;
}

export function MetricsTrendsControls({
  dateRangeLabel,
  startDate,
  endDate,
  onStartDateChange,
  onEndDateChange,
  maxEndDate,
  dayCount,
  daysLabel,
  isRangeInvalid,
  invalidRangeLabel,
  presets,
  onSelectPreset,
  onRefresh,
  refreshDisabled,
  refreshLabel,
  onExport,
  exportDisabled,
  exportLabel,
  autoRefresh,
  onAutoRefreshChange,
  autoRefreshLabel,
  saveStatus,
  savePreferencesLabel,
  preferencesSavedLabel,
  preferencesFailedLabel,
}: MetricsTrendsControlsProps) {
  return (
    <div className="mb-4 grid gap-4">
      <div className="grid gap-2">
        <MetricsDateRangeField
          label={dateRangeLabel}
          startDate={startDate}
          endDate={endDate}
          onStartDateChange={onStartDateChange}
          onEndDateChange={onEndDateChange}
          maxEndDate={maxEndDate}
          dayCount={dayCount}
          daysLabel={daysLabel}
          isRangeInvalid={isRangeInvalid}
          invalidRangeLabel={invalidRangeLabel}
        />
        <MetricsDatePresetButtons presets={presets} onSelectPreset={onSelectPreset} />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={onRefresh} disabled={refreshDisabled} variant="secondary" size="sm">
          <RefreshCw className={`h-4 w-4 ${refreshDisabled ? "animate-spin" : ""}`} />
          {refreshLabel}
        </Button>
        <Button onClick={onExport} disabled={exportDisabled} variant="secondary" size="sm">
          <Download className="h-4 w-4" />
          {exportLabel}
        </Button>
        <MetricsInlineToggle
          checked={autoRefresh}
          onToggle={onAutoRefreshChange}
          label={autoRefreshLabel}
        />
      </div>

      <MetricsSaveStatusIndicator
        status={saveStatus}
        savingLabel={savePreferencesLabel}
        savedLabel={preferencesSavedLabel}
        errorLabel={preferencesFailedLabel}
      />
    </div>
  );
}
