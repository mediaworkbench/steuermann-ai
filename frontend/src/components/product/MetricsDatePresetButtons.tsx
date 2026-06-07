import { Button } from "@/components/ui/button";

interface MetricsDatePreset {
  key: string;
  label: string;
  days: number;
}

interface MetricsDatePresetButtonsProps {
  presets: MetricsDatePreset[];
  onSelectPreset: (days: number) => void;
}

export function MetricsDatePresetButtons({ presets, onSelectPreset }: MetricsDatePresetButtonsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {presets.map((preset) => (
        <Button
          key={preset.key}
          type="button"
          variant="secondary"
          size="sm"
          onClick={() => onSelectPreset(preset.days)}
        >
          {preset.label}
        </Button>
      ))}
    </div>
  );
}
