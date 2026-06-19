"use client";

import {
  Cloud,
  CloudDrizzle,
  CloudFog,
  CloudLightning,
  CloudRain,
  CloudSnow,
  CloudSun,
  Sun,
  type LucideIcon,
} from "lucide-react";
import type { ReactElement } from "react";
import type { WeatherData, WeatherReading, WeatherForecastDay } from "@/lib/types";
import { useI18n } from "@/hooks/useI18n";
import { weatherCodeGroup, WEATHER_GROUP_I18N, type WeatherConditionGroup } from "@/lib/weather";

const GROUP_ICON: Record<WeatherConditionGroup, LucideIcon> = {
  clear: Sun,
  partlyCloudy: CloudSun,
  overcast: Cloud,
  fog: CloudFog,
  drizzle: CloudDrizzle,
  rain: CloudRain,
  snow: CloudSnow,
  showers: CloudRain,
  thunderstorm: CloudLightning,
  unknown: Cloud,
};

function fmtTemp(value: number, unit: string): string {
  return `${Math.round(value)}${unit}`;
}

interface Props {
  data: WeatherData;
}

export function WeatherWidget({ data }: Props) {
  const { t } = useI18n();

  const conditionLabel = (code: number): string => t(WEATHER_GROUP_I18N[weatherCodeGroup(code)]);
  const ConditionIcon = ({ code, className }: { code: number; className?: string }) => {
    const Icon = GROUP_ICON[weatherCodeGroup(code)];
    return <Icon className={className} aria-hidden="true" />;
  };

  return (
    <div
      role="group"
      aria-label={data.summary}
      className="w-full max-w-sm rounded-lg border border-border bg-surface p-3 shadow-sm"
    >
      {data.type === "current" && data.reading && (
        <CurrentView
          reading={data.reading}
          conditionLabel={conditionLabel}
          ConditionIcon={ConditionIcon}
          t={t}
        />
      )}

      {data.type === "compare" && data.readings && data.readings.length === 2 && (
        <div>
          <p className="text-sm font-medium text-foreground">
            {data.warmer
              ? t("workspace.weatherWarmerBy", {
                  place: data.warmer.split(",")[0],
                  delta: `${Math.round(data.delta ?? 0)}${data.delta_unit ?? ""}`,
                })
              : t("workspace.weatherSameTemp")}
          </p>
          <div className="mt-2 grid grid-cols-2 gap-2">
            {data.readings.map((r, i) => (
              <CompareCard
                key={`${r.label}-${i}`}
                reading={r}
                conditionLabel={conditionLabel}
                ConditionIcon={ConditionIcon}
              />
            ))}
          </div>
        </div>
      )}

      {data.type === "forecast" && data.location && data.days && (
        <div>
          <p className="text-sm font-medium text-foreground">{data.location.label}</p>
          <p className="text-xs text-muted-foreground">
            {t("workspace.weatherForecastTitle", { days: data.days.length })}
          </p>
          <div className="mt-2 flex gap-2 overflow-x-auto pb-1">
            {data.days.map((d) => (
              <ForecastDayCard
                key={d.date}
                day={d}
                conditionLabel={conditionLabel}
                ConditionIcon={ConditionIcon}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function CurrentView({
  reading,
  conditionLabel,
  ConditionIcon,
  t,
}: {
  reading: WeatherReading;
  conditionLabel: (code: number) => string;
  ConditionIcon: (props: { code: number; className?: string }) => ReactElement;
  t: ReturnType<typeof useI18n>["t"];
}) {
  return (
    <div>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-foreground">{reading.label}</p>
          <p className="text-xs text-muted-foreground">{conditionLabel(reading.weather_code)}</p>
        </div>
        <ConditionIcon code={reading.weather_code} className="h-9 w-9 shrink-0 text-primary" />
      </div>
      <p className="mt-1 text-3xl font-semibold text-foreground">
        {fmtTemp(reading.temperature, reading.temperature_unit)}
      </p>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span>{t("workspace.weatherFeelsLike", { temp: fmtTemp(reading.apparent_temperature, reading.temperature_unit) })}</span>
        <span>{t("workspace.weatherHumidity")} {reading.humidity_pct}%</span>
        <span>{t("workspace.weatherWind")} {Math.round(reading.wind_speed)} {reading.wind_speed_unit}</span>
      </div>
    </div>
  );
}

function CompareCard({
  reading,
  conditionLabel,
  ConditionIcon,
}: {
  reading: WeatherReading;
  conditionLabel: (code: number) => string;
  ConditionIcon: (props: { code: number; className?: string }) => ReactElement;
}) {
  return (
    <div className="rounded-md border border-border p-2">
      <p className="truncate text-xs font-medium text-foreground">{reading.label.split(",")[0]}</p>
      <div className="mt-1 flex items-center justify-between gap-1">
        <span className="text-xl font-semibold text-foreground">
          {fmtTemp(reading.temperature, reading.temperature_unit)}
        </span>
        <ConditionIcon code={reading.weather_code} className="h-6 w-6 shrink-0 text-primary" />
      </div>
      <p className="truncate text-xs text-muted-foreground">{conditionLabel(reading.weather_code)}</p>
    </div>
  );
}

function ForecastDayCard({
  day,
  conditionLabel,
  ConditionIcon,
}: {
  day: WeatherForecastDay;
  conditionLabel: (code: number) => string;
  ConditionIcon: (props: { code: number; className?: string }) => ReactElement;
}) {
  const { formatDate } = useI18n();
  // Open-Meteo daily `time` is a date-only string (local calendar day). `new Date("YYYY-MM-DD")`
  // parses as UTC midnight, so format in UTC to avoid an off-by-one weekday west of UTC.
  const weekday = day.date ? formatDate(day.date, { weekday: "short", timeZone: "UTC" }) : day.date;
  return (
    <div className="flex min-w-16 flex-col items-center gap-1 rounded-md border border-border p-2">
      <span className="text-xs font-medium text-foreground">{weekday}</span>
      <ConditionIcon code={day.weather_code} className="h-6 w-6 text-primary" />
      <span className="text-xs text-foreground">{Math.round(day.temp_max)}{day.temperature_unit}</span>
      <span className="text-xs text-muted-foreground">{Math.round(day.temp_min)}{day.temperature_unit}</span>
      <span className="sr-only">{conditionLabel(day.weather_code)}</span>
    </div>
  );
}
