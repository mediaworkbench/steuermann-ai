/**
 * Pure helpers for the weather widget. The backend sends the raw WMO `weather_code`
 * (and an English `condition` fallback); the widget derives both the icon and the
 * localized condition label from the code via the group returned here.
 */

export type WeatherConditionGroup =
  | "clear"
  | "partlyCloudy"
  | "overcast"
  | "fog"
  | "drizzle"
  | "rain"
  | "snow"
  | "showers"
  | "thunderstorm"
  | "unknown";

/** Bucket a WMO weather interpretation code into a renderable condition group. */
export function weatherCodeGroup(code: number): WeatherConditionGroup {
  if (code === 0) return "clear";
  if (code === 1 || code === 2) return "partlyCloudy";
  if (code === 3) return "overcast";
  if (code === 45 || code === 48) return "fog";
  if (code >= 51 && code <= 57) return "drizzle";
  if (code >= 61 && code <= 67) return "rain";
  if ((code >= 71 && code <= 77) || code === 85 || code === 86) return "snow";
  if (code >= 80 && code <= 82) return "showers";
  if (code >= 95 && code <= 99) return "thunderstorm";
  return "unknown";
}

/** i18n key (under `workspace`) for each condition group's label. */
export const WEATHER_GROUP_I18N: Record<WeatherConditionGroup, string> = {
  clear: "workspace.weatherClear",
  partlyCloudy: "workspace.weatherPartlyCloudy",
  overcast: "workspace.weatherOvercast",
  fog: "workspace.weatherFog",
  drizzle: "workspace.weatherDrizzle",
  rain: "workspace.weatherRain",
  snow: "workspace.weatherSnow",
  showers: "workspace.weatherShowers",
  thunderstorm: "workspace.weatherThunderstorm",
  unknown: "workspace.weatherUnknownCondition",
};
