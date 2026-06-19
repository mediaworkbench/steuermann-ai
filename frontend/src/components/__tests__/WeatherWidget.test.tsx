import { render, screen } from "@testing-library/react";
import { axe, toHaveNoViolations } from "jest-axe";
import { WeatherWidget } from "@/components/WeatherWidget";
import type { WeatherData, WeatherReading } from "@/lib/types";
import { useI18n } from "@/hooks/useI18n";

expect.extend(toHaveNoViolations);

jest.mock("@/hooks/useI18n");
const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;

beforeEach(() => {
  mockUseI18n.mockReturnValue({
    locale: "en",
    setLocale: jest.fn(),
    // Return the key so we can assert which condition/label was chosen.
    t: (key: string) => key,
    formatDate: () => "Wed",
    formatTime: () => "",
    formatDateTime: () => "",
    formatNumber: (value: number) => String(value),
    formatRelativeTime: () => "",
  } as unknown as ReturnType<typeof useI18n>);
});

function reading(label: string, temperature: number, weather_code: number): WeatherReading {
  return {
    label,
    lat: 0,
    lon: 0,
    country: "",
    timezone: "",
    temperature,
    apparent_temperature: temperature - 1,
    weather_code,
    condition: "x",
    humidity_pct: 60,
    wind_speed: 12,
    precipitation: 0,
    temperature_unit: "°C",
    wind_speed_unit: "km/h",
    precipitation_unit: "mm",
    observed_at: "",
  };
}

test("current: renders temperature, condition group and feels-like", () => {
  const data: WeatherData = {
    type: "current",
    summary: "Barcelona, Spain: 25°C",
    reading: reading("Barcelona, Spain", 25, 2),
  };
  render(<WeatherWidget data={data} />);
  expect(screen.getByText("25°C")).toBeInTheDocument();
  expect(screen.getByText("workspace.weatherPartlyCloudy")).toBeInTheDocument();
  expect(screen.getByText("workspace.weatherFeelsLike")).toBeInTheDocument();
});

test("compare: renders the warmer banner and both city temps", () => {
  const data: WeatherData = {
    type: "compare",
    summary: "Barcelona is warmer",
    warmer: "Barcelona, Spain",
    delta: 12,
    delta_unit: "°C",
    readings: [reading("Barcelona, Spain", 25, 2), reading("Schwerin, Germany", 13, 61)],
  };
  render(<WeatherWidget data={data} />);
  expect(screen.getByText("workspace.weatherWarmerBy")).toBeInTheDocument();
  expect(screen.getByText("25°C")).toBeInTheDocument();
  expect(screen.getByText("13°C")).toBeInTheDocument();
  expect(screen.getByText("Barcelona")).toBeInTheDocument();
});

test("forecast: renders day cards and the forecast title", () => {
  const data: WeatherData = {
    type: "forecast",
    summary: "3-day forecast",
    location: { label: "Berlin, Germany", lat: 0, lon: 0, country: "Germany", timezone: "" },
    days: [
      { date: "2026-06-18", weather_code: 2, condition: "x", temp_max: 24, temp_min: 15, precipitation: 0, temperature_unit: "°C", precipitation_unit: "mm" },
      { date: "2026-06-19", weather_code: 61, condition: "x", temp_max: 19, temp_min: 12, precipitation: 4, temperature_unit: "°C", precipitation_unit: "mm" },
    ],
  };
  render(<WeatherWidget data={data} />);
  expect(screen.getByText("workspace.weatherForecastTitle")).toBeInTheDocument();
  expect(screen.getByText("24°C")).toBeInTheDocument();
  expect(screen.getByText("19°C")).toBeInTheDocument();
});

test("has no axe violations (current)", async () => {
  const data: WeatherData = {
    type: "current",
    summary: "Barcelona, Spain: 25°C",
    reading: reading("Barcelona, Spain", 25, 2),
  };
  const { container } = render(<WeatherWidget data={data} />);
  expect(await axe(container)).toHaveNoViolations();
});
