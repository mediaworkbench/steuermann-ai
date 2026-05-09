import { render, screen } from "@testing-library/react";
import { Sidebar } from "@/components/Sidebar";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";

jest.mock("@/hooks/useProfile");
jest.mock("@/hooks/useI18n");
jest.mock("@/lib/api", () => ({
  searchConversations: jest.fn().mockResolvedValue([]),
}));

const mockUseProfile = useProfile as jest.MockedFunction<typeof useProfile>;
const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;

describe("Sidebar", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseI18n.mockReturnValue({
      locale: "en",
      setLocale: jest.fn(),
      t: (key: string, vars?: Record<string, string | number>) => {
        if (key === "sidebar.settingsForUser" && vars?.name) {
          return `Open settings for ${vars.name}`;
        }
        return key;
      },
      formatDate: () => "",
      formatTime: () => "",
      formatDateTime: () => "",
      formatNumber: (value: number) => String(value),
      formatRelativeTime: () => "",
    });
  });

  test("renders profile-driven branding and settings identity", () => {
    mockUseProfile.mockReturnValue({
      id: "medical",
      displayName: "Medical Assistant",
      roleLabel: "Clinical Assistant",
      appName: "Med Console",
      description: "Clinical profile",
      frameworkVersion: "0.2.1",
      theme: {
        colors: {},
        fonts: {},
        radius: {},
        custom_css_vars: {},
      },
      loading: false,
    });

    render(<Sidebar isOpen={true} onClose={() => {}} conversations={[]} />);

    expect(screen.getByRole("heading", { name: "Med Console" })).toBeInTheDocument();
    expect(screen.getAllByText("Med Console").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByLabelText("Open settings for Med Console")).toBeInTheDocument();
  });

  test("falls back to default app name when profile appName is missing", () => {
    mockUseProfile.mockReturnValue({
      id: "base",
      displayName: "Single User",
      roleLabel: "Local Profile",
      appName: "",
      description: "",
      frameworkVersion: "0.2.1",
      theme: {
        colors: {},
        fonts: {},
        radius: {},
        custom_css_vars: {},
      },
      loading: false,
    });

    render(<Sidebar isOpen={true} onClose={() => {}} conversations={[]} />);

    expect(screen.getByRole("heading", { name: "Steuermann" })).toBeInTheDocument();
  });
});
