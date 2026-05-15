import { render, screen, waitFor } from "@testing-library/react";
import { ProfileProvider, useProfile } from "@/hooks/useProfile";
import { fetchSystemConfig } from "@/lib/api";

jest.mock("@/lib/api", () => ({
  fetchSystemConfig: jest.fn(),
}));

const mockFetchSystemConfig = fetchSystemConfig as jest.MockedFunction<typeof fetchSystemConfig>;

function ProfileProbe() {
  const profile = useProfile();
  return (
    <div>
      <span data-testid="profile-id">{profile.id}</span>
      <span data-testid="display-name">{profile.displayName}</span>
      <span data-testid="role-label">{profile.roleLabel}</span>
      <span data-testid="framework-version">{profile.frameworkVersion}</span>
      <span data-testid="loading">{String(profile.loading)}</span>
    </div>
  );
}

describe("useProfile", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    document.documentElement.removeAttribute("data-profile");
    document.documentElement.style.removeProperty("--primary");
    document.documentElement.style.removeProperty("--font-display");
    document.documentElement.style.removeProperty("--card-radius");
    document.documentElement.style.removeProperty("--surface-muted");
  });

  test("loads profile and applies theme tokens to root element", async () => {
    mockFetchSystemConfig.mockResolvedValue({
      available_tools: [],
      model_roles: [],
      rag_defaults: { collection_name: "framework", top_k: 5 },
      default_model: "base-model",
      framework_version: "0.2.1",
      supported_languages: ["en"],
      profile: {
        id: "medical",
        display_name: "Medical Assistant",
        role_label: "Clinical Assistant",
        app_name: "Med Console",
        description: "Clinical profile",
        theme: {
          colors: { primary: "#008080" },
          fonts: { font_display: "\"IBM Plex Sans\", sans-serif" },
          radius: { card_radius: "14px" },
          custom_css_vars: { surface_muted: "#f3f8f8" },
        },
      },
    });

    render(
      <ProfileProvider>
        <ProfileProbe />
      </ProfileProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId("profile-id")).toHaveTextContent("medical");
      expect(screen.getByTestId("display-name")).toHaveTextContent("Medical Assistant");
      expect(screen.getByTestId("role-label")).toHaveTextContent("Clinical Assistant");
      expect(screen.getByTestId("framework-version")).toHaveTextContent("0.2.1");
      expect(screen.getByTestId("loading")).toHaveTextContent("false");
    });

    expect(document.documentElement.getAttribute("data-profile")).toBe("medical");
    expect(document.documentElement.style.getPropertyValue("--primary")).toBe("#008080");
    expect(document.documentElement.style.getPropertyValue("--font-display")).toBe(
      '"IBM Plex Sans", sans-serif'
    );
    expect(document.documentElement.style.getPropertyValue("--card-radius")).toBe("14px");
    expect(document.documentElement.style.getPropertyValue("--surface-muted")).toBe("#f3f8f8");
  });

  test("throws when useProfile is used outside provider", () => {
    const renderOutsideProvider = () => render(<ProfileProbe />);
    expect(renderOutsideProvider).toThrow("useProfile must be used within a ProfileProvider");
  });
});
