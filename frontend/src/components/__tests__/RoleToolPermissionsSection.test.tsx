import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { RoleToolPermissionsSection } from "@/components/product/RoleToolPermissionsSection";
import { useI18n } from "@/hooks/useI18n";
import { fetchRoleTools, updateRoleTools } from "@/lib/api";

jest.mock("@/hooks/useI18n");
jest.mock("@/lib/api", () => ({
  fetchRoleTools: jest.fn(),
  updateRoleTools: jest.fn(),
}));

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;
const mockFetchRoleTools = fetchRoleTools as jest.MockedFunction<typeof fetchRoleTools>;
const mockUpdateRoleTools = updateRoleTools as jest.MockedFunction<typeof updateRoleTools>;

const CATALOG = [
  { id: "web_search_mcp", label: "Web Search", group: "text" as const },
  { id: "datetime_tool", label: "Datetime", group: "auxiliary" as const },
];

describe("RoleToolPermissionsSection", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseI18n.mockReturnValue({
      locale: "en",
      setLocale: jest.fn(),
      t: (key: string) => key,
      formatDate: () => "",
      formatTime: () => "",
      formatDateTime: () => "",
      formatNumber: (value: number) => String(value),
      formatRelativeTime: () => "",
    });
    mockFetchRoleTools.mockResolvedValue({
      tools: CATALOG,
      roles: { user: ["web_search_mcp"], researcher: [] },
    });
    mockUpdateRoleTools.mockResolvedValue({
      tools: CATALOG,
      roles: { user: ["web_search_mcp", "datetime_tool"], researcher: [] },
    });
  });

  test("renders both configurable roles with grouped tools — no a11y violations", async () => {
    const { container } = render(<RoleToolPermissionsSection />);
    await screen.findByText("roleTools.roleUser");
    expect(screen.getByText("roleTools.roleResearcher")).toBeInTheDocument();
    // Each role column lists the full catalog, so a label appears once per role.
    expect(screen.getAllByText("Web Search").length).toBe(2);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("toggling a tool and saving calls updateRoleTools for that role", async () => {
    render(<RoleToolPermissionsSection />);
    await screen.findByText("roleTools.roleUser");

    // The user column is rendered first; turn its Datetime tool on.
    const datetimeCheckboxes = screen.getAllByRole("checkbox", { name: /datetime/i });
    fireEvent.click(datetimeCheckboxes[0]);

    const saveButtons = screen.getAllByText("roleTools.save");
    fireEvent.click(saveButtons[0]);

    await waitFor(() => {
      expect(mockUpdateRoleTools).toHaveBeenCalledWith(
        "user",
        expect.arrayContaining(["web_search_mcp", "datetime_tool"])
      );
    });
  });
});
