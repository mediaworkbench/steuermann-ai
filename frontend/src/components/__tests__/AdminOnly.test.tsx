import { render, screen } from "@testing-library/react";
import { AdminOnly } from "@/components/AdminOnly";
import { useRole } from "@/context/RoleContext";

jest.mock("@/context/RoleContext");

const mockUseRole = useRole as jest.MockedFunction<typeof useRole>;

describe("AdminOnly", () => {
  test("renders null while role is loading", () => {
    mockUseRole.mockReturnValue({ role: "user", isAdmin: false, roleLoading: true });

    const { container } = render(
      <AdminOnly>
        <span>admin content</span>
      </AdminOnly>
    );

    expect(screen.queryByText("admin content")).not.toBeInTheDocument();
    expect(container.firstChild).toBeNull();
  });

  test("renders children when admin and not loading", () => {
    mockUseRole.mockReturnValue({ role: "administrator", isAdmin: true, roleLoading: false });

    render(
      <AdminOnly>
        <span>admin content</span>
      </AdminOnly>
    );

    expect(screen.getByText("admin content")).toBeInTheDocument();
  });

  test("renders null for non-admin with no fallback", () => {
    mockUseRole.mockReturnValue({ role: "user", isAdmin: false, roleLoading: false });

    const { container } = render(
      <AdminOnly>
        <span>admin content</span>
      </AdminOnly>
    );

    expect(screen.queryByText("admin content")).not.toBeInTheDocument();
    expect(container.firstChild).toBeNull();
  });

  test("renders fallback for non-admin", () => {
    mockUseRole.mockReturnValue({ role: "user", isAdmin: false, roleLoading: false });

    render(
      <AdminOnly fallback={<span>access denied</span>}>
        <span>admin content</span>
      </AdminOnly>
    );

    expect(screen.getByText("access denied")).toBeInTheDocument();
    expect(screen.queryByText("admin content")).not.toBeInTheDocument();
  });

  test("renders children and not fallback when admin", () => {
    mockUseRole.mockReturnValue({ role: "administrator", isAdmin: true, roleLoading: false });

    render(
      <AdminOnly fallback={<span>access denied</span>}>
        <span>admin content</span>
      </AdminOnly>
    );

    expect(screen.getByText("admin content")).toBeInTheDocument();
    expect(screen.queryByText("access denied")).not.toBeInTheDocument();
  });

  test("does not render fallback while loading (prevents premature access denied flash)", () => {
    mockUseRole.mockReturnValue({ role: "user", isAdmin: false, roleLoading: true });

    render(
      <AdminOnly fallback={<span>access denied</span>}>
        <span>admin content</span>
      </AdminOnly>
    );

    expect(screen.queryByText("access denied")).not.toBeInTheDocument();
    expect(screen.queryByText("admin content")).not.toBeInTheDocument();
  });
});
