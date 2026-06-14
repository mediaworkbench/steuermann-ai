import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { AdminUsersPanel } from "@/components/AdminUsersPanel";
import * as api from "@/lib/api";

jest.mock("sonner", () => ({ toast: { error: jest.fn(), success: jest.fn() } }));
jest.mock("@/hooks/useI18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));
jest.mock("@/context/SessionContext", () => ({
  useCurrentUser: () => ({ userId: "self-id", username: "boss" }),
}));
jest.mock("@/lib/api", () => ({
  fetchUsers: jest.fn(),
  fetchRoles: jest.fn(),
  createUser: jest.fn(),
  updateUser: jest.fn(),
  deleteUser: jest.fn(),
}));

const mockApi = api as jest.Mocked<typeof api>;

function makeUser(overrides: Partial<api.AdminUser> = {}): api.AdminUser {
  return {
    user_id: "u-1",
    username: "alice",
    email: "alice@example.com",
    role_name: "user",
    status: "active",
    must_change_password: false,
    created_at: null,
    updated_at: null,
    ...overrides,
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  mockApi.fetchRoles.mockResolvedValue([
    { role_id: 1, role_name: "user" },
    { role_id: 2, role_name: "researcher" },
    { role_id: 3, role_name: "administrator" },
  ]);
  mockApi.fetchUsers.mockResolvedValue({
    users: [
      makeUser(),
      makeUser({ user_id: "self-id", username: "boss", email: "boss@example.com", role_name: "administrator" }),
    ],
    total: 2,
  });
});

test("has no axe violations", async () => {
  const { container } = render(<AdminUsersPanel />);
  await waitFor(() => expect(screen.getByText("alice")).toBeInTheDocument());
  expect(await axe(container)).toHaveNoViolations();
});

test("lists users from the API", async () => {
  render(<AdminUsersPanel />);
  await waitFor(() => expect(screen.getByText("alice")).toBeInTheDocument());
  expect(screen.getByText("alice@example.com")).toBeInTheDocument();
  // self row is annotated
  expect(screen.getByText("(you)")).toBeInTheDocument();
});

test("create user shows the temporary password once", async () => {
  mockApi.createUser.mockResolvedValue({
    user: makeUser({ user_id: "u-2", username: "newbie", must_change_password: true }),
    temporary_password: "tmp-secret-123",
  });
  render(<AdminUsersPanel />);
  await waitFor(() => expect(screen.getByText("alice")).toBeInTheDocument());

  fireEvent.change(screen.getByLabelText("Username"), { target: { value: "newbie" } });
  fireEvent.change(screen.getByLabelText("Email"), { target: { value: "newbie@example.com" } });
  fireEvent.click(screen.getByRole("button", { name: /create/i }));

  await waitFor(() =>
    expect(mockApi.createUser).toHaveBeenCalledWith({
      username: "newbie",
      email: "newbie@example.com",
      role: "user",
    }),
  );
  expect(await screen.findByText("tmp-secret-123")).toBeInTheDocument();
});

test("self row cannot be deleted", async () => {
  render(<AdminUsersPanel />);
  await waitFor(() => expect(screen.getByText("boss")).toBeInTheDocument());
  expect(screen.getByRole("button", { name: "Delete boss" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Delete alice" })).toBeEnabled();
});
