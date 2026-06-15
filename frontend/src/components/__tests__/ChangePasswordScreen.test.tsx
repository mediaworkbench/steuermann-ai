import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { ChangePasswordScreen } from "@/components/ChangePasswordScreen";
import { hardNavigate } from "@/lib/navigation";

jest.mock("@/lib/navigation", () => ({ hardNavigate: jest.fn() }));
const mockNavigate = hardNavigate as jest.Mock;

describe("ChangePasswordScreen", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.fetch = jest.fn();
  });

  afterEach(() => {
    delete (global as Record<string, unknown>).fetch;
  });

  test("has no axe violations", async () => {
    const { container } = render(<ChangePasswordScreen />);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("rejects mismatched new passwords without calling the API", async () => {
    render(<ChangePasswordScreen />);
    fireEvent.change(screen.getByLabelText(/current/i), { target: { value: "temp-pass" } });
    fireEvent.change(screen.getByLabelText("New password"), { target: { value: "longenough1" } });
    fireEvent.change(screen.getByLabelText(/confirm/i), { target: { value: "different1" } });
    fireEvent.click(screen.getByRole("button", { name: /change password/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/do not match/i));
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test("submits and redirects on success", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, json: async () => ({ ok: true }) });
    render(<ChangePasswordScreen />);
    fireEvent.change(screen.getByLabelText(/current/i), { target: { value: "temp-pass" } });
    fireEvent.change(screen.getByLabelText("New password"), { target: { value: "brand-new-pass" } });
    fireEvent.change(screen.getByLabelText(/confirm/i), { target: { value: "brand-new-pass" } });
    fireEvent.click(screen.getByRole("button", { name: /change password/i }));

    await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith("/"));
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/auth/change-password",
      expect.objectContaining({ method: "POST" })
    );
  });
});
