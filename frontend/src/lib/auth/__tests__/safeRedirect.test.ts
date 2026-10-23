import { sanitizeNextPath } from "../safeRedirect";

describe("sanitizeNextPath", () => {
  it("keeps same-origin relative paths", () => {
    expect(sanitizeNextPath("/chats?q=a")).toBe("/chats?q=a");
    expect(sanitizeNextPath("/admin/rag")).toBe("/admin/rag");
  });

  it.each([
    undefined,
    "",
    "chats",
    "//evil.com",
    "/\\evil.com",
    "/\\/evil.com",
    "/\t/evil.com",
    "/\n/evil.com",
    "https://evil.com",
  ])("rejects %p", (input) => {
    expect(sanitizeNextPath(input)).toBe("/");
  });
});
