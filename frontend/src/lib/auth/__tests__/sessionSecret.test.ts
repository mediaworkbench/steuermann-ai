import { MIN_SESSION_SECRET_LENGTH, isWeakSessionSecret } from "@/lib/auth/sessionSecret";

describe("isWeakSessionSecret", () => {
  test.each(["change-me", "Change-Me", "changeme", "secret", "your-secret"])(
    "flags the known default %s",
    (secret) => {
      expect(isWeakSessionSecret(secret)).toBe(true);
    }
  );

  test("flags a too-short custom secret", () => {
    expect(isWeakSessionSecret("a".repeat(MIN_SESSION_SECRET_LENGTH - 1))).toBe(true);
  });

  test("accepts a secret at the minimum length", () => {
    expect(isWeakSessionSecret("a".repeat(MIN_SESSION_SECRET_LENGTH))).toBe(false);
  });

  test("accepts a strong 64-char secret", () => {
    expect(isWeakSessionSecret("a".repeat(64))).toBe(false);
  });
});
