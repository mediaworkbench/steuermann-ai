export const AUTH_ENABLED = ["true", "1", "yes"].includes(
  (process.env.NEXT_PUBLIC_AUTH_ENABLED || "false").toLowerCase()
);

export const CURRENT_USER_ID = process.env.NEXT_PUBLIC_SINGLE_USER_ID || "anonymous";
export const SINGLE_USER_DISPLAY_NAME =
  process.env.NEXT_PUBLIC_SINGLE_USER_DISPLAY_NAME || "Single User";
export const SINGLE_USER_APP_NAME =
  process.env.NEXT_PUBLIC_SINGLE_USER_APP_NAME || process.env.NEXT_PUBLIC_APP_NAME || "Steuermann";
export const SINGLE_USER_ROLE_LABEL =
  process.env.NEXT_PUBLIC_SINGLE_USER_ROLE_LABEL || "Local Profile";
export const SINGLE_USER_EMAIL = process.env.NEXT_PUBLIC_SINGLE_USER_EMAIL || "";
