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

// When AUTH_ENABLED=false, this env var sets the assumed role (default "user").
// Set to "administrator" locally to access admin features without enabling full auth.
export const DEV_ROLE =
  (process.env.NEXT_PUBLIC_AUTH_USER_ROLE || "user").toLowerCase() === "administrator"
    ? "administrator" as const
    : "user" as const;
