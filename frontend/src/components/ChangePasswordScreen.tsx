"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const MIN_LENGTH = 8;

export function ChangePasswordScreen() {
  const router = useRouter();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (newPassword.length < MIN_LENGTH) {
      setError(`New password must be at least ${MIN_LENGTH} characters.`);
      return;
    }
    if (newPassword !== confirmPassword) {
      setError("New passwords do not match.");
      return;
    }
    if (newPassword === currentPassword) {
      setError("New password must differ from the current password.");
      return;
    }

    setSubmitting(true);
    try {
      const response = await fetch("/api/auth/change-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
        setError(payload?.detail || "Could not change password.");
        return;
      }

      router.replace("/");
      router.refresh();
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="fixed inset-0 w-screen h-screen overflow-auto text-foreground"
      style={{ background: "var(--login-main-bg)" }}
    >
      <div className="min-h-full w-full flex items-center justify-center px-6 py-10">
        <section
          className="w-full max-w-xl rounded-4xl border border-border/60 bg-surface/88 p-8 backdrop-blur-xl lg:p-10"
          style={{ boxShadow: "var(--login-panel-shadow)" }}
        >
          <p className="text-xs font-mono uppercase tracking-[0.35em] text-primary/80">Security</p>
          <h1 className="mt-4 text-3xl font-bold tracking-tight">Set a new password</h1>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            Your account uses a temporary password. Choose a new password to continue.
          </p>

          <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
            <label className="block" htmlFor="current-password">
              <span className="mb-2 block text-sm font-semibold text-foreground/80">
                Current (temporary) password
              </span>
              <Input
                id="current-password"
                type="password"
                autoComplete="current-password"
                value={currentPassword}
                onChange={(event) => setCurrentPassword(event.target.value)}
                required
              />
            </label>

            <label className="block" htmlFor="new-password">
              <span className="mb-2 block text-sm font-semibold text-foreground/80">New password</span>
              <Input
                id="new-password"
                type="password"
                autoComplete="new-password"
                value={newPassword}
                onChange={(event) => setNewPassword(event.target.value)}
                required
              />
            </label>

            <label className="block" htmlFor="confirm-password">
              <span className="mb-2 block text-sm font-semibold text-foreground/80">
                Confirm new password
              </span>
              <Input
                id="confirm-password"
                type="password"
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                required
              />
            </label>

            {error && (
              <div
                role="alert"
                className="rounded-2xl border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive"
              >
                {error}
              </div>
            )}

            <Button
              type="submit"
              disabled={submitting}
              variant="primary"
              size="lg"
              className="w-full rounded-full"
            >
              {submitting ? "Saving…" : "Change password"}
            </Button>
          </form>
        </section>
      </div>
    </div>
  );
}
