"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import { AUTH_ENABLED } from "@/lib/runtime";

interface LoginScreenProps {
  nextPath: string;
}

export function LoginScreen({ nextPath }: LoginScreenProps) {
  const router = useRouter();
  const profile = useProfile();
  const { t } = useI18n();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
        setError(payload?.detail || t("login.loginFailed"));
        return;
      }

      router.replace(nextPath);
      router.refresh();
    } finally {
      setSubmitting(false);
    }
  }

  if (!AUTH_ENABLED) {
    return (
      <div className="min-h-screen text-foreground flex items-center justify-center px-6 py-12" style={{ background: "var(--login-dev-bg)" }}>
        <div className="w-full max-w-xl rounded-4xl border border-border/60 bg-surface/85 backdrop-blur-xl p-8" style={{ boxShadow: "var(--login-card-shadow)" }}>
          <p className="text-xs font-mono uppercase tracking-[0.35em] text-primary/80">{t("login.developmentMode")}</p>
          <h1 className="mt-4 text-4xl font-bold tracking-tight">{t("login.authDisabled")}</h1>
          <p className="mt-4 text-muted-foreground leading-7">
            {t("login.authDisabledDescription")}
          </p>
          <Button asChild variant="primary" size="lg" className="mt-8 rounded-full">
            <Link href="/" prefetch={false}>
            {t("login.enterApplication", { app: profile.appName || t("login.applicationFallback") })}
            </Link>
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 w-screen h-screen overflow-auto text-foreground" style={{ background: "var(--login-main-bg)" }}>
      <div className="min-h-full w-full flex items-center justify-center px-6 py-10">
        <section className="w-full max-w-xl rounded-4xl border border-border/60 bg-surface/88 p-8 backdrop-blur-xl lg:p-10" style={{ boxShadow: "var(--login-panel-shadow)" }}>
          <p className="text-xs font-mono uppercase tracking-[0.35em] text-primary/80">{t("login.login")}</p>
          <h2 className="mt-4 text-3xl font-bold tracking-tight">{t("login.welcomeTo", { app: profile.appName || t("login.platformFallback") })}</h2>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            {t("login.signInAsRole", { role: profile.roleLabel.toLowerCase() })}
          </p>

          <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-foreground/80">{t("login.username")}</span>
              <Input
                type="text"
                autoComplete="username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder={t("login.enterUsername")}
                required
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-foreground/80">{t("login.password")}</span>
              <Input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder={t("login.enterPassword")}
                required
              />
            </label>

            {error && (
              <div className="rounded-2xl border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
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
              {submitting ? t("login.signingIn") : t("login.signIn")}
            </Button>
          </form>
        </section>
      </div>
    </div>
  );
}