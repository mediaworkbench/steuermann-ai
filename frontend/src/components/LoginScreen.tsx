"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
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
      <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(39,95,87,0.16),transparent_40%),linear-gradient(135deg,#f7fbfb,#eaf5f4)] text-evergreen flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-xl rounded-4xl border border-evergreen/10 bg-white/85 backdrop-blur-xl p-8 shadow-[0_30px_80px_rgba(18,55,51,0.12)]">
          <p className="text-xs font-mono uppercase tracking-[0.35em] text-pacific-blue/80">{t("login.developmentMode")}</p>
          <h1 className="mt-4 text-4xl font-bold tracking-tight">{t("login.authDisabled")}</h1>
          <p className="mt-4 text-evergreen/70 leading-7">
            {t("login.authDisabledDescription")}
          </p>
          <Link
            href="/"
            prefetch={false}
            className="mt-8 inline-flex items-center rounded-full bg-evergreen px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-evergreen/90"
          >
            {t("login.enterApplication", { app: profile.appName || t("login.applicationFallback") })}
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 w-screen h-screen overflow-auto bg-[radial-gradient(circle_at_top_left,rgba(39,95,87,0.22),transparent_38%),radial-gradient(circle_at_bottom_right,rgba(248,145,64,0.18),transparent_32%),linear-gradient(145deg,#f6fbfa,#e3f1ee_55%,#fdf7f1)] text-evergreen">
      <div className="min-h-full w-full flex items-center justify-center px-6 py-10">
        <section className="w-full max-w-xl rounded-4xl border border-evergreen/10 bg-white/88 p-8 shadow-[0_28px_80px_rgba(18,55,51,0.12)] backdrop-blur-xl lg:p-10">
          <p className="text-xs font-mono uppercase tracking-[0.35em] text-pacific-blue/80">{t("login.login")}</p>
          <h2 className="mt-4 text-3xl font-bold tracking-tight">{t("login.welcomeTo", { app: profile.appName || t("login.platformFallback") })}</h2>
          <p className="mt-3 text-sm leading-6 text-evergreen/70">
            {t("login.signInAsRole", { role: profile.roleLabel.toLowerCase() })}
          </p>

          <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-evergreen/80">{t("login.username")}</span>
              <input
                type="text"
                autoComplete="username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                className="w-full rounded-2xl border border-evergreen/10 bg-white px-4 py-3 text-evergreen shadow-sm outline-none transition-colors focus:border-pacific-blue/60"
                placeholder={t("login.enterUsername")}
                required
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-evergreen/80">{t("login.password")}</span>
              <input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="w-full rounded-2xl border border-evergreen/10 bg-white px-4 py-3 text-evergreen shadow-sm outline-none transition-colors focus:border-pacific-blue/60"
                placeholder={t("login.enterPassword")}
                required
              />
            </label>

            {error && (
              <div className="rounded-2xl border border-burnt-tangerine/20 bg-burnt-tangerine/8 px-4 py-3 text-sm text-burnt-tangerine">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="inline-flex w-full items-center justify-center rounded-full bg-evergreen px-6 py-3.5 text-sm font-semibold text-white transition-colors hover:bg-evergreen/92 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {submitting ? t("login.signingIn") : t("login.signIn")}
            </button>
          </form>
        </section>
      </div>
    </div>
  );
}