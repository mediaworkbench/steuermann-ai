"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { ShipWheel } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import { AUTH_ENABLED } from "@/lib/runtime";
import { hardNavigate } from "@/lib/navigation";

interface LoginScreenProps {
  nextPath: string;
}

export function LoginScreen({ nextPath }: LoginScreenProps) {
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

      hardNavigate(nextPath);
    } finally {
      setSubmitting(false);
    }
  }

  if (!AUTH_ENABLED) {
    return (
      <div className="flex min-h-svh w-full flex-col items-center justify-center bg-muted p-6 md:p-10">
        <div className="w-full max-w-sm md:max-w-xl">
          <Card>
            <CardHeader>
              <CardTitle className="text-xs font-mono uppercase tracking-widest text-primary/80">
                {t("login.developmentMode")}
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4">
              <h1 className="text-2xl font-bold tracking-tight">{t("login.authDisabled")}</h1>
              <p className="text-sm text-muted-foreground leading-6">
                {t("login.authDisabledDescription")}
              </p>
              <Button asChild variant="default" className="w-full">
                <Link href="/" prefetch={false}>
                  {t("login.enterApplication", { app: profile.appName || t("login.applicationFallback") })}
                </Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-svh w-full flex-col items-center justify-center bg-muted p-6 md:p-10">
      <div className="w-full max-w-sm md:max-w-md">
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader className="text-center">
              <div className="mb-2 flex justify-center">
                <ShipWheel className="size-8 text-foreground" />
              </div>
              <CardTitle className="text-xl">
                {t("login.welcomeTo", { app: profile.appName || t("login.platformFallback") })}
              </CardTitle>
              <CardDescription>
                {t("login.signInAsRole")}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit}>
                <div className="flex flex-col gap-4">
                  <div className="grid gap-2">
                    <Label htmlFor="username">{t("login.username")}</Label>
                    <Input
                      id="username"
                      type="text"
                      autoComplete="username"
                      value={username}
                      onChange={(event) => setUsername(event.target.value)}
                      placeholder={t("login.enterUsername")}
                      required
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="password">{t("login.password")}</Label>
                    <Input
                      id="password"
                      type="password"
                      autoComplete="current-password"
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      placeholder={t("login.enterPassword")}
                      required
                    />
                    <p className="text-xs text-muted-foreground">
                      Forgot your password? Contact your Administrator.
                    </p>
                  </div>
                  {error && (
                    <div
                      role="alert"
                      className="rounded-md border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive"
                    >
                      {error}
                    </div>
                  )}
                  <Button
                    type="submit"
                    disabled={submitting}
                    className="w-full bg-foreground text-background hover:bg-foreground/90"
                  >
                    {submitting ? t("login.signingIn") : t("login.signIn")}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
          <div className="px-6 text-balance text-center text-xs text-muted-foreground [&_a]:underline [&_a]:underline-offset-4 [&_a]:hover:text-primary">
            By clicking continue, you agree to our{" "}
            <a href="/terms">Terms of Service</a> and <a href="/privacy">Privacy Policy</a>.
          </div>
        </div>
      </div>
    </div>
  );
}
