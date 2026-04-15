import type { Metadata, Viewport } from "next";
import "./globals.css";
import { LayoutShell } from "@/components/LayoutShell";
import { ProfileProvider } from "@/hooks/useProfile";
import { ThemeProvider } from "@/hooks/useTheme";
import { I18nProvider } from "@/hooks/useI18n";

export const metadata: Metadata = {
  title: "Steuermann",
  description: "Modern AI agent orchestration platform",
  icons: {
    icon: [
      { url: "/favicon.ico" },
      { url: "/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon-96x96.png", sizes: "96x96", type: "image/png" },
    ],
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
  manifest: "/manifest.webmanifest",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-white text-evergreen h-screen overflow-hidden flex flex-col md:flex-row">
        <ThemeProvider>
          <I18nProvider>
            <ProfileProvider>
              <LayoutShell>{children}</LayoutShell>
            </ProfileProvider>
          </I18nProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
