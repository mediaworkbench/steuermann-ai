import type { Metadata, Viewport } from "next";
import "./globals.css";
import "katex/dist/katex.min.css";
import "maplibre-gl/dist/maplibre-gl.css";
import { LayoutShell } from "@/components/LayoutShell";
import { ProfileProvider } from "@/hooks/useProfile";
import { ThemeProvider } from "@/hooks/useTheme";
import { I18nProvider } from "@/hooks/useI18n";
import { SessionProvider } from "@/context/SessionContext";


export const metadata: Metadata = {
  title: "Steuermann",
  description: "Modern AI agent orchestration platform",
  icons: {
    icon: { url: "/icon.svg", type: "image/svg+xml" },
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
    <html lang="en" className="font-sans">
      <body className="bg-background text-foreground h-screen overflow-hidden flex flex-col md:flex-row">
        <ThemeProvider>
          <SessionProvider>
            <I18nProvider>
              <ProfileProvider>
                <LayoutShell>{children}</LayoutShell>
              </ProfileProvider>
            </I18nProvider>
          </SessionProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
