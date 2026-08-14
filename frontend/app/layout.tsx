import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "ArmForge",
  description: "An Arm-aware inference configuration engine.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="mx-auto max-w-5xl px-5 py-8">
          <header className="mb-8 flex items-center justify-between">
            <Link href="/" className="flex items-baseline gap-2">
              <span className="text-lg font-semibold">ArmForge</span>
              <span className="text-xs text-muted">
                Arm-aware inference configuration engine
              </span>
            </Link>
            <nav className="flex gap-4 text-sm">
              <Link href="/" className="text-muted hover:text-fg">
                Dashboard
              </Link>
              <Link
                href="/new"
                className="rounded border border-line px-3 py-1 hover:border-accent hover:text-accent"
              >
                New optimization
              </Link>
            </nav>
          </header>
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
