"use client";

import type { Metadata } from "next";
import Link from "next/link";
import { Bell, MapPin, Settings, AlertTriangle, MessageSquare, BarChart, Plus, Users } from "lucide-react";
import { usePathname } from "next/navigation";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {

  const pathname = usePathname();

  const navLinks = [
    { href: "/dashboard", icon: BarChart, label: "Dashboard", badge: null },
    { href: "/alerts", icon: Bell, label: "Alerts", badge: 3 },
    { href: "/affected-areas", icon: MapPin, label: "Affected Areas", badge: null },
    { href: "/communications", icon: MessageSquare, label: "Communications", badge: null },
    { href: "/teams", icon: Users, label: "Teams", badge: null },
    { href: "/settings", icon: Settings, label: "Settings", badge: null },
  ]

  const handleNewAlert = () => {
    alert("Initiating new alert creation process!")
    // In a real app, this would navigate to an alert creation form or open a modal
  }

  return (
    <div className="grid min-h-screen w-full lg:grid-cols-[280px_1fr]">
      {/* Sidebar */}
      <div className="hidden border-r bg-muted/40 lg:block">
        <div className="flex h-full max-h-screen flex-col gap-2">
          <div className="flex h-[60px] items-center border-b px-6">
            <Link href="/dashboard" className="flex items-center gap-2 font-semibold">
              <AlertTriangle className="h-6 w-6 text-red-500" />
              <span className="text-lg">CrisisConnect</span>
            </Link>
          </div>
          <nav className="grid items-start px-4 text-sm font-medium lg:px-6">
            {navLinks.map((link) => {
              const Icon = link.icon
              const isActive = pathname === link.href
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-3 rounded-lg px-3 py-2 transition-all hover:text-primary ${
                    isActive ? "bg-muted text-primary" : "text-muted-foreground"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {link.label}
                  {link.badge && (
                    <Badge className="ml-auto flex h-6 w-6 shrink-0 items-center justify-center rounded-full">
                      {link.badge}
                    </Badge>
                  )}
                </Link>
              )
            })}
          </nav>
        </div>
      </div>

      {/* Main Content Wrapper */}
      <div className="flex flex-col">
        <header className="flex h-14 lg:h-[60px] items-center gap-4 border-b bg-muted/40 px-6">
          <Link href="/dashboard" className="lg:hidden flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-6 w-6 text-red-500" />
            <span className="text-lg">CrisisConnect</span>
          </Link>
          <h1 className="font-semibold text-lg md:text-2xl capitalize">
            {pathname.split("/").pop()?.replace("-", " ") || "Dashboard"}
          </h1>
          <Button className="ml-auto" size="sm" onClick={handleNewAlert}>
            <Plus className="h-4 w-4 mr-2" /> New Alert
          </Button>
        </header>
        <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">{children}</main>
      </div>
    </div>
  );
}
