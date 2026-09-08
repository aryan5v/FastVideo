"use client";

import React from "react";
import { FolderOpen, Home, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";

export type AppNavSection = "explore" | "create" | "assets";

interface AppNavRailProps {
	activeSection?: AppNavSection;
	onSectionChange?: (section: AppNavSection) => void;
	onOpenProjects?: () => void;
	className?: string;
}

const NAV_ITEMS: Array<{ id: AppNavSection; label: string; icon: typeof Home }> = [
	{ id: "explore", label: "Explore", icon: Home },
	{ id: "create", label: "Create", icon: Sparkles },
	{ id: "assets", label: "Assets", icon: FolderOpen },
];

export default function AppNavRail({
	activeSection = "create",
	onSectionChange = () => {},
	onOpenProjects,
	className,
}: AppNavRailProps) {
	return (
		<aside className={cn("hidden shrink-0 flex-col items-center gap-3 px-3 py-4 lg:flex", className)} aria-label="Primary navigation">
			{NAV_ITEMS.map((item) => {
				const Icon = item.icon;
				const isActive = item.id === activeSection;
				return (
					<button
						key={item.id}
						type="button"
						onClick={() => {
							if (item.id === "assets") {
								onOpenProjects?.();
							}
							onSectionChange(item.id);
						}}
						className={cn(
							"flex w-16 flex-col items-center gap-1 rounded-2xl px-2 py-3 text-[11px] font-medium transition-colors",
							isActive ? "bg-card/80 text-foreground shadow-sm" : "text-muted-foreground hover:bg-card/50 hover:text-foreground",
						)}
					>
						<Icon className={cn("size-5", isActive && "text-accent-blue")} />
						{item.label}
					</button>
				);
			})}
		</aside>
	);
}
