"use client";

import React from "react";
import { Film, ImageIcon, Mic, Palette, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";

const QUICK_ACTIONS = [
	{
		id: "canvas",
		label: "Canvas",
		description: "Try it now",
		icon: Palette,
		accent: "text-sky-400",
	},
	{
		id: "video",
		label: "AI Video",
		description: "FastLTX 2.3",
		icon: Sparkles,
		accent: "text-accent-blue",
	},
	{
		id: "image",
		label: "AI Image",
		description: "Dream presets",
		icon: ImageIcon,
		accent: "text-orange-400",
	},
	{
		id: "avatar",
		label: "AI Avatar",
		description: "Character refs",
		icon: Film,
		accent: "text-violet-400",
	},
	{
		id: "audio",
		label: "AI Audio",
		description: "Turn text into speech",
		icon: Mic,
		accent: "text-emerald-400",
	},
];

export default function QuickActionCards() {
	return (
		<div className="mx-auto grid w-full max-w-3xl grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
			{QUICK_ACTIONS.map((action) => {
				const Icon = action.icon;
				return (
					<button
						key={action.id}
						type="button"
						disabled
						className={cn(
							"flex min-h-[88px] flex-col justify-between rounded-2xl border border-border/70 bg-card/60 p-3 text-left opacity-80 transition-colors",
							action.id === "video" && "border-accent-blue/30 bg-accent-blue/5",
						)}
					>
						<Icon className={cn("size-4", action.accent)} />
						<span>
							<span className="block text-sm font-medium text-foreground">{action.label}</span>
							<span className="block text-xs text-muted-foreground">{action.description}</span>
						</span>
					</button>
				);
			})}
		</div>
	);
}
