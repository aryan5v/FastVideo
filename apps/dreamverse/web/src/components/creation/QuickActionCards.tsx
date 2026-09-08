"use client";

import React from "react";
import { Film, ImageIcon, Mic, Palette, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";

const QUICK_ACTIONS = [
	{
		id: "canvas",
		label: "Canvas",
		description: "Soon",
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
		description: "Presets",
		icon: ImageIcon,
		accent: "text-orange-400",
	},
	{
		id: "avatar",
		label: "AI Avatar",
		description: "Characters",
		icon: Film,
		accent: "text-violet-400",
	},
	{
		id: "audio",
		label: "AI Audio",
		description: "Text to speech",
		icon: Mic,
		accent: "text-emerald-400",
	},
];

export default function QuickActionCards() {
	return (
		<div className="mx-auto grid w-full max-w-3xl grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
			{QUICK_ACTIONS.map((action) => {
				const Icon = action.icon;
				return (
					<button
						key={action.id}
						type="button"
						disabled
						className={cn(
							"group flex min-h-[74px] flex-col justify-between rounded-xl border border-border/50 bg-secondary/70 p-3 text-left transition-colors hover:border-border hover:bg-secondary/90",
							action.id === "video" && "border-accent-blue/35 bg-accent-blue/[0.08] ring-1 ring-accent-blue/15",
						)}
					>
						<span className={cn("inline-flex size-7 items-center justify-center rounded-lg bg-background/60", action.accent)}>
							<Icon className="size-3.5" />
						</span>
						<span>
							<span className="block text-sm font-medium text-foreground">{action.label}</span>
							<span className="block text-[11px] text-muted-foreground">{action.description}</span>
						</span>
					</button>
				);
			})}
		</div>
	);
}
