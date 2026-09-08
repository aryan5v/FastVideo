"use client";

import React from "react";

import { cn } from "@/lib/utils";

export default function ConfigPill({
	children,
	className,
	...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
	return (
		<button
			type="button"
			className={cn(
				"inline-flex h-7 shrink-0 items-center gap-1 rounded-full border border-border/50 bg-background/80 px-2.5 text-[11px] font-medium text-foreground/90 transition-colors hover:border-border hover:bg-accent/50",
				className,
			)}
			{...props}
		>
			{children}
		</button>
	);
}
