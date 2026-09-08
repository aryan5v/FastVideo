"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
	ArrowUpRight,
	Blocks,
	Cat,
	ChevronLeft,
	ChevronRight,
	Dog,
	Gamepad2,
	type LucideIcon,
	Newspaper,
	PartyPopper,
	Sparkles,
} from "lucide-react";

import { cn } from "@/lib/utils";

export interface StoryPresetLike {
	id: string;
	label: string;
	description?: string;
	segmentCount?: number;
	styleTag?: string;
}

interface PresetQuickLaunchRailProps {
	storyPresets: StoryPresetLike[];
	disabled?: boolean;
	onPresetGenerate: (presetId: string) => void;
}

const PRESET_ACCENTS = [
	{
		surface: "from-sky-500/20 via-sky-400/10 to-indigo-500/25",
		orb: "bg-sky-400/30",
		icon: "bg-sky-500/15 text-sky-700 dark:text-sky-300",
		chip: "border-sky-400/25 bg-sky-500/10 text-sky-700 dark:text-sky-300",
		hover: "hover:border-sky-400/45 hover:shadow-sky-500/15",
	},
	{
		surface: "from-violet-500/20 via-purple-400/10 to-fuchsia-500/25",
		orb: "bg-violet-400/30",
		icon: "bg-violet-500/15 text-violet-700 dark:text-violet-300",
		chip: "border-violet-400/25 bg-violet-500/10 text-violet-700 dark:text-violet-300",
		hover: "hover:border-violet-400/45 hover:shadow-violet-500/15",
	},
	{
		surface: "from-amber-500/20 via-orange-400/10 to-rose-500/25",
		orb: "bg-amber-400/30",
		icon: "bg-amber-500/15 text-amber-800 dark:text-amber-300",
		chip: "border-amber-400/25 bg-amber-500/10 text-amber-800 dark:text-amber-300",
		hover: "hover:border-amber-400/45 hover:shadow-amber-500/15",
	},
	{
		surface: "from-emerald-500/20 via-teal-400/10 to-cyan-500/25",
		orb: "bg-emerald-400/30",
		icon: "bg-emerald-500/15 text-emerald-800 dark:text-emerald-300",
		chip: "border-emerald-400/25 bg-emerald-500/10 text-emerald-800 dark:text-emerald-300",
		hover: "hover:border-emerald-400/45 hover:shadow-emerald-500/15",
	},
	{
		surface: "from-rose-500/20 via-pink-400/10 to-orange-500/25",
		orb: "bg-rose-400/30",
		icon: "bg-rose-500/15 text-rose-700 dark:text-rose-300",
		chip: "border-rose-400/25 bg-rose-500/10 text-rose-700 dark:text-rose-300",
		hover: "hover:border-rose-400/45 hover:shadow-rose-500/15",
	},
] as const;

const PRESET_META: Record<string, { icon: LucideIcon; styleTag: string }> = {
	death_star_console_delay_lego_funny: { icon: Blocks, styleTag: "LEGO comedy" },
	cat_litter_box_clay_custom: { icon: Cat, styleTag: "Stop motion" },
	boy_walking_dog_park_custom: { icon: Dog, styleTag: "Pixar 3D" },
	butterfly_wings_dad: { icon: PartyPopper, styleTag: "Warm comedy" },
	gaming_ban: { icon: Gamepad2, styleTag: "Gaming" },
	garden_sign: { icon: Sparkles, styleTag: "School comedy" },
	oil_strike_reporter: { icon: Newspaper, styleTag: "News satire" },
};

function presetAccent(id: string) {
	let hash = 0;
	for (let i = 0; i < id.length; i += 1) {
		hash = (hash + id.charCodeAt(i) * (i + 1)) % PRESET_ACCENTS.length;
	}
	return PRESET_ACCENTS[hash];
}

function formatSceneCount(segmentCount?: number) {
	if (!segmentCount || segmentCount <= 0) return null;
	return segmentCount === 1 ? "1 scene" : `${segmentCount} scenes`;
}

export default function PresetQuickLaunchRail({
	storyPresets,
	disabled = false,
	onPresetGenerate,
}: PresetQuickLaunchRailProps) {
	const scrollRef = useRef<HTMLDivElement>(null);
	const [canScrollLeft, setCanScrollLeft] = useState(false);
	const [canScrollRight, setCanScrollRight] = useState(false);
	const [presetRailDragging, setPresetRailDragging] = useState(false);
	const presetDragStateRef = useRef({
		pointerId: null as number | null,
		startX: 0,
		startScrollLeft: 0,
		moved: false,
	});
	const suppressPresetClickRef = useRef(false);

	const accentByPresetId = useMemo(() => {
		const map = new Map<string, (typeof PRESET_ACCENTS)[number]>();
		storyPresets.forEach((preset) => map.set(preset.id, presetAccent(preset.id)));
		return map;
	}, [storyPresets]);

	const updateScrollState = useCallback(() => {
		const el = scrollRef.current;
		if (!el) return;
		setCanScrollLeft(el.scrollLeft > 2);
		setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 2);
	}, []);

	const scrollByAmount = useCallback(
		(direction: "left" | "right") => {
			const el = scrollRef.current;
			if (!el) return;
			const delta = direction === "left" ? -248 : 248;
			el.scrollBy({ left: delta, behavior: "smooth" });
			window.setTimeout(updateScrollState, 220);
		},
		[updateScrollState],
	);

	const handlePresetWheel = useCallback(
		(event: React.WheelEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			if (!el) return;
			if (el.scrollWidth <= el.clientWidth + 1) return;

			const dominantDelta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
			if (!dominantDelta) return;

			const maxScrollLeft = Math.max(el.scrollWidth - el.clientWidth, 0);
			const nextScrollLeft = Math.min(Math.max(el.scrollLeft + dominantDelta, 0), maxScrollLeft);
			if (nextScrollLeft === el.scrollLeft) return;

			event.preventDefault();
			el.scrollLeft = nextScrollLeft;
			updateScrollState();
		},
		[updateScrollState],
	);

	const finishPresetDrag = useCallback(() => {
		presetDragStateRef.current = {
			pointerId: null,
			startX: 0,
			startScrollLeft: 0,
			moved: false,
		};
		setPresetRailDragging(false);
	}, []);

	const handlePresetPointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
		const el = scrollRef.current;
		if (!el) return;
		if (event.pointerType !== "mouse" || event.button !== 0) return;
		if (el.scrollWidth <= el.clientWidth + 1) return;

		suppressPresetClickRef.current = false;
		presetDragStateRef.current = {
			pointerId: event.pointerId,
			startX: event.clientX,
			startScrollLeft: el.scrollLeft,
			moved: false,
		};
	}, []);

	const handlePresetPointerMove = useCallback(
		(event: React.PointerEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			const dragState = presetDragStateRef.current;
			if (!el || dragState.pointerId !== event.pointerId) return;

			const deltaX = event.clientX - dragState.startX;
			if (!dragState.moved && Math.abs(deltaX) > 4) {
				dragState.moved = true;
				suppressPresetClickRef.current = true;
				setPresetRailDragging(true);
				el.setPointerCapture?.(event.pointerId);
			}
			if (!dragState.moved) return;

			event.preventDefault();
			const maxScrollLeft = Math.max(el.scrollWidth - el.clientWidth, 0);
			el.scrollLeft = Math.min(Math.max(dragState.startScrollLeft - deltaX, 0), maxScrollLeft);
			updateScrollState();
		},
		[updateScrollState],
	);

	const handlePresetPointerUp = useCallback(
		(event: React.PointerEvent<HTMLDivElement>) => {
			const el = scrollRef.current;
			if (!el || presetDragStateRef.current.pointerId !== event.pointerId) return;
			if (el.hasPointerCapture?.(event.pointerId)) {
				el.releasePointerCapture(event.pointerId);
			}
			finishPresetDrag();
		},
		[finishPresetDrag],
	);

	const handlePresetClickCapture = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
		if (!suppressPresetClickRef.current) return;
		suppressPresetClickRef.current = false;
		event.preventDefault();
		event.stopPropagation();
	}, []);

	useEffect(() => {
		updateScrollState();
	}, [storyPresets, updateScrollState]);

	useEffect(() => {
		const el = scrollRef.current;
		if (!el) return;

		const observer = new ResizeObserver(() => updateScrollState());
		observer.observe(el);
		return () => observer.disconnect();
	}, [updateScrollState]);

	if (storyPresets.length === 0) return null;

	const scrollMaskStyle =
		canScrollLeft && canScrollRight
			? {
					maskImage: "linear-gradient(to right, transparent, black 24px, black calc(100% - 24px), transparent)",
					WebkitMaskImage: "linear-gradient(to right, transparent, black 24px, black calc(100% - 24px), transparent)",
				}
			: canScrollLeft
				? {
						maskImage: "linear-gradient(to right, transparent, black 24px, black)",
						WebkitMaskImage: "linear-gradient(to right, transparent, black 24px, black)",
					}
				: canScrollRight
					? {
							maskImage: "linear-gradient(to right, black, black calc(100% - 24px), transparent)",
							WebkitMaskImage: "linear-gradient(to right, black, black calc(100% - 24px), transparent)",
						}
					: undefined;

	return (
		<div className={cn("mx-auto w-full max-w-3xl transition-opacity duration-200", disabled && "pointer-events-none opacity-40")}>
			<div className="mb-3 flex items-end justify-between gap-3 px-1">
				<div>
					<p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">Suggested prompts</p>
					<p className="mt-1 text-sm text-muted-foreground/90">Curated story starters — pick one to generate instantly.</p>
				</div>
				<p className="hidden text-[11px] text-muted-foreground/80 sm:block">Tap a card to generate</p>
			</div>

			<div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1 sm:gap-2">
				<div className="flex w-7 shrink-0 justify-center sm:w-8">
					{canScrollLeft ? (
						<button
							type="button"
							aria-label="Scroll suggested prompts left"
							onClick={() => scrollByAmount("left")}
							className="inline-flex size-7 items-center justify-center rounded-full border border-border/60 bg-background/95 text-muted-foreground shadow-sm backdrop-blur-sm transition hover:text-foreground sm:size-8"
						>
							<ChevronLeft className="size-4" />
						</button>
					) : null}
				</div>

				<div
					ref={scrollRef}
					onScroll={updateScrollState}
					onWheel={handlePresetWheel}
					onPointerDown={handlePresetPointerDown}
					onPointerMove={handlePresetPointerMove}
					onPointerUp={handlePresetPointerUp}
					onPointerCancel={handlePresetPointerUp}
					onLostPointerCapture={finishPresetDrag}
					onClickCapture={handlePresetClickCapture}
					style={scrollMaskStyle}
					className={cn(
						"scrollbar-hidden flex gap-3 overflow-x-auto overflow-y-visible py-1 select-none",
						presetRailDragging ? "cursor-grabbing" : "cursor-grab",
					)}
				>
					{storyPresets.map((preset) => {
						const accent = accentByPresetId.get(preset.id) ?? PRESET_ACCENTS[0];
						const meta = PRESET_META[preset.id];
						const Icon = meta?.icon ?? Sparkles;
						const styleTag = preset.styleTag || meta?.styleTag;
						const sceneCount = formatSceneCount(preset.segmentCount);

						return (
							<button
								key={preset.id}
								type="button"
								disabled={disabled}
								onClick={() => onPresetGenerate(preset.id)}
								className={cn(
									"group relative isolate flex h-[11.75rem] w-[13.75rem] shrink-0 flex-col overflow-hidden rounded-2xl border border-border/60 bg-card/95 text-left shadow-sm backdrop-blur-sm transition-[border-color,box-shadow,background-color] duration-200",
									"hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-blue/40",
									accent.hover,
								)}
							>
								<div className={cn("relative h-[4.75rem] shrink-0 overflow-hidden bg-gradient-to-br px-3.5 py-3", accent.surface)}>
									<span
										aria-hidden="true"
										className={cn("pointer-events-none absolute -right-4 -top-6 size-24 rounded-full blur-2xl", accent.orb)}
									/>
									<div className="relative flex min-w-0 items-start justify-between gap-2">
										<span className={cn("inline-flex size-9 shrink-0 items-center justify-center rounded-xl border border-white/10 shadow-sm", accent.icon)}>
											<Icon className="size-4" />
										</span>
										{styleTag && (
											<span
												className={cn(
													"max-w-[7.25rem] truncate rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.08em]",
													accent.chip,
												)}
											>
												{styleTag}
											</span>
										)}
									</div>
								</div>

								<div className="flex min-h-0 flex-1 flex-col gap-2 px-3.5 py-3">
									<div className="flex min-w-0 items-start justify-between gap-2">
										<span className="line-clamp-2 min-h-[2.5rem] flex-1 text-sm font-semibold leading-5 text-foreground">{preset.label}</span>
										<span className="inline-flex size-7 shrink-0 items-center justify-center rounded-full border border-border/40 bg-background/70 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">
											<ArrowUpRight className="size-3.5" />
										</span>
									</div>
									{preset.description ? (
										<span className="line-clamp-2 min-h-[2.5rem] text-xs leading-5 text-muted-foreground">{preset.description}</span>
									) : (
										<span className="min-h-[2.5rem]" aria-hidden="true" />
									)}
									<div className="mt-auto flex items-center justify-between gap-2 pt-1">
										{sceneCount ? (
											<span className="rounded-full border border-border/50 bg-muted/40 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
												{sceneCount}
											</span>
										) : (
											<span aria-hidden="true" />
										)}
										<span className="text-[10px] font-medium uppercase tracking-[0.12em] text-accent-blue opacity-0 transition-opacity group-hover:opacity-100">
											Generate
										</span>
									</div>
								</div>
							</button>
						);
					})}
				</div>

				<div className="flex w-7 shrink-0 justify-center sm:w-8">
					{canScrollRight ? (
						<button
							type="button"
							aria-label="Scroll suggested prompts right"
							onClick={() => scrollByAmount("right")}
							className="inline-flex size-7 items-center justify-center rounded-full border border-border/60 bg-background/95 text-muted-foreground shadow-sm backdrop-blur-sm transition hover:text-foreground sm:size-8"
						>
							<ChevronRight className="size-4" />
						</button>
					) : null}
				</div>
			</div>
		</div>
	);
}
