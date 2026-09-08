"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Film } from "lucide-react";

import { cn } from "@/lib/utils";

export interface StoryPresetLike {
	id: string;
	label: string;
	description?: string;
}

interface PresetQuickLaunchRailProps {
	storyPresets: StoryPresetLike[];
	disabled?: boolean;
	onPresetGenerate: (presetId: string) => void;
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

	const updateScrollState = useCallback(() => {
		const el = scrollRef.current;
		if (!el) return;
		setCanScrollLeft(el.scrollLeft > 2);
		setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 2);
	}, []);

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

	if (storyPresets.length === 0) return null;

	return (
		<div className={cn("relative mx-auto w-full max-w-3xl transition-opacity duration-200", disabled && "pointer-events-none opacity-40")}>
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
				className={cn(
					"scrollbar-hidden flex gap-3 overflow-x-auto px-1 select-none",
					presetRailDragging ? "cursor-grabbing" : "cursor-grab",
				)}
			>
				{storyPresets.map((preset) => (
					<button
						key={preset.id}
						type="button"
						disabled={disabled}
						onClick={() => onPresetGenerate(preset.id)}
						className="flex max-w-42 shrink-0 flex-col items-start gap-1.5 rounded-xl border border-input bg-card/80 p-2.5 text-left text-muted-foreground backdrop-blur-sm transition-colors hover:border-slate-400 hover:bg-slate-200/60 hover:text-slate-700 sm:max-w-[215px] sm:flex-row dark:bg-slate-800/80 dark:text-slate-300 dark:hover:border-slate-500 dark:hover:bg-slate-700/50 dark:hover:text-slate-200"
					>
						<Film className="mt-0.5 size-4 shrink-0 opacity-60" />
						<span className="flex min-w-0 flex-col gap-1">
							<span className="line-clamp-1 text-[14px] font-medium">{preset.label}</span>
							{preset.description && <span className="line-clamp-3 text-xs leading-tight opacity-70 sm:line-clamp-2">{preset.description}</span>}
						</span>
					</button>
				))}
			</div>

			<div
				className={cn(
					"pointer-events-none absolute inset-y-0 left-0 w-8 bg-background transition-opacity duration-150",
					canScrollLeft ? "opacity-100" : "opacity-0",
				)}
				style={{ maskImage: "linear-gradient(to right, black, transparent)", WebkitMaskImage: "linear-gradient(to right, black, transparent)" }}
				aria-hidden="true"
			/>
			<div
				className={cn(
					"pointer-events-none absolute inset-y-0 right-0 w-8 bg-background transition-opacity duration-150",
					canScrollRight ? "opacity-100" : "opacity-0",
				)}
				style={{ maskImage: "linear-gradient(to left, black, transparent)", WebkitMaskImage: "linear-gradient(to left, black, transparent)" }}
				aria-hidden="true"
			/>
		</div>
	);
}
