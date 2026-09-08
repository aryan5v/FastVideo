"use client";

import React from "react";
import { Clock3, Play } from "lucide-react";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
	DISCOVERY_TABS,
	MOCK_DISCOVERY_ASSETS,
	type AspectRatioId,
	type DiscoveryAsset,
} from "@/lib/creationConfig";
import { cn } from "@/lib/utils";

const ASPECT_CLASSES: Record<AspectRatioId, string> = {
	"21:9": "aspect-[21/9]",
	"16:9": "aspect-video",
	"4:3": "aspect-[4/3]",
	"1:1": "aspect-square",
	"3:4": "aspect-[3/4]",
	"9:16": "aspect-[9/16]",
};

function AssetCard({ asset }: { asset: DiscoveryAsset }) {
	if (asset.featured) {
		return (
			<article className="relative mb-3 overflow-hidden rounded-2xl border border-border/50 bg-secondary/80 p-5 shadow-sm sm:min-h-[200px]">
				<div className={cn("absolute inset-0 bg-gradient-to-br opacity-90", asset.gradient)} />
				<div className="relative z-10 flex h-full flex-col justify-between gap-6">
					<div>
						<p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-white/70">Featured</p>
						<h3 className="mt-2 max-w-sm text-xl font-semibold text-white sm:text-2xl">{asset.title}</h3>
					</div>
					<p className="text-sm text-white/75">Posted by {asset.author}</p>
				</div>
			</article>
		);
	}

	return (
		<article className="group relative mb-3 break-inside-avoid overflow-hidden rounded-2xl border border-border/50 bg-secondary/70 shadow-sm transition-transform duration-200 hover:-translate-y-0.5 hover:shadow-md">
			<div className={cn("bg-gradient-to-br", ASPECT_CLASSES[asset.aspect], asset.gradient)} />
			<div className="absolute inset-0 bg-gradient-to-t from-black/75 via-black/15 to-transparent" />
			<div className="absolute inset-x-0 bottom-0 p-3">
				<p className="line-clamp-2 text-sm font-medium text-white">{asset.title}</p>
				<div className="mt-1 flex items-center justify-between text-[11px] text-white/70">
					<span>{asset.author}</span>
					<span className="inline-flex items-center gap-1">
						<Clock3 className="size-3" />
						{asset.durationSec}s
					</span>
				</div>
			</div>
			<div className="absolute inset-0 flex items-center justify-center bg-black/10 opacity-0 transition-opacity group-hover:opacity-100">
				<span className="inline-flex size-10 items-center justify-center rounded-full bg-white/15 text-white backdrop-blur-sm">
					<Play className="size-4 fill-current" />
				</span>
			</div>
		</article>
	);
}

function DiscoveryGrid() {
	const featuredAssets = MOCK_DISCOVERY_ASSETS.filter((asset) => asset.featured);
	const regularAssets = MOCK_DISCOVERY_ASSETS.filter((asset) => !asset.featured);

	return (
		<>
			{featuredAssets.map((asset) => (
				<AssetCard key={asset.id} asset={asset} />
			))}
			<div className="columns-2 gap-3 sm:columns-3 lg:columns-4">
				{regularAssets.map((asset) => (
					<AssetCard key={asset.id} asset={asset} />
				))}
			</div>
		</>
	);
}

export default function DiscoveryFeed() {
	return (
		<section className="mx-auto w-full max-w-5xl px-1 pb-10">
			<Tabs defaultValue={DISCOVERY_TABS[0]}>
				<div className="mb-5 flex flex-col items-center gap-3 sm:flex-row sm:items-end sm:justify-between">
					<div className="text-center sm:text-left">
						<h2 className="text-lg font-semibold text-foreground">Discover</h2>
						<p className="mt-0.5 text-xs text-muted-foreground">Community creations, skills, and templates</p>
					</div>
					<TabsList className="mx-auto h-auto border border-border/50 bg-secondary/80 p-1 sm:mx-0">
						{DISCOVERY_TABS.map((tab) => (
							<TabsTrigger key={tab} value={tab} className="px-3 py-1 text-[11px]">
								{tab}
							</TabsTrigger>
						))}
					</TabsList>
				</div>
				{DISCOVERY_TABS.map((tab) => (
					<TabsContent key={tab} value={tab} className="mt-0">
						<DiscoveryGrid />
					</TabsContent>
				))}
			</Tabs>
		</section>
	);
}
