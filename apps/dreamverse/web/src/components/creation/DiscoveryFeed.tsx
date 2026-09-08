"use client";

import React from "react";
import { Clock3, Play } from "lucide-react";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DISCOVERY_TABS, MOCK_DISCOVERY_ASSETS, type DiscoveryAsset } from "@/lib/creationConfig";
import { cn } from "@/lib/utils";

function AssetCard({ asset }: { asset: DiscoveryAsset }) {
	if (asset.featured) {
		return (
			<article className="relative col-span-2 overflow-hidden rounded-3xl border border-border/70 bg-card/70 p-5 shadow-sm sm:min-h-[220px]">
				<div className={cn("absolute inset-0 bg-gradient-to-br opacity-90", asset.gradient)} />
				<div className="relative z-10 flex h-full flex-col justify-between gap-6">
					<div>
						<p className="text-xs font-semibold uppercase tracking-[0.2em] text-white/70">Featured</p>
						<h3 className="mt-2 max-w-sm text-2xl font-semibold text-white">{asset.title}</h3>
					</div>
					<p className="text-sm text-white/80">Posted by {asset.author}</p>
				</div>
			</article>
		);
	}

	return (
		<article className="group relative overflow-hidden rounded-3xl border border-border/70 bg-card/70 shadow-sm">
			<div className={cn("aspect-[4/5] bg-gradient-to-br", asset.gradient)} />
			<div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/10 to-transparent" />
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
			<div className="absolute inset-0 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100">
				<span className="inline-flex size-10 items-center justify-center rounded-full bg-white/15 text-white backdrop-blur-sm">
					<Play className="size-4 fill-current" />
				</span>
			</div>
		</article>
	);
}

export default function DiscoveryFeed() {
	return (
		<section className="mx-auto w-full max-w-5xl px-1 pb-8">
			<Tabs defaultValue={DISCOVERY_TABS[0]}>
				<TabsList className="mx-auto">
					{DISCOVERY_TABS.map((tab) => (
						<TabsTrigger key={tab} value={tab}>
							{tab}
						</TabsTrigger>
					))}
				</TabsList>
				{DISCOVERY_TABS.map((tab) => (
					<TabsContent key={tab} value={tab}>
						<div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
							{MOCK_DISCOVERY_ASSETS.map((asset) => (
								<AssetCard key={`${tab}-${asset.id}`} asset={asset} />
							))}
						</div>
					</TabsContent>
				))}
			</Tabs>
		</section>
	);
}
