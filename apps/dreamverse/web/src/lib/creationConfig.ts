export type CreationModeId = "t2v" | "fl2av" | "ref2av";

export type CreationModelId = "fast-ltx2" | "fast-ltx23" | "fast-h3";

export type AspectRatioId = "21:9" | "16:9" | "4:3" | "1:1" | "3:4" | "9:16";

export type ResolutionId = "720p" | "1080p" | "4k";

export interface CreationModeOption {
	id: CreationModeId;
	label: string;
	description: string;
}

export interface CreationModelOption {
	id: CreationModelId;
	label: string;
	description: string;
	badge?: string;
}

export interface MentionOption {
	id: string;
	label: string;
	kind: "preset" | "asset" | "character";
	description?: string;
}

export interface DiscoveryAsset {
	id: string;
	title: string;
	author: string;
	aspect: AspectRatioId;
	durationSec: number;
	gradient: string;
	featured?: boolean;
}

export const CREATION_MODES: CreationModeOption[] = [
	{ id: "t2v", label: "Text to video", description: "Generate from a text prompt" },
	{ id: "fl2av", label: "First & last frame", description: "Animate between two keyframes" },
	{ id: "ref2av", label: "Reference guided", description: "Use a reference image or clip" },
];

export const CREATION_MODELS: CreationModelOption[] = [
	{
		id: "fast-ltx23",
		label: "FastLTX 2.3",
		description: "LTX 2.3 with OmniNFT LoRA",
		badge: "New",
	},
	{
		id: "fast-ltx2",
		label: "FastLTX 2",
		description: "FastLTX 2 for streaming",
	},
	{
		id: "fast-h3",
		label: "FastH3 Preview",
		description: "H3 audio-video segments",
		badge: "New",
	},
];

export const ASPECT_RATIOS: AspectRatioId[] = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];

export const RESOLUTIONS: ResolutionId[] = ["720p", "1080p", "4k"];

export const DURATION_MARKS = [5, 10, 15] as const;

export const DISCOVERY_TABS = ["Trends", "Skills", "AI Shorts", "Events"] as const;

export const MOCK_DISCOVERY_ASSETS: DiscoveryAsset[] = [
	{
		id: "featured-program",
		title: "Partner program",
		author: "Dreamverse",
		aspect: "16:9",
		durationSec: 0,
		gradient: "from-sky-500 via-indigo-500 to-violet-600",
		featured: true,
	},
	{
		id: "neon-city",
		title: "Neon rain over downtown",
		author: "studio-07",
		aspect: "9:16",
		durationSec: 5,
		gradient: "from-cyan-500/70 via-blue-700/70 to-slate-900",
	},
	{
		id: "snack-cascade",
		title: "Snack cascade slow motion",
		author: "foodlab",
		aspect: "1:1",
		durationSec: 5,
		gradient: "from-amber-400/70 via-orange-500/70 to-rose-700/70",
	},
	{
		id: "poolside",
		title: "Poolside golden hour",
		author: "lumen",
		aspect: "3:4",
		durationSec: 10,
		gradient: "from-teal-400/70 via-emerald-500/70 to-cyan-900/70",
	},
	{
		id: "retro-console",
		title: "Retro console glow",
		author: "pixelwave",
		aspect: "16:9",
		durationSec: 5,
		gradient: "from-fuchsia-500/70 via-purple-600/70 to-indigo-900/70",
	},
	{
		id: "paper-cut",
		title: "Paper cut city timelapse",
		author: "craftroom",
		aspect: "4:3",
		durationSec: 15,
		gradient: "from-rose-300/70 via-orange-300/70 to-amber-700/70",
	},
];

export function formatResolutionLabel(resolution: ResolutionId): string {
	return resolution === "4k" ? "4K" : resolution.toUpperCase();
}

export function formatDurationLabel(seconds: number): string {
	return `${seconds}s`;
}

export function modeRequiresReference(modeId: CreationModeId): boolean {
	return modeId === "ref2av";
}

export function modeUsesDualFrames(modeId: CreationModeId): boolean {
	return modeId === "fl2av";
}

export function buildMentionOptions(storyPresets: Array<{ id?: string; label?: string; description?: string }>): MentionOption[] {
	return storyPresets
		.filter((preset) => typeof preset.label === "string" && preset.label.trim())
		.map((preset) => ({
			id: String(preset.id || preset.label),
			label: String(preset.label),
			kind: "preset" as const,
			description: typeof preset.description === "string" ? preset.description : undefined,
		}));
}
