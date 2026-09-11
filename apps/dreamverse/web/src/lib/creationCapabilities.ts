import type {
	AspectRatioId,
	CreationModeId,
	CreationModelId,
	ResolutionId,
} from "@/lib/creationConfig";
import { fromGenerationMode, toGenerationMode, type GenerationMode } from "@/lib/generationMode";

export interface LobbyCreationCapabilities {
	model_ids: CreationModelId[];
	generation_modes: GenerationMode[];
	aspect_ratios: AspectRatioId[];
	resolutions: ResolutionId[];
	duration_sec: number[];
	unsupported_generation_modes: Record<string, string>;
	reference_assets: {
		mime_types: string[];
		max_bytes: number;
	};
}

export const DEFAULT_LOBBY_CREATION_CAPABILITIES: LobbyCreationCapabilities = {
	model_ids: ["fast-ltx23", "fast-ltx2"],
	generation_modes: ["t2va", "ref2va"],
	aspect_ratios: ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
	resolutions: ["480p", "720p", "1080p"],
	duration_sec: [5, 10, 15],
	unsupported_generation_modes: {
		fl2va: "First/last frame mode (FL2VA) is not supported on FastLTX models yet.",
	},
	reference_assets: {
		mime_types: ["image/png", "image/jpeg", "image/webp"],
		max_bytes: 15 * 1024 * 1024,
	},
};

export function parseLobbyCreationCapabilities(payload: unknown): LobbyCreationCapabilities {
	if (!payload || typeof payload !== "object") {
		return DEFAULT_LOBBY_CREATION_CAPABILITIES;
	}
	const data = payload as Record<string, unknown>;
	const pickStrings = <T extends string>(value: unknown, allowed: readonly T[], fallback: readonly T[]): T[] => {
		if (!Array.isArray(value)) return [...fallback];
		return value.filter((item): item is T => typeof item === "string" && allowed.includes(item as T));
	};
	return {
		model_ids: pickStrings(data.model_ids, ["fast-ltx2", "fast-ltx23"], DEFAULT_LOBBY_CREATION_CAPABILITIES.model_ids),
		generation_modes: pickStrings(
			data.generation_modes,
			["t2va", "fl2va", "ref2va"],
			DEFAULT_LOBBY_CREATION_CAPABILITIES.generation_modes,
		),
		aspect_ratios: pickStrings(
			data.aspect_ratios,
			["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
			DEFAULT_LOBBY_CREATION_CAPABILITIES.aspect_ratios,
		),
		resolutions: pickStrings(
			data.resolutions,
			["480p", "720p", "1080p", "4k"],
			DEFAULT_LOBBY_CREATION_CAPABILITIES.resolutions,
		),
		duration_sec: Array.isArray(data.duration_sec)
			? data.duration_sec.filter((item): item is number => typeof item === "number")
			: DEFAULT_LOBBY_CREATION_CAPABILITIES.duration_sec,
		unsupported_generation_modes:
			typeof data.unsupported_generation_modes === "object" && data.unsupported_generation_modes
				? (data.unsupported_generation_modes as Record<string, string>)
				: DEFAULT_LOBBY_CREATION_CAPABILITIES.unsupported_generation_modes,
		reference_assets:
			typeof data.reference_assets === "object" && data.reference_assets
				? {
						mime_types: Array.isArray((data.reference_assets as Record<string, unknown>).mime_types)
							? ((data.reference_assets as Record<string, unknown>).mime_types as string[])
							: DEFAULT_LOBBY_CREATION_CAPABILITIES.reference_assets.mime_types,
						max_bytes:
							typeof (data.reference_assets as Record<string, unknown>).max_bytes === "number"
								? ((data.reference_assets as Record<string, unknown>).max_bytes as number)
								: DEFAULT_LOBBY_CREATION_CAPABILITIES.reference_assets.max_bytes,
					}
				: DEFAULT_LOBBY_CREATION_CAPABILITIES.reference_assets,
	};
}

export function supportedCreationModes(capabilities: LobbyCreationCapabilities) {
	return capabilities.generation_modes.map((wireMode) => ({
		wireMode,
		modeId: fromGenerationMode(wireMode),
	}));
}

export function isSupportedCreationMode(modeId: CreationModeId, capabilities: LobbyCreationCapabilities): boolean {
	return capabilities.generation_modes.includes(toGenerationMode(modeId));
}

export function isSupportedResolution(resolution: ResolutionId, capabilities: LobbyCreationCapabilities): boolean {
	return capabilities.resolutions.includes(resolution);
}

export function isSupportedReferenceImage(file: File, capabilities: LobbyCreationCapabilities): boolean {
	return capabilities.reference_assets.mime_types.includes(file.type);
}

export function unsupportedModeNotice(modeId: CreationModeId, capabilities: LobbyCreationCapabilities): string | null {
	const wireMode = toGenerationMode(modeId);
	return capabilities.unsupported_generation_modes[wireMode] ?? null;
}

export function clampLobbySelectionToCapabilities(input: {
	capabilities: LobbyCreationCapabilities;
	modelId: CreationModelId;
	modeId: CreationModeId;
	aspectRatio: AspectRatioId;
	resolution: ResolutionId;
	durationSec: number;
}): {
	modelId: CreationModelId;
	modeId: CreationModeId;
	aspectRatio: AspectRatioId;
	resolution: ResolutionId;
	durationSec: number;
} {
	const { capabilities } = input;
	const modelId = capabilities.model_ids.includes(input.modelId)
		? input.modelId
		: (capabilities.model_ids[0] ?? "fast-ltx23");
	const supportedModes = supportedCreationModes(capabilities);
	const modeId = isSupportedCreationMode(input.modeId, capabilities)
		? input.modeId
		: (supportedModes[0]?.modeId ?? "t2v");
	const aspectRatio = capabilities.aspect_ratios.includes(input.aspectRatio)
		? input.aspectRatio
		: (capabilities.aspect_ratios[0] ?? "16:9");
	const resolution = isSupportedResolution(input.resolution, capabilities)
		? input.resolution
		: (capabilities.resolutions[0] ?? "720p");
	const durationSec = capabilities.duration_sec.includes(input.durationSec)
		? input.durationSec
		: (capabilities.duration_sec[0] ?? 5);
	return { modelId, modeId, aspectRatio, resolution, durationSec };
}

export function validateLobbyCreationSelection(input: {
	capabilities: LobbyCreationCapabilities;
	modelId: CreationModelId;
	modeId: CreationModeId;
	aspectRatio: AspectRatioId;
	resolution: ResolutionId;
	durationSec: number;
	referenceFile?: File | null;
	firstFrameFile?: File | null;
	lastFrameFile?: File | null;
}): string | null {
	const unsupportedMode = unsupportedModeNotice(input.modeId, input.capabilities);
	if (unsupportedMode) return unsupportedMode;
	if (!input.capabilities.model_ids.includes(input.modelId)) {
		return "Selected model is not supported yet.";
	}
	if (!isSupportedCreationMode(input.modeId, input.capabilities)) {
		return "Selected mode is not supported yet.";
	}
	if (!input.capabilities.aspect_ratios.includes(input.aspectRatio)) {
		return "Selected aspect ratio is not supported yet.";
	}
	if (!isSupportedResolution(input.resolution, input.capabilities)) {
		return "Selected resolution is not supported on FastLTX models yet.";
	}
	if (!input.capabilities.duration_sec.includes(input.durationSec)) {
		return "Selected duration is not supported yet.";
	}
	if (input.modeId === "ref2av" && !input.referenceFile) {
		return "Upload a reference image to use reference-guided mode.";
	}
	if (input.referenceFile && !isSupportedReferenceImage(input.referenceFile, input.capabilities)) {
		return "Reference assets must be PNG, JPEG, or WebP images.";
	}
	if (input.firstFrameFile && !isSupportedReferenceImage(input.firstFrameFile, input.capabilities)) {
		return "First frame must be a PNG, JPEG, or WebP image.";
	}
	if (input.lastFrameFile && !isSupportedReferenceImage(input.lastFrameFile, input.capabilities)) {
		return "Last frame must be a PNG, JPEG, or WebP image.";
	}
	return null;
}
