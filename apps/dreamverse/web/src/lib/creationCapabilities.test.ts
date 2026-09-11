import { describe, expect, it } from "vitest";

import {
	DEFAULT_LOBBY_CREATION_CAPABILITIES,
	clampLobbySelectionToCapabilities,
	parseLobbyCreationCapabilities,
	validateLobbyCreationSelection,
} from "@/lib/creationCapabilities";

describe("creationCapabilities", () => {
	it("parses backend capability payloads", () => {
		expect(
			parseLobbyCreationCapabilities({
				model_ids: ["fast-ltx2"],
				generation_modes: ["t2va"],
				resolutions: ["480p", "720p"],
				duration_sec: [5, 10],
			}),
		).toMatchObject({
			model_ids: ["fast-ltx2"],
			generation_modes: ["t2va"],
			resolutions: ["480p", "720p"],
			duration_sec: [5, 10],
		});
	});

	it("clamps unsupported lobby selections to supported defaults", () => {
		expect(
			clampLobbySelectionToCapabilities({
				capabilities: DEFAULT_LOBBY_CREATION_CAPABILITIES,
				modelId: "fast-ltx23",
				modeId: "fl2av",
				aspectRatio: "16:9",
				resolution: "4k",
				durationSec: 99,
			}),
		).toEqual({
			modelId: "fast-ltx23",
			modeId: "t2v",
			aspectRatio: "16:9",
			resolution: "480p",
			durationSec: 5,
		});
	});

	it("rejects unsupported generation modes with a clear message", () => {
		expect(
			validateLobbyCreationSelection({
				capabilities: DEFAULT_LOBBY_CREATION_CAPABILITIES,
				modelId: "fast-ltx23",
				modeId: "fl2av",
				aspectRatio: "16:9",
				resolution: "720p",
				durationSec: 5,
			}),
		).toMatch(/FL2VA/i);
	});

	it("rejects unsupported resolutions", () => {
		expect(
			validateLobbyCreationSelection({
				capabilities: DEFAULT_LOBBY_CREATION_CAPABILITIES,
				modelId: "fast-ltx23",
				modeId: "t2v",
				aspectRatio: "16:9",
				resolution: "4k",
				durationSec: 5,
			}),
		).toMatch(/resolution/i);
	});
});
