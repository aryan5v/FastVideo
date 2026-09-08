import { describe, expect, it } from "vitest";

import { buildMentionOptions, formatDurationLabel, formatResolutionLabel } from "@/lib/creationConfig";

describe("creationConfig", () => {
	it("formats resolution labels", () => {
		expect(formatResolutionLabel("720p")).toBe("720P");
		expect(formatResolutionLabel("4k")).toBe("4K");
	});

	it("formats duration labels", () => {
		expect(formatDurationLabel(5)).toBe("5s");
	});

	it("builds mention options from presets", () => {
		expect(
			buildMentionOptions([
				{ id: "preset-a", label: "Preset A", description: "A short preset" },
				{ label: "Missing id" },
			]),
		).toEqual([
			{
				id: "preset-a",
				label: "Preset A",
				kind: "preset",
				description: "A short preset",
			},
			{
				id: "Missing id",
				label: "Missing id",
				kind: "preset",
				description: undefined,
			},
		]);
	});
});
