import sourceCatalog from "../../docs/assets/cookbook-recipes.json" with { type: "json" };

/** Presentation metadata only. Deployments and editable fields come from the cookbook backend. */
export const recipePages = [
  {
    id: "fasth3-8step",
    family: "FastH3",
    title: "FastH3 8-Step V2",
    mark: "H3",
    path: "/cookbook/minimax-h3/fasth3-8step",
    description: "Distilled MiniMax-H3 video generation with synchronized audio.",
    tags: ["Text to video", "Audio", "8-step"],
    featured: true,
  },
  {
    id: "fasth3-trim",
    family: "FastH3",
    title: "FastH3 Trim",
    mark: "H3",
    path: "/cookbook/minimax-h3/fasth3-trim",
    description: "The smaller FastH3 model for eight-step video generation with synchronized audio.",
    tags: ["Text to video", "Audio", "NVFP4", "8-step"],
  },
  {
    id: "fastwan21",
    family: "Wan",
    title: "FastWan2.1 1.3B",
    mark: "Wan",
    path: "/cookbook/wan/fastwan21",
    description: "A compact, distilled Wan model for text-to-video generation.",
    tags: ["Text to video", "1.3B", "3-step"],
  },
  {
    id: "wan21-i2v",
    family: "Wan",
    title: "Wan2.1 I2V 14B",
    mark: "Wan",
    path: "/cookbook/wan/wan21-i2v",
    description: "Turn a reference image into video with Wan2.1.",
    tags: ["Image to video", "14B", "480p"],
  },
  {
    id: "ltx2-distilled",
    family: "LTX",
    title: "LTX2 Distilled",
    mark: "LTX",
    path: "/cookbook/ltx/ltx2-distilled",
    description: "Generate video with incremental delivery over WebSocket.",
    tags: ["Text to video", "Streaming"],
  },
  {
    id: "cosmos25-t2v",
    family: "Cosmos",
    title: "Cosmos Predict2.5 2B T2V",
    mark: "Cosmos",
    path: "/cookbook/cosmos/cosmos25-t2v",
    description: "Text-to-video generation with the Cosmos Predict2.5 2B Diffusers checkpoint.",
    tags: ["Text to video"],
  },
  {
    id: "flux1-dev",
    family: "FLUX",
    title: "FLUX.1-dev",
    mark: "FLUX",
    path: "/cookbook/flux/flux1-dev",
    description: "Text-to-image generation with FLUX.1-dev through the OpenAI-compatible image routes.",
    tags: ["Text to image"],
  },
  {
    id: "flux2-dev",
    family: "FLUX",
    title: "FLUX.2-dev",
    mark: "FLUX",
    path: "/cookbook/flux/flux2-dev",
    description: "Text-to-image generation with FLUX.2-dev through the OpenAI-compatible image routes.",
    tags: ["Text to image"],
  },
  {
    id: "flux2-klein",
    family: "FLUX",
    title: "FLUX.2 Klein 4B",
    mark: "FLUX",
    path: "/cookbook/flux/flux2-klein",
    description: "Text-to-image generation with the smaller FLUX.2-klein-4B checkpoint.",
    tags: ["Text to image"],
  },
  {
    id: "hunyuan15-i2v-480p",
    family: "Hunyuan",
    title: "HunyuanVideo 1.5 I2V 480p",
    mark: "HY",
    path: "/cookbook/hunyuan/hunyuan15-i2v-480p",
    description: "First-frame-conditioned video generation with the step-distilled HunyuanVideo 1.5 model.",
    tags: ["Image to video"],
  },
  {
    id: "hunyuan15-t2v-480p",
    family: "Hunyuan",
    title: "HunyuanVideo 1.5 T2V 480p",
    mark: "HY",
    path: "/cookbook/hunyuan/hunyuan15-t2v-480p",
    description: "Text-to-video generation at 480p with HunyuanVideo 1.5.",
    tags: ["Text to video"],
  },
  {
    id: "kandinsky5-t2v",
    family: "Kandinsky 5",
    title: "Kandinsky 5 T2V Lite",
    mark: "K5",
    path: "/cookbook/kandinsky5/kandinsky5-t2v",
    description: "Text-to-video generation with the Kandinsky-5.0 Lite 5-second model.",
    tags: ["Text to video"],
  },
  {
    id: "kandinsky6-ti2va-distilled",
    family: "Kandinsky 6",
    title: "Kandinsky 6 TI2VA Distilled",
    mark: "K6",
    path: "/cookbook/kandinsky6/kandinsky6-ti2va-distilled",
    description: "Ten-step pi-Flow Kandinsky 6 Pro video with synchronized audio.",
    tags: ["Text to video", "Audio"],
  },
  {
    id: "kandinsky6-ti2va",
    family: "Kandinsky 6",
    title: "Kandinsky 6 TI2VA",
    mark: "K6",
    path: "/cookbook/kandinsky6/kandinsky6-ti2va",
    description: "Text- or image-conditioned video with synchronized audio from Kandinsky 6 Pro.",
    tags: ["Text to video", "Audio"],
  },
  {
    id: "mmaudio",
    family: "MMAudio",
    title: "MMAudio large-44k-v2",
    mark: "Audio",
    path: "/cookbook/mmaudio/mmaudio",
    description: "Video-to-audio generation through the OpenAI-compatible media-job routes.",
    tags: ["Video to audio"],
  },
  {
    id: "sd35",
    family: "Stable Diffusion",
    title: "SD3.5 Medium",
    mark: "SD",
    path: "/cookbook/stable-diffusion/sd35",
    description:
      "Text-to-image generation with Stable Diffusion 3.5 Medium through the OpenAI-compatible image routes.",
    tags: ["Text to image"],
  },
  {
    id: "stable-audio",
    family: "Stable Audio",
    title: "Stable Audio Open 1.0",
    mark: "Audio",
    path: "/cookbook/stable-audio/stable-audio",
    description: "Text-to-audio generation through the OpenAI-compatible media-job routes.",
    tags: ["Text to audio"],
  },
  {
    id: "wan21-t2v-14b",
    family: "Wan",
    title: "Wan2.1 T2V 14B",
    mark: "Wan",
    path: "/cookbook/wan/wan21-t2v-14b",
    description: "Text-to-video generation with the Wan2.1 14B base model at 720p.",
    tags: ["Text to video"],
  },
  {
    id: "wan22-i2v-a14b",
    family: "Wan",
    title: "Wan2.2 I2V A14B",
    mark: "Wan",
    path: "/cookbook/wan/wan22-i2v-a14b",
    description: "Image-conditioned video generation with the Wan2.2 mixture-of-experts model.",
    tags: ["Image to video"],
  },
];
export type RecipePage = (typeof recipePages)[number];
export function sitePath(path: string) {
  return `${import.meta.env.BASE_URL.replace(/\/$/, "")}${path}`;
}

export interface SourceRecipe {
  id: string;
  family: string;
  task: string;
  label: string;
  summary: string;
  model: string;
  source: string;
  command: string;
  gpu_types: string[];
  hardware?: { gpu_count?: number; platform?: string; accelerator?: string };
  serving?: { source: string; install: string; env?: string; prepare?: string };
  knobs?: {
    key: string;
    label: string;
    hint?: string;
    flag: string;
    default: string | number;
    options: (string | number | { value: string | number; label: string })[];
  }[];
}
export const sourceRecipes = sourceCatalog.recipes as SourceRecipe[];
const familyInfo: Record<string, { title: string; path: string; logo: string; category: string }> = {
  minimax_h3: { title: "MiniMax H3", path: "minimax-h3", logo: "minimax", category: "Video" },
  wan: { title: "Wan", path: "wan", logo: "wan-ai", category: "Video" },
  ltx2: { title: "LTX", path: "ltx", logo: "ltx", category: "Video" },
  hunyuan: { title: "Hunyuan", path: "hunyuan", logo: "tencent-hunyuan", category: "Video" },
  kandinsky5: { title: "Kandinsky 5", path: "kandinsky5", logo: "kandinsky", category: "Video" },
  kandinsky6: { title: "Kandinsky 6", path: "kandinsky6", logo: "kandinsky", category: "Video" },
  flux: { title: "FLUX", path: "flux", logo: "black-forest-labs", category: "Image" },
  sd35: { title: "Stable Diffusion", path: "stable-diffusion", logo: "stabilityai", category: "Image" },
  stable_audio: { title: "Stable Audio", path: "stable-audio", logo: "stabilityai", category: "Audio" },
  mmaudio: { title: "MMAudio", path: "mmaudio", logo: "fastvideo", category: "Audio" },
  cosmos: { title: "Cosmos", path: "cosmos", logo: "nvidia", category: "World" },
  matrixgame: { title: "Matrix Game", path: "matrix-game", logo: "fastvideo", category: "World" },
};
const backendModels: Record<string, string> = {
  "FastVideo/FastVideo-FastH3-8-Step-V2": "fasth3-8step",
  "FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4": "fasth3-trim",
  "FastVideo/FastWan2.1-T2V-1.3B-Diffusers": "fastwan21",
  "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": "wan21-i2v",
  "FastVideo/LTX2-Distilled-Diffusers": "ltx2-distilled",
  "KyleShao/Cosmos-Predict2.5-2B-Diffusers": "cosmos25-t2v",
  "black-forest-labs/FLUX.1-dev": "flux1-dev",
  "black-forest-labs/FLUX.2-dev": "flux2-dev",
  "black-forest-labs/FLUX.2-klein-4B": "flux2-klein",
  "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled": "hunyuan15-i2v-480p",
  "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": "hunyuan15-t2v-480p",
  "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers": "kandinsky5-t2v",
  "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers": "kandinsky6-ti2va-distilled",
  "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers": "kandinsky6-ti2va",
  "FastVideo/MMAudio-large-44k-v2-Diffusers": "mmaudio",
  "stabilityai/stable-diffusion-3.5-medium": "sd35",
  "FastVideo/stable-audio-open-1.0-Diffusers": "stable-audio",
  "Wan-AI/Wan2.1-T2V-14B-Diffusers": "wan21-t2v-14b",
  "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "wan22-i2v-a14b",
};
const TASK_GROUPS: Array<[RegExp, string]> = [
  [/world/, "Text to world"],
  [/video\/text to audio|video.*to audio/, "Video to audio"],
  [/text to audio/, "Text to audio"],
  [/text to image/, "Text to image"],
  [/reference media to video/, "Reference to video"],
  [/first\/last frame to video/, "Image to video"],
  [/text or image to video/, "Text/image to video"],
  [/image to video/, "Image to video"],
  [/super-resolution/, "Video upscale"],
  [/video/, "Text to video"],
];
/** A small discovery vocabulary derived from the recipe task, not a capability claim. */
export function taskLabel(value: string): string {
  const normalized = value.toLowerCase();
  return TASK_GROUPS.find(([pattern]) => pattern.test(normalized))?.[1] ?? "Text to video";
}

const directoryModels = [...new Set(sourceRecipes.map((recipe) => recipe.model))].map((model) => {
  const recipes = sourceRecipes.filter((recipe) => recipe.model === model);
  const first = recipes[0];
  const family = familyInfo[first.family];
  const backend = recipePages.find((page) => page.id === backendModels[model]);
  const ownModel = model.startsWith("FastVideo/Fast");
  const logo = ownModel ? "fastvideo" : family.logo;
  const providers: Record<string, string> = {
    fastvideo: "FastVideo",
    minimax: "MiniMax",
    "wan-ai": "Wan AI",
    ltx: "Lightricks",
    "tencent-hunyuan": "Tencent",
    kandinsky: "Kandinsky Lab",
    "black-forest-labs": "Black Forest Labs",
    stabilityai: "Stability AI",
    nvidia: "NVIDIA",
  };
  return {
    id: backend?.id ?? first.id,
    model,
    recipes,
    title:
      backend?.title ??
      (model.includes("FastH3-4-step")
        ? "FastH3 4-Step V1"
        : model.includes("FastH3-Preview-v0.2")
          ? "FastH3 Preview V0.2"
          : model === "MiniMaxAI/MiniMax-H3"
            ? "MiniMax H3"
            : first.label.split(" on ")[0]),
    family: first.family === "minimax_h3" ? "FastH3" : (backend?.family ?? family.title),
    familyPath: `/cookbook/${family.path}`,
    path: backend?.path ?? `/cookbook/${family.path}/models/${first.id}`,
    logo,
    provider: providers[logo],
    category: family.category,
    description: backend?.description ?? first.summary,
    tags: backend?.tags ?? [...new Set(recipes.map((recipe) => recipe.task))],
    task: taskLabel(backend?.tags[0] ?? first.task),
    platforms: [...new Set(recipes.flatMap((recipe) => recipe.gpu_types))],
    configurable: !!backend,
    featured: backend?.featured ?? false,
  };
});
const pageFamily: Record<string, string> = {
  "cosmos25-t2v": "cosmos",
  "flux1-dev": "flux",
  "flux2-dev": "flux",
  "flux2-klein": "flux",
  "hunyuan15-i2v-480p": "hunyuan",
  "hunyuan15-t2v-480p": "hunyuan",
  "kandinsky5-t2v": "kandinsky5",
  "kandinsky6-ti2va-distilled": "kandinsky6",
  "kandinsky6-ti2va": "kandinsky6",
  mmaudio: "mmaudio",
  sd35: "sd35",
  "stable-audio": "stable_audio",
  "wan21-t2v-14b": "wan",
  "wan22-i2v-a14b": "wan",
};
const directoryProviders: Record<string, string> = {
  fastvideo: "FastVideo",
  minimax: "MiniMax",
  "wan-ai": "Wan AI",
  ltx: "Lightricks",
  "tencent-hunyuan": "Tencent",
  kandinsky: "Kandinsky Lab",
  "black-forest-labs": "Black Forest Labs",
  stabilityai: "Stability AI",
  nvidia: "NVIDIA",
};
const pageOnlyModels = recipePages
  .filter((page) => !directoryModels.some((entry) => entry.id === page.id))
  .map((page) => {
    const model = Object.entries(backendModels).find(([, id]) => id === page.id)?.[0] ?? page.id;
    const family = familyInfo[pageFamily[page.id]] ?? familyInfo.wan;
    const logo = model.startsWith("FastVideo/Fast") ? "fastvideo" : family.logo;
    return {
      id: page.id,
      model,
      recipes: [],
      title: page.title,
      family: page.family,
      familyPath: `/cookbook/${family.path}`,
      path: page.path,
      logo,
      provider: directoryProviders[logo] ?? "FastVideo",
      category: family.category,
      description: page.description,
      tags: page.tags,
      task: taskLabel(page.tags[0]),
      platforms: ["NVIDIA"],
      configurable: true,
      featured: page.featured ?? false,
    };
  });
export const modelDirectory = [...directoryModels, ...pageOnlyModels].sort(
  (a, b) =>
    Number(b.featured) - Number(a.featured) ||
    Number(b.family === "FastH3") - Number(a.family === "FastH3") ||
    Number(b.configurable) - Number(a.configurable),
);
export type DirectoryModel = (typeof modelDirectory)[number];
