// Node port of ../../bundle.py (manifest and simple layouts) used at build time.

import fs from "node:fs";
import path from "node:path";

const VIDEO_SUFFIXES = [".mp4", ".webm", ".mov", ".m4v"];
const SIDE_POLICIES = ["balanced", "random"];
const SITE_TEXT_KEYS = ["title", "heading", "reveal_label", "guidance_label"];
const CHOICE_KEYS = ["left", "right", "tie", "both_bad"];

export class BundleError extends Error {}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (err) {
    throw new BundleError(`cannot read ${file}: ${err.message}`);
  }
}

const isFile = (file) => fs.existsSync(file) && fs.statSync(file).isFile();
const isDir = (dir) => fs.existsSync(dir) && fs.statSync(dir).isDirectory();

function parseArm(raw, source) {
  if (!raw || typeof raw !== "object" || typeof raw.slug !== "string" || !raw.slug) {
    throw new BundleError(`${source}: every arm needs a non-empty string 'slug', got ${JSON.stringify(raw)}`);
  }
  const { slug } = raw;
  if (slug.includes("/") || slug.includes("\\") || slug === "." || slug === "..") {
    throw new BundleError(`${source}: invalid arm slug ${JSON.stringify(slug)}`);
  }
  const speed = raw.speed || {};
  if (typeof speed !== "object" || Array.isArray(speed)) throw new BundleError(`${source}: arm '${slug}' 'speed' must be an object`);
  return { slug, display_name: String(raw.display_name || slug), notes: String(raw.notes || ""), speed: { ...speed } };
}

export function loadArms(root) {
  const file = path.join(root, "arms.json");
  if (!isFile(file)) throw new BundleError(`missing arms.json in ${root}`);
  const data = readJson(file);
  const rawArms = data && typeof data === "object" ? data.arms : null;
  if (!Array.isArray(rawArms) || !rawArms.length) throw new BundleError(`${file}: expected a non-empty 'arms' list`);
  const arms = rawArms.map((raw) => parseArm(raw, file));
  const slugs = arms.map((a) => a.slug);
  if (new Set(slugs).size !== slugs.length) throw new BundleError(`${file}: duplicate arm slugs in ${slugs}`);
  return arms;
}

function manifestClipId(row) {
  const { index, sample_id: sampleId } = row;
  if (Number.isInteger(index) && sampleId) return `${String(index).padStart(3, "0")}_${sampleId}`;
  if (sampleId) return String(sampleId);
  if (Number.isInteger(index)) return String(index).padStart(3, "0");
  throw new BundleError(`manifest row has neither 'index' nor 'sample_id': ${Object.keys(row).sort()}`);
}

function manifestVideo(root, slug, clipId, entry) {
  const candidates = [];
  if (entry && typeof entry === "object" && typeof entry.path === "string") candidates.push(path.join(root, entry.path));
  else if (typeof entry === "string") candidates.push(path.join(root, entry));
  candidates.push(path.join(root, "arms", slug, "videos", `${clipId}.mp4`));
  const found = candidates.find(isFile);
  return found ? fs.realpathSync(found) : null;
}

function loadManifestClips(root, arms) {
  const file = path.join(root, "manifest.jsonl");
  const slugs = arms.map((a) => a.slug).sort();
  const clips = [];
  fs.readFileSync(file, "utf8").split("\n").forEach((line, i) => {
    if (!line.trim()) return;
    let row;
    try {
      row = JSON.parse(line);
    } catch (err) {
      throw new BundleError(`${file}:${i + 1}: invalid JSON: ${err.message}`);
    }
    if (!row || typeof row !== "object" || Array.isArray(row)) throw new BundleError(`${file}:${i + 1}: expected an object`);
    const clipId = manifestClipId(row);
    const entries = row.arms && typeof row.arms === "object" ? row.arms : {};
    const videos = {};
    for (const slug of slugs) {
      const video = manifestVideo(root, slug, clipId, entries[slug]);
      if (video) videos[slug] = video;
      else console.warn(`missing video for arm ${slug} clip ${clipId}`);
    }
    clips.push({ clip_id: clipId, prompt: String(row.prompt || ""), guidance: "", group: "", meta: {}, videos });
  });
  return clips;
}

const isPlainObject = (x) => Boolean(x) && typeof x === "object" && !Array.isArray(x);

/** Port of bundle.parse_prompt_entry: a prompts.json value (string or object) -> clip fields. */
export function parsePromptEntry(raw, clipId, source) {
  if (raw === null || raw === undefined) return { prompt: "", guidance: "", group: "", meta: {} };
  if (typeof raw === "string") return { prompt: raw, guidance: "", group: "", meta: {} };
  if (!isPlainObject(raw)) throw new BundleError(`${source}: entry for '${clipId}' must be a string or an object`);
  const meta = raw.meta || {};
  if (!isPlainObject(meta) || Object.values(meta).some((v) => v !== null && typeof v === "object")) {
    throw new BundleError(`${source}: 'meta' for '${clipId}' must be a flat object`);
  }
  return {
    prompt: String(raw.prompt || ""), guidance: String(raw.guidance || ""), group: String(raw.group || ""),
    meta: { ...meta },
  };
}

/** Port of bundle.load_site: the optional site.json (page text and side policy). */
export function loadSite(root) {
  const file = path.join(root, "site.json");
  if (!isFile(file)) return {};
  const data = readJson(file);
  if (!isPlainObject(data)) throw new BundleError(`${file}: expected an object`);
  if (!SIDE_POLICIES.includes(data.sides ?? "balanced")) throw new BundleError(`${file}: 'sides' must be one of ${SIDE_POLICIES}`);
  const intro = data.intro ?? [];
  if (!Array.isArray(intro) || !intro.every((x) => typeof x === "string")) {
    throw new BundleError(`${file}: 'intro' must be a list of paragraph strings`);
  }
  const choices = data.choices ?? {};
  if (!isPlainObject(choices) || !Object.keys(choices).every((k) => CHOICE_KEYS.includes(k))) {
    throw new BundleError(`${file}: 'choices' may only relabel ${CHOICE_KEYS}`);
  }
  for (const key of SITE_TEXT_KEYS) {
    if (key in data && typeof data[key] !== "string") throw new BundleError(`${file}: '${key}' must be a string`);
  }
  return { ...data };
}

/** Port of bundle.select_arms: keep only the named arms (arms.json order); empty keeps all. */
export function selectArms(arms, only) {
  if (!only || !only.length) return arms;
  const unknown = only.filter((slug) => !arms.some((a) => a.slug === slug)).sort();
  if (unknown.length) throw new BundleError(`unknown arm(s) ${unknown}; arms.json has ${arms.map((a) => a.slug)}`);
  const kept = arms.filter((a) => only.includes(a.slug));
  if (kept.length < 2) throw new BundleError(`need at least two arms, got ${kept.map((a) => a.slug)}`);
  return kept;
}

function loadSimpleClips(root, arms) {
  const promptsFile = path.join(root, "prompts.json");
  const prompts = isFile(promptsFile) ? readJson(promptsFile) : {};
  if (!prompts || typeof prompts !== "object" || Array.isArray(prompts)) {
    throw new BundleError(`${promptsFile}: expected an object mapping clip_id -> prompt`);
  }
  const byClip = new Map();
  for (const arm of arms) {
    const armDir = path.join(root, "arms", arm.slug);
    if (!isDir(armDir)) {
      console.warn(`arm directory missing: ${armDir}`);
      continue;
    }
    for (const name of fs.readdirSync(armDir).sort()) {
      const full = path.join(armDir, name);
      const ext = path.extname(name);
      if (!VIDEO_SUFFIXES.includes(ext.toLowerCase()) || !isFile(full)) continue;
      const stem = name.slice(0, -ext.length);
      if (!byClip.has(stem)) byClip.set(stem, {});
      byClip.get(stem)[arm.slug] = fs.realpathSync(full);
    }
  }
  return [...byClip.keys()].sort().map((clipId) => ({
    clip_id: clipId, ...parsePromptEntry(prompts[clipId], clipId, promptsFile), videos: byClip.get(clipId),
  }));
}

/**
 * Parse a bundle directory, optionally restricted to the arms in `onlyArms`. Returns {root, layout, arms, clips,
 * site} with clips[].videos = slug -> absolute path.
 */
export function loadBundle(dir, onlyArms = null) {
  const root = fs.realpathSync(path.resolve(dir));
  if (!isDir(root)) throw new BundleError(`bundle directory not found: ${root}`);
  const arms = selectArms(loadArms(root), onlyArms);
  const manifest = isFile(path.join(root, "manifest.jsonl"));
  const clips = manifest ? loadManifestClips(root, arms) : loadSimpleClips(root, arms);
  const usable = clips.filter((c) => Object.keys(c.videos).length >= 2);
  if (!usable.length) throw new BundleError(`${root}: no clip has videos for at least two arms`);
  if (usable.length < clips.length) console.warn(`skipping ${clips.length - usable.length} clips with fewer than two arms`);
  return { root, layout: manifest ? "manifest" : "simple", arms, clips: usable, site: loadSite(root) };
}
