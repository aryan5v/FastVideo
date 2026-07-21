import { createServer as createHTTPServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const root = dirname(fileURLToPath(import.meta.url));
const publicRoot = join(root, "public");
const manifestPath = join(root, "artifacts.json");
const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml"],
]);

function storageConfiguration(environment) {
  const endpoint = environment.BUCKET_ENDPOINT || environment.AWS_ENDPOINT_URL || environment.ENDPOINT;
  const accessKeyId = environment.BUCKET_ACCESS_KEY_ID || environment.AWS_ACCESS_KEY_ID || environment.ACCESS_KEY_ID;
  const secretAccessKey = environment.BUCKET_SECRET_ACCESS_KEY || environment.AWS_SECRET_ACCESS_KEY || environment.SECRET_ACCESS_KEY;
  const bucket = environment.BUCKET_NAME || environment.AWS_S3_BUCKET_NAME || environment.BUCKET;
  const region = environment.BUCKET_REGION || environment.AWS_DEFAULT_REGION || environment.REGION || "auto";
  if (!endpoint || !accessKeyId || !secretAccessKey || !bucket) return null;
  return { endpoint, accessKeyId, secretAccessKey, bucket, region };
}

async function loadManifest() {
  return JSON.parse(await readFile(manifestPath, "utf8"));
}

function releaseReady(manifest) {
  return Object.values(manifest.artifacts).every((artifact) => artifact.bytes > 0 && artifact.sha256.length === 64);
}

function publicRelease(manifest) {
  return {
    release: manifest.release,
    minimum_macos: manifest.minimum_macos,
    ready: releaseReady(manifest),
    artifacts: Object.fromEntries(Object.entries(manifest.artifacts).map(([id, artifact]) => [id, {
      label: artifact.label,
      filename: artifact.filename,
      bytes: artifact.bytes,
      sha256: artifact.sha256,
      download: `/download/${id}`,
      available: artifact.bytes > 0 && artifact.sha256.length === 64,
    }])),
  };
}

function requestOrigin(request, environment) {
  if (environment.PUBLIC_BASE_URL) return environment.PUBLIC_BASE_URL.replace(/\/$/, "");
  const protocol = request.headers["x-forwarded-proto"] || "http";
  const host = request.headers["x-forwarded-host"] || request.headers.host;
  return `${protocol}://${host}`;
}

function modelCatalog(manifest, origin) {
  const asset = (id) => ({
    url: `${origin}/download/${id}`,
    sha256: manifest.artifacts[id].sha256,
    bytes: manifest.artifacts[id].bytes,
  });
  return {
    catalog_version: 1,
    product: "FastWan QAD 1.3B",
    release: manifest.release,
    shared: asset("shared"),
    fast_mode: asset("fast_mode"),
    variants: { ema: asset("ema"), raw: asset("raw") },
  };
}

function json(response, status, body) {
  response.writeHead(status, {
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
  });
  response.end(JSON.stringify(body));
}

async function staticResponse(response, pathname) {
  const relative = pathname === "/" ? "index.html" : pathname.slice(1);
  if (!/^[a-zA-Z0-9._/-]+$/.test(relative) || relative.includes("..")) return false;
  try {
    const body = await readFile(join(publicRoot, relative));
    response.writeHead(200, {
      "Cache-Control": relative === "index.html" ? "no-cache" : "public, max-age=86400",
      "Content-Type": contentTypes.get(extname(relative)) || "application/octet-stream",
    });
    response.end(body);
    return true;
  } catch {
    return false;
  }
}

export function createServer(environment = process.env) {
  const storage = storageConfiguration(environment);
  const s3 = storage ? new S3Client({
    endpoint: storage.endpoint,
    region: storage.region,
    forcePathStyle: false,
    credentials: { accessKeyId: storage.accessKeyId, secretAccessKey: storage.secretAccessKey },
  }) : null;

  return createHTTPServer(async (request, response) => {
    try {
      const url = new URL(request.url || "/", requestOrigin(request, environment));
      const manifest = await loadManifest();
      if (request.method !== "GET" && request.method !== "HEAD") {
        json(response, 405, { error: "Method not allowed" });
        return;
      }
      if (url.pathname === "/healthz") {
        json(response, 200, { ok: true, storage: Boolean(storage), release: manifest.release, ready: releaseReady(manifest) });
        return;
      }
      if (url.pathname === "/api/release") {
        json(response, 200, publicRelease(manifest));
        return;
      }
      if (url.pathname === "/catalog.json") {
        json(response, 200, modelCatalog(manifest, requestOrigin(request, environment)));
        return;
      }
      if (url.pathname.startsWith("/download/")) {
        const id = url.pathname.slice("/download/".length);
        const artifact = manifest.artifacts[id];
        if (!artifact) {
          json(response, 404, { error: "Unknown download" });
          return;
        }
        if (!artifact.bytes || artifact.sha256.length !== 64) {
          json(response, 503, { error: "This release asset is not published yet." });
          return;
        }
        if (!s3 || !storage) {
          json(response, 503, { error: "Download storage is not configured." });
          return;
        }
        const command = new GetObjectCommand({
          Bucket: storage.bucket,
          Key: artifact.key,
          ResponseContentDisposition: `attachment; filename="${artifact.filename}"`,
          ResponseContentType: artifact.content_type,
        });
        const signed = await getSignedUrl(s3, command, { expiresIn: 900 });
        response.writeHead(302, { "Cache-Control": "no-store", Location: signed });
        response.end();
        return;
      }
      if (await staticResponse(response, url.pathname)) return;
      json(response, 404, { error: "Not found" });
    } catch (error) {
      json(response, 500, { error: error instanceof Error ? error.message : "Unexpected error" });
    }
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 3000);
  createServer().listen(port, "0.0.0.0", () => {
    process.stdout.write(`FastWan downloads listening on ${port}\n`);
  });
}
