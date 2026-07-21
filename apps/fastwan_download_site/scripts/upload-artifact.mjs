import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { S3Client } from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = join(root, "artifacts.json");
const [id, input] = process.argv.slice(2);
if (!id || !input) throw new Error("Usage: npm run upload -- <artifact-id> <file>");

const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const artifact = manifest.artifacts[id];
if (!artifact) throw new Error(`Unknown artifact: ${id}`);
const source = resolve(input);
const info = await stat(source);
const endpoint = process.env.BUCKET_ENDPOINT || process.env.AWS_ENDPOINT_URL || process.env.ENDPOINT;
const accessKeyId = process.env.BUCKET_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID || process.env.ACCESS_KEY_ID;
const secretAccessKey = process.env.BUCKET_SECRET_ACCESS_KEY || process.env.AWS_SECRET_ACCESS_KEY || process.env.SECRET_ACCESS_KEY;
const bucket = process.env.BUCKET_NAME || process.env.AWS_S3_BUCKET_NAME || process.env.BUCKET;
const region = process.env.BUCKET_REGION || process.env.AWS_DEFAULT_REGION || process.env.REGION || "auto";
if (!endpoint || !accessKeyId || !secretAccessKey || !bucket) throw new Error("Railway bucket credentials are missing.");

const hash = createHash("sha256");
for await (const chunk of createReadStream(source)) hash.update(chunk);
const sha256 = hash.digest("hex");
const client = new S3Client({
  endpoint,
  region,
  forcePathStyle: false,
  credentials: { accessKeyId, secretAccessKey },
});
const upload = new Upload({
  client,
  params: {
    Bucket: bucket,
    Key: artifact.key,
    Body: createReadStream(source),
    ContentType: artifact.content_type,
    Metadata: { sha256 },
  },
  queueSize: 3,
  partSize: 64 * 1024 * 1024,
});
upload.on("httpUploadProgress", ({ loaded, total }) => {
  const percent = total ? Math.round((loaded / total) * 100) : 0;
  process.stdout.write(`\rUploading ${id}: ${percent}%`);
});
await upload.done();
process.stdout.write("\n");
artifact.bytes = info.size;
artifact.sha256 = sha256;
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
process.stdout.write(`${id} uploaded: ${info.size} bytes, sha256 ${sha256}\n`);
