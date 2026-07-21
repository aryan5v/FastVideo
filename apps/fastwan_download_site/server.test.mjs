import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer } from "./server.mjs";

let server;
let origin;

before(async () => {
  server = createServer({ PUBLIC_BASE_URL: "https://fastwan.example" });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});

test("landing page and health endpoint respond", async () => {
  const landing = await fetch(origin);
  assert.equal(landing.status, 200);
  assert.match(await landing.text(), /FastWan QAD/);
  const health = await fetch(`${origin}/healthz`);
  assert.deepEqual(await health.json(), { ok: true, storage: false, release: "v2-mlx-int8", ready: false });
});

test("catalog uses stable landing endpoints", async () => {
  const response = await fetch(`${origin}/catalog.json`);
  const catalog = await response.json();
  assert.equal(catalog.shared.url, "https://fastwan.example/download/shared");
  assert.equal(catalog.variants.ema.url, "https://fastwan.example/download/ema");
});

test("unpublished artifacts fail closed", async () => {
  const response = await fetch(`${origin}/download/app`, { redirect: "manual" });
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), { error: "This release asset is not published yet." });
});
