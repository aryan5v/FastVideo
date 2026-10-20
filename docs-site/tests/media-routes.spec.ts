import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";

const index = JSON.parse(
  await readFile(new URL("../public/assets/cookbook-config/index.json", import.meta.url), "utf8"),
) as { models: { id: string; title: string }[] };

test("every generated model has a directory card and a reachable cookbook page", async ({ page }) => {
  test.setTimeout(180000);
  await page.route(/^https?:\/\/(?!127\.0\.0\.1|localhost)/, (request) => request.abort());
  for (const model of index.models) {
    await page.goto("/FastVideo/cookbook");
    const href = await page
      .locator(".recipe-card")
      .evaluateAll(
        (links, id) =>
          links.map((link) => link.getAttribute("href")).find((value) => value?.split("?")[0].endsWith(`/${id}`)),
        model.id,
      );
    expect(href, `${model.id} should have a directory card`).toBeTruthy();
    await page.goto(href!);
    await expect(page.locator(".recipe-matrix"), `${model.id} should open an editor`).toBeVisible({ timeout: 30000 });
    await expect(page.locator("h1")).toBeVisible();
  }
});
