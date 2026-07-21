const button = document.querySelector("#download");
const note = document.querySelector("#release-note");

try {
  const response = await fetch("/api/release", { cache: "no-store" });
  const release = await response.json();
  const app = release.artifacts.app;
  if (app.available) {
    button.classList.remove("disabled");
    button.removeAttribute("aria-disabled");
    button.textContent = "Download for Apple silicon";
    const size = new Intl.NumberFormat(undefined, { style: "unit", unit: "megabyte", maximumFractionDigits: 0 }).format(app.bytes / 1_000_000);
    note.textContent = `${size} · ${release.release}`;
  } else {
    note.textContent = "The signed release is being prepared.";
  }
} catch {
  note.textContent = "Release status is temporarily unavailable.";
}
