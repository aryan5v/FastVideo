"use strict";
// Blind A/B ballot page: synchronized playback, voting, optional reveal.

const $ = (id) => document.getElementById(id);
const videos = { left: $("video-left"), right: $("video-right") };
const viewports = { left: $("vp-left"), right: $("vp-right") };
const STORE = {
  voter: "blindab.voter", reveal: "blindab.reveal", loop: "blindab.loop", audio: "blindab.audio",
};
const DRIFT_TOLERANCE_S = 0.08;

let ballot = null;          // current ballot from the server
let voted = false;
let playing = false;
let shownAt = 0;            // performance.now() when the ballot was shown
let playedMs = 0;           // wall time spent playing for this ballot
let lastTick = 0;
let seeking = false;

// ---------- settings ----------
function loadSettings() {
  $("reveal-chk").checked = localStorage.getItem(STORE.reveal) !== "0";
  $("loop-chk").checked = localStorage.getItem(STORE.loop) !== "0";
  $("audio-sel").value = localStorage.getItem(STORE.audio) || "off";
  applyAudio();
}
function saveSetting(key, value) { localStorage.setItem(key, value); }

function getVoter(forcePrompt) {
  let name = localStorage.getItem(STORE.voter) || "";
  while (forcePrompt || !name.trim()) {
    const entered = window.prompt("Your name (stored in this browser, saved with each vote):", name);
    if (entered === null && name.trim()) break;
    name = (entered || "").trim().slice(0, 64);
    forcePrompt = false;
  }
  localStorage.setItem(STORE.voter, name);
  $("voter-btn").textContent = name;
  return name;
}

function setStatus(text, isError) {
  $("status").textContent = text || "";
  $("status").classList.toggle("error", Boolean(isError));
}

// ---------- playback sync ----------
function durationOf(v) { return Number.isFinite(v.duration) ? v.duration : 0; }
function master() {
  return durationOf(videos.right) > durationOf(videos.left) ? videos.right : videos.left;
}
function maxDuration() { return Math.max(durationOf(videos.left), durationOf(videos.right)); }

function seekTo(t) {
  for (const v of Object.values(videos)) {
    const d = durationOf(v);
    if (d) v.currentTime = Math.min(Math.max(0, t), Math.max(0, d - 0.04));
  }
  updateTimeUi();
}

function play() {
  if (!ballot) return;
  const m = master();
  if (durationOf(m) && m.currentTime >= durationOf(m) - 0.05) seekTo(0);
  playing = true;
  lastTick = performance.now();
  for (const v of Object.values(videos)) {
    if (v.currentTime < durationOf(v) - 0.05) v.play().catch(() => {});
  }
  $("play-btn").textContent = "Pause";
  requestAnimationFrame(tick);
}

function pause() {
  playing = false;
  for (const v of Object.values(videos)) v.pause();
  $("play-btn").textContent = "Play";
}

function togglePlay() { playing ? pause() : play(); }

function tick(now) {
  if (!playing) return;
  playedMs += now - lastTick;
  lastTick = now;
  const m = master();
  const s = m === videos.left ? videos.right : videos.left;
  if (!seeking && durationOf(s) && m.currentTime < durationOf(s) - 0.05
      && Math.abs(s.currentTime - m.currentTime) > DRIFT_TOLERANCE_S) {
    s.currentTime = m.currentTime;
  }
  updateTimeUi();
  requestAnimationFrame(tick);
}

function updateTimeUi() {
  const d = maxDuration();
  const t = master().currentTime || 0;
  if (!seeking) $("seek").value = d ? Math.round((t / d) * 1000) : 0;
  $("time").textContent = `${t.toFixed(1)} / ${d.toFixed(1)} s`;
}

function onEnded(event) {
  if (event.target !== master()) return;
  if ($("loop-chk").checked && playing) {
    seekTo(0);
    play();
  } else {
    pause();
  }
}

function applyAudio() {
  const sel = $("audio-sel").value;
  videos.left.muted = sel !== "left";
  videos.right.muted = sel !== "right";
}

function cycleAudio() {
  const order = ["off", "left", "right"];
  const sel = $("audio-sel");
  sel.value = order[(order.indexOf(sel.value) + 1) % order.length];
  saveSetting(STORE.audio, sel.value);
  applyAudio();
}

// ---------- 1:1 crop ----------
let syncingScroll = false;
function applyZoom() {
  const on = $("zoom-chk").checked;
  for (const side of ["left", "right"]) {
    const v = videos[side];
    const vp = viewports[side];
    vp.classList.toggle("zoom", on);
    v.style.width = on && v.videoWidth ? `${v.videoWidth}px` : "";
    v.style.height = on && v.videoHeight ? `${v.videoHeight}px` : "";
    if (on) {
      vp.scrollLeft = (vp.scrollWidth - vp.clientWidth) / 2;
      vp.scrollTop = (vp.scrollHeight - vp.clientHeight) / 2;
    }
  }
}
function syncScroll(from, to) {
  if (syncingScroll) return;
  syncingScroll = true;
  to.scrollLeft = from.scrollLeft;
  to.scrollTop = from.scrollTop;
  requestAnimationFrame(() => { syncingScroll = false; });
}

// ---------- ballots & votes ----------
async function loadBallot() {
  pause();
  const voter = getVoter(false);
  setStatus("Loading next pair...");
  let data;
  try {
    const resp = await fetch(`/api/ballot?voter=${encodeURIComponent(voter)}`);
    data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
  } catch (err) {
    setStatus(`Could not load a ballot: ${err.message}`, true);
    return;
  }
  ballot = data;
  voted = false;
  playedMs = 0;
  $("comment").value = "";
  $("reveal-left").textContent = "";
  $("reveal-right").textContent = "";
  $("after-panel").classList.add("hidden");
  $("vote-panel").classList.remove("disabled");
  document.querySelectorAll(".choice").forEach((b) => { b.disabled = false; b.classList.remove("picked"); });
  $("prompt-text").textContent = data.prompt || "(no prompt text in bundle)";
  $("prompt-preview").textContent = (data.prompt || "").slice(0, 140) + ((data.prompt || "").length > 140 ? "..." : "");
  $("vote-count").textContent = `${data.voter_votes} votes by you`;
  let ready = 0;
  const onReady = () => {
    ready += 1;
    if (ready === 2) {
      applyZoom();
      updateTimeUi();
      shownAt = performance.now();
      setStatus("");
      play();
    }
  };
  for (const side of ["left", "right"]) {
    videos[side].addEventListener("loadeddata", onReady, { once: true });
    videos[side].src = data[`${side}_url`];
    videos[side].load();
  }
  applyAudio();
}

async function vote(choice) {
  if (!ballot || voted) return;
  voted = true;
  document.querySelectorAll(".choice").forEach((b) => {
    b.disabled = true;
    b.classList.toggle("picked", b.dataset.choice === choice);
  });
  const payload = {
    ballot_id: ballot.ballot_id,
    voter: getVoter(false),
    choice,
    comment: $("comment").value.trim(),
    watch_seconds: shownAt ? (performance.now() - shownAt) / 1000 : 0,
    played_seconds: playedMs / 1000,
  };
  let data;
  try {
    const resp = await fetch("/api/vote", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    data = await resp.json();
    if (resp.status === 410) { setStatus(data.error, true); loadBallot(); return; }
    if (!resp.ok) throw new Error(data.error || resp.statusText);
  } catch (err) {
    voted = false;
    document.querySelectorAll(".choice").forEach((b) => { b.disabled = false; b.classList.remove("picked"); });
    setStatus(`Vote not saved: ${err.message}`, true);
    return;
  }
  $("vote-count").textContent = `${data.voter_votes} votes by you`;
  if (!$("reveal-chk").checked) { loadBallot(); return; }
  $("reveal-left").textContent = data.reveal.left.display_name;
  $("reveal-right").textContent = data.reveal.right.display_name;
  $("after-text").textContent = `Saved. Left = ${data.reveal.left.display_name}, Right = ${data.reveal.right.display_name}.`;
  $("vote-panel").classList.add("disabled");
  $("after-panel").classList.remove("hidden");
  $("next-btn").focus();
}

// ---------- wiring ----------
function isTyping(target) {
  return target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) && target.type !== "checkbox"
    && target.type !== "range";
}

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (isTyping(e.target)) {
    if (e.key === "Escape") e.target.blur();
    if (e.key === "Enter" && e.target.id === "comment") e.target.blur();
    return;
  }
  const choiceKeys = { 1: "left", 2: "right", 3: "tie", 4: "both_bad" };
  if (choiceKeys[e.key]) { e.preventDefault(); vote(choiceKeys[e.key]); }
  else if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    seekTo((master().currentTime || 0) + (e.key === "ArrowLeft" ? -1 : 1));
  }
  else if (e.key === "z") { $("zoom-chk").checked = !$("zoom-chk").checked; applyZoom(); }
  else if (e.key === "m") cycleAudio();
  else if (e.key === "p") $("prompt-box").open = !$("prompt-box").open;
  else if ((e.key === "Enter" || e.key === "n") && voted) { e.preventDefault(); loadBallot(); }
});

$("play-btn").addEventListener("click", togglePlay);
$("next-btn").addEventListener("click", loadBallot);
$("voter-btn").addEventListener("click", () => getVoter(true));
document.querySelectorAll(".choice").forEach((b) => b.addEventListener("click", () => vote(b.dataset.choice)));
$("seek").addEventListener("input", (e) => {
  seeking = true;
  seekTo((Number(e.target.value) / 1000) * maxDuration());
});
$("seek").addEventListener("change", () => { seeking = false; });
$("loop-chk").addEventListener("change", (e) => saveSetting(STORE.loop, e.target.checked ? "1" : "0"));
$("reveal-chk").addEventListener("change", (e) => saveSetting(STORE.reveal, e.target.checked ? "1" : "0"));
$("audio-sel").addEventListener("change", (e) => { saveSetting(STORE.audio, e.target.value); applyAudio(); });
$("zoom-chk").addEventListener("change", applyZoom);
for (const side of ["left", "right"]) {
  videos[side].addEventListener("ended", onEnded);
  videos[side].addEventListener("click", togglePlay);
  videos[side].addEventListener("error", () => setStatus(`The ${side} video failed to load.`, true));
}
viewports.left.addEventListener("scroll", () => syncScroll(viewports.left, viewports.right));
viewports.right.addEventListener("scroll", () => syncScroll(viewports.right, viewports.left));

fetch("/api/info").then((r) => r.json()).then((info) => {
  $("bundle-name").textContent = `${info.bundle}: ${info.arms} arms, ${info.clips} prompts`;
}).catch(() => {});
loadSettings();
loadBallot();
