"use strict";
// Blind A/B ballot page: synchronized playback, voting, optional reveal, finite sessions.

const $ = (id) => document.getElementById(id);
const videos = { left: $("video-left"), right: $("video-right") };
const viewports = { left: $("vp-left"), right: $("vp-right") };
const STORE = {
  voter: "blindab.voter", reveal: "blindab.reveal", loop: "blindab.loop", intro: "blindab.introSeen",
  sessionPrefix: "blindab.session.", audio: "blindab.audio",
};
const MAX_NAME_LEN = 64;
const AUDIO_PROBE_S = 0.5;  // fallback detection: check decoded audio bytes after this much playback
const DRIFT_TOLERANCE_S = 0.08;
const SESSION_GOAL = 12;

let ballot = null;          // current ballot from the server
let voted = false;
let playing = false;
let shownAt = 0;            // performance.now() when the ballot was shown
let playedMs = 0;           // wall time spent playing for this ballot
let lastTick = 0;
let seeking = false;
let totalVotes = null;      // this voter's votes across all sessions, from the server
const AUDIO_CHOICES = ["off", "left", "right"];
let audioChoice = "off";    // which side you hear (remembered across ballots)
let clipHasAudio = false;   // whether the current clip has an audio track
let audioProbePending = false;

// ---------- settings ----------
function loadSettings() {
  $("reveal-chk").checked = localStorage.getItem(STORE.reveal) !== "0";
  $("loop-chk").checked = localStorage.getItem(STORE.loop) !== "0";
  const saved = localStorage.getItem(STORE.audio);
  audioChoice = AUDIO_CHOICES.includes(saved) ? saved : "off";
  applyAudio();
}
function saveSetting(key, value) { localStorage.setItem(key, value); }

// ---------- voter name (required; blank or whitespace counts as missing) ----------
function currentVoter() { return (localStorage.getItem(STORE.voter) || "").trim(); }
function showVoter() {
  $("voter-name-label").textContent = currentVoter() || "(not set)";
  updateProgress();
}

// ---------- sessions (goal of SESSION_GOAL votes, persisted per voter) ----------
function sessionKey() { return STORE.sessionPrefix + (localStorage.getItem(STORE.voter) || ""); }
function getSession() {
  try {
    const saved = JSON.parse(localStorage.getItem(sessionKey()) || "null");
    if (saved && Number.isInteger(saved.done) && Number.isInteger(saved.goal)) return saved;
  } catch (err) {
    console.warn("resetting unreadable session progress", err);
  }
  return { done: 0, goal: SESSION_GOAL };
}
function saveSession(session) {
  localStorage.setItem(sessionKey(), JSON.stringify(session));
  updateProgress();
}
function updateProgress() {
  const { done, goal } = getSession();
  $("progress").textContent = `${Math.min(done, goal)} / ${goal}`;
}
function sessionFinished() {
  const { done, goal } = getSession();
  return done >= goal;
}

function overlayOpen() { return !$("intro").classList.contains("hidden") || !$("done").classList.contains("hidden"); }

function showDone() {
  pause();
  const { done } = getSession();
  const total = totalVotes !== null && totalVotes > done ? ` (${totalVotes} in total)` : "";
  $("done-text").textContent = `You compared ${done} pair${done === 1 ? "" : "s"}${total}. `
    + "Every vote helps; you can stop here or keep going.";
  $("done").classList.remove("hidden");
  $("more-btn").focus();
}

function moreVotes() {
  const session = getSession();
  saveSession({ done: session.done, goal: session.done + SESSION_GOAL });
  $("done").classList.add("hidden");
  loadBallot();
}

// Next ballot, or the thank-you screen once the session goal is reached.
function advance() {
  if (sessionFinished()) showDone();
  else loadBallot();
}

function showIntro() {
  pause();
  $("voter-input").value = currentVoter();
  updateStartButton();
  $("intro").classList.remove("hidden");
  (currentVoter() ? $("intro-ok") : $("voter-input")).focus();
}

function updateStartButton() {
  $("intro-ok").disabled = !$("voter-input").value.trim();
}

function submitIntro(event) {
  event.preventDefault();
  const name = $("voter-input").value.trim().slice(0, MAX_NAME_LEN);
  if (!name) {
    $("name-hint").textContent = "Please enter your name to start.";
    $("voter-input").focus();
    return;
  }
  const changed = name !== currentVoter();
  localStorage.setItem(STORE.voter, name);
  localStorage.setItem(STORE.intro, "1");
  showVoter();
  $("intro").classList.add("hidden");
  if (changed) $("done").classList.add("hidden");
  if (!ballot || changed || sessionFinished()) start();
}

function start() {
  if (!currentVoter()) { showIntro(); return; }
  if (sessionFinished()) showDone();
  else if (!ballot || voted) loadBallot();
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
    if (v.currentTime < durationOf(v) - 0.05) v.play().catch((err) => onPlayRejected(v, err));
  }
  $("play-btn").textContent = "Pause";
  requestAnimationFrame(tick);
}

function onPlayRejected(v, err) {
  // Browsers may block unmuted autoplay until the user interacts; fall back to muted playback.
  if (err && err.name === "NotAllowedError" && !v.muted) {
    v.muted = true;
    setStatus("The browser blocked sound; pick an Audio side again to hear it.");
    if (playing) v.play().catch(() => {});
  }
}

function pause() {
  playing = false;
  for (const v of Object.values(videos)) v.pause();
  $("play-btn").textContent = "Play";
}

function togglePlay() { playing ? pause() : play(); }

function tick(now) {
  if (!playing) return;
  playedMs += Math.max(0, now - lastTick); // rAF timestamps can predate the performance.now() taken in play()
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

// ---------- audio: Off / Left / Right, with the audible side highlighted ----------
function applyAudio() {
  const active = clipHasAudio ? audioChoice : "off";
  for (const side of ["left", "right"]) {
    videos[side].muted = active !== side;
    viewports[side].classList.toggle("audible", active === side);
    $(`speaker-${side}`).classList.toggle("hidden", active !== side);
  }
  document.querySelectorAll("#audio-ctl button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.audio === audioChoice));
  });
  $("audio-ctl").classList.toggle("hidden", !clipHasAudio);
  $("no-audio-note").classList.toggle("hidden", clipHasAudio);
}

function setAudio(choice) {
  if (!AUDIO_CHOICES.includes(choice)) return;
  audioChoice = choice;
  saveSetting(STORE.audio, choice);
  applyAudio();
}

function cycleAudio() {
  if (!clipHasAudio) return;
  setAudio(AUDIO_CHOICES[(AUDIO_CHOICES.indexOf(audioChoice) + 1) % AUDIO_CHOICES.length]);
}

function setClipHasAudio(value) {
  clipHasAudio = Boolean(value);
  applyAudio();
}

// Fallback when the server does not say whether the clip has audio (e.g. the local Python server).
function probeAudio(v) {
  if (v.audioTracks && v.audioTracks.length > 0) return true;
  if (v.mozHasAudio) return true;
  if (typeof v.webkitAudioDecodedByteCount === "number" && v.webkitAudioDecodedByteCount > 0) return true;
  return false;
}

function onLeftTimeUpdate() {
  if (!audioProbePending || videos.left.currentTime < AUDIO_PROBE_S) return;
  audioProbePending = false;
  setClipHasAudio(probeAudio(videos.left) || probeAudio(videos.right));
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
  const voter = currentVoter();
  if (!voter) { showIntro(); return; }
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
  totalVotes = data.voter_votes;
  audioProbePending = typeof data.has_audio !== "boolean";
  setClipHasAudio(data.has_audio === true);
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
  if (!currentVoter()) { showIntro(); return; }
  voted = true;
  document.querySelectorAll(".choice").forEach((b) => {
    b.disabled = true;
    b.classList.toggle("picked", b.dataset.choice === choice);
  });
  const payload = {
    ballot_id: ballot.ballot_id,
    voter: currentVoter(),
    choice,
    comment: $("comment").value.trim(),
    watch_seconds: shownAt ? (performance.now() - shownAt) / 1000 : 0,
    played_seconds: Math.max(0, playedMs) / 1000,
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
  totalVotes = data.voter_votes;
  const session = getSession();
  saveSession({ done: session.done + 1, goal: session.goal });
  if (!$("reveal-chk").checked) { advance(); return; }
  $("reveal-left").textContent = data.reveal.left.display_name;
  $("reveal-right").textContent = data.reveal.right.display_name;
  $("after-text").textContent = `Saved. Left was ${data.reveal.left.display_name}; right was ${data.reveal.right.display_name}.`;
  $("next-btn").lastChild.textContent = sessionFinished() ? " Finish" : " Next";
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
  if (e.metaKey || e.ctrlKey || e.altKey || overlayOpen()) return;
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
  else if (e.key === "m") cycleAudio();
  else if (e.key === "z") { $("zoom-chk").checked = !$("zoom-chk").checked; applyZoom(); }
  else if (e.key === "p") $("prompt-box").open = !$("prompt-box").open;
  else if ((e.key === "Enter" || e.key === "n") && voted) { e.preventDefault(); advance(); }
});

$("play-btn").addEventListener("click", togglePlay);
$("next-btn").addEventListener("click", advance);
$("voter-btn").addEventListener("click", showIntro);
document.querySelectorAll("#audio-ctl button").forEach((b) => b.addEventListener("click", () => setAudio(b.dataset.audio)));
$("voter-input").addEventListener("input", () => {
  updateStartButton();
  $("name-hint").textContent = "Your name is saved with each vote.";
});
$("help-btn").addEventListener("click", showIntro);
$("intro-form").addEventListener("submit", submitIntro);
$("more-btn").addEventListener("click", moreVotes);
document.querySelectorAll(".choice").forEach((b) => b.addEventListener("click", () => vote(b.dataset.choice)));
$("seek").addEventListener("input", (e) => {
  seeking = true;
  seekTo((Number(e.target.value) / 1000) * maxDuration());
});
$("seek").addEventListener("change", () => { seeking = false; });
$("loop-chk").addEventListener("change", (e) => saveSetting(STORE.loop, e.target.checked ? "1" : "0"));
$("reveal-chk").addEventListener("change", (e) => saveSetting(STORE.reveal, e.target.checked ? "1" : "0"));
$("zoom-chk").addEventListener("change", applyZoom);
videos.left.addEventListener("timeupdate", onLeftTimeUpdate);
for (const side of ["left", "right"]) {
  videos[side].addEventListener("ended", onEnded);
  videos[side].addEventListener("click", togglePlay);
  videos[side].addEventListener("error", () => setStatus(`The ${side} video failed to load.`, true));
}
viewports.left.addEventListener("scroll", () => syncScroll(viewports.left, viewports.right));
viewports.right.addEventListener("scroll", () => syncScroll(viewports.right, viewports.left));

fetch("/api/info").then((r) => r.json()).then((info) => {
  $("bundle-name").textContent = `${info.clips} clips`;
  $("intro-audio").classList.toggle("hidden", info.has_audio !== true);
}).catch((err) => console.warn("could not load /api/info", err));
loadSettings();
showVoter();
if (localStorage.getItem(STORE.intro) === "1" && currentVoter()) start();
else showIntro();
