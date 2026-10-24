/* AI Media Filter — replays precomputed results (results/P0.jsonl, P1.jsonl). Nothing is inferred here. */
(() => {
  "use strict";
  const THRESHOLD = 0.5;
  const ANALYZE_MIN = 1.2, ANALYZE_MAX = 3.0;   // "Analyzing…" time = the clip's measured model time, clamped
  const CHECK_SHOWN_MS = 1000;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const $ = (id) => document.getElementById(id);
  const svgUse = (id) => `<svg class="ico" aria-hidden="true"><use href="#${id}"/></svg>`;
  const fmt = (score) => score.toFixed(2);
  const isPlaceholder = (name) => /^Team member \d+$/.test(name);

  async function loadJSON(url, fallbackId) {
    try {
      const res = await fetch(url, { cache: "no-cache" });
      if (res.ok) return await res.json();
    } catch (e) { /* file:// or offline: use the copy embedded in index.html */ }
    return JSON.parse($(fallbackId).textContent);
  }

  // ---------------------------------------------------------------- team and course (site.json)
  function membersNode(members) {
    const span = document.createElement("span");
    members.forEach((name, i) => {
      const s = document.createElement("span");
      s.textContent = name + (i < members.length - 1 ? ", " : "");
      if (isPlaceholder(name)) s.className = "placeholder";
      span.appendChild(s);
    });
    return span;
  }

  function renderSite(site) {
    const head = `${site.course} · ${site.group}`;
    document.querySelectorAll('[data-site="course-group"]').forEach((el) => { el.textContent = head; });
    document.querySelectorAll('[data-site="members"]').forEach((el) => el.replaceChildren(membersNode(site.members)));
    document.querySelectorAll('[data-site="footer-line"]').forEach((el) => {
      el.replaceChildren(document.createTextNode(head + " · "), membersNode(site.members));
    });
  }

  // ---------------------------------------------------------------- P0 vs P1 example (c020)
  function renderPair(clips) {
    const c = clips.find((x) => x.id === "c020");
    if (!c) return;
    const card = (title, r) => {
      const warn = r.score >= THRESHOLD;
      const el = document.createElement("div");
      el.className = "card" + (warn ? " warn" : "");
      el.innerHTML = `<div class="top"><span class="who"></span><span class="${warn ? "warn" : "ok"}">${fmt(r.score)} ${svgUse(warn ? "i-warn" : "i-check")}</span></div><div class="reason"></div>`;
      el.querySelector(".who").textContent = title;
      el.querySelector(".reason").textContent = `“${r.reason}”`;
      return el;
    };
    $("pair").replaceChildren(card("P0 · NO DEFINITION", c.p0), card("P1 · WITH DEFINITION", c.p1));
  }

  // ---------------------------------------------------------------- feed grid
  let clips = [];
  let prompt = "p1";
  const seen = new Map();          // clip id -> "o" | "w", cleared when the prompt changes

  function renderGrid() {
    const grid = $("grid");
    grid.replaceChildren(...clips.map((c, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "tile";
      b.dataset.index = i;
      b.innerHTML = `<img src="${c.thumb}" alt="" loading="lazy">`;
      b.setAttribute("aria-label", `Play clip ${c.id} (dataset label: ${c.label})`);
      b.addEventListener("click", () => openViewer(i, b));
      return b;
    }));
    $("clip-count").textContent = `${clips.length} clips`;
  }

  function setBadge(id, kind) {
    seen.set(id, kind);
    const tile = $("grid").querySelector(`.tile[data-index="${clips.findIndex((c) => c.id === id)}"]`);
    if (!tile) return;
    tile.querySelector(".badge")?.remove();
    const s = document.createElement("span");
    s.className = `badge ${kind}`;
    s.innerHTML = `<svg aria-hidden="true"><use href="#${kind === "o" ? "i-check" : "i-bang"}"/></svg>`;
    tile.appendChild(s);
    const c = clips.find((x) => x.id === id);
    tile.setAttribute("aria-label", `Play clip ${c.id} (dataset label: ${c.label}; filter: ${kind === "o" ? "checked" : "possible brainrot"})`);
  }

  function clearBadges() {
    seen.clear();
    $("grid").querySelectorAll(".badge").forEach((b) => b.remove());
    $("grid").querySelectorAll(".tile").forEach((t) => {
      const c = clips[+t.dataset.index];
      t.setAttribute("aria-label", `Play clip ${c.id} (dataset label: ${c.label})`);
    });
  }

  document.querySelectorAll(".seg button").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.prompt === prompt) return;
      prompt = btn.dataset.prompt;
      document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      clearBadges();
    });
  });

  // ---------------------------------------------------------------- viewer
  const viewer = $("viewer"), vid = $("vid"), stage = $("stage"), chip = $("chip"), dim = $("dim"), wcard = $("wcard");
  let index = 0, run = 0, opener = null, cardUp = false, busy = false, rafId = 0;

  function resetOverlays() {
    run++;                                           // cancels a pending analysis timer
    chip.className = "chip gone";
    chip.replaceChildren();
    dim.classList.remove("on");
    wcard.classList.remove("up");
    cardUp = false;
    vid.volume = 1;
  }

  function analyze() {
    const c = clips[index], r = c[prompt], token = ++run;
    chip.className = "chip";
    chip.innerHTML = '<span class="spin" aria-hidden="true"></span>Analyzing…';
    const ms = Math.min(Math.max(r.sec, ANALYZE_MIN), ANALYZE_MAX) * 1000;
    setTimeout(() => { if (token === run) verdict(c, r, token); }, ms);
  }

  function verdict(c, r, token) {
    if (r.score < THRESHOLD) {
      chip.className = "chip ok";
      chip.innerHTML = `${svgUse("i-check")}<span></span>`;
      chip.querySelector("span").textContent = `Checked · ${fmt(r.score)}`;
      setBadge(c.id, "o");
      setTimeout(() => { if (token === run) chip.classList.add("gone"); }, CHECK_SHOWN_MS);
      return;
    }
    chip.className = "chip gone";
    $("w-sub").textContent = `Split-screen sludge · score ${fmt(r.score)} · threshold ${fmt(THRESHOLD)}`;
    $("w-reason").textContent = `“${r.reason}”`;
    const note = $("w-note");
    note.hidden = !c.note;
    if (c.note) {
      const m = c.note.match(/^(Our check:)\s*(.*)$/);
      note.replaceChildren();
      if (m) {
        const b = document.createElement("b");
        b.textContent = m[1];
        note.append(b, " " + m[2]);
      } else {
        note.textContent = c.note;
      }
    }
    $("w-label").textContent = `Dataset label: ${c.label}`;
    dim.classList.add("on");
    if (!vid.muted) vid.volume = 0.3;
    wcard.scrollTop = 0;
    wcard.classList.add("up");
    cardUp = true;
    setBadge(c.id, "w");
  }

  function load(i) {
    index = (i + clips.length) % clips.length;
    const c = clips[index];
    resetOverlays();
    vid.poster = c.poster;
    vid.src = c.video;
    vid.loop = true;
    vid.currentTime = 0;
    vid.play().catch(() => {});
    $("v-label").textContent = `Dataset label: ${c.label}`;
    analyze();
  }

  async function go(delta) {
    if (busy) return;
    busy = true;
    resetOverlays();
    if (!reduceMotion.matches) {
      await stage.animate([{ transform: "translateY(0)" }, { transform: `translateY(${delta > 0 ? -100 : 100}%)` }],
                          { duration: 220, easing: "ease-in" }).finished;
      load(index + delta);
      await stage.animate([{ transform: `translateY(${delta > 0 ? 100 : -100}%)` }, { transform: "translateY(0)" }],
                          { duration: 240, easing: "ease-out" }).finished;
    } else {
      load(index + delta);
    }
    busy = false;
  }

  function tick() {
    if (vid.duration) $("v-prog").style.width = `${(vid.currentTime / vid.duration) * 100}%`;
    rafId = requestAnimationFrame(tick);
  }

  function openViewer(i, from) {
    opener = from;
    viewer.hidden = false;
    if (window.matchMedia("(max-width: 899px)").matches) document.body.style.overflow = "hidden";
    load(i);
    cancelAnimationFrame(rafId);
    tick();
    $("close").focus({ preventScroll: true });
  }

  function closeViewer() {
    resetOverlays();
    vid.pause();
    vid.removeAttribute("src");
    vid.load();
    viewer.hidden = true;
    document.body.style.overflow = "";
    cancelAnimationFrame(rafId);
    opener?.focus({ preventScroll: true });
  }

  function setMuted(muted) {
    vid.muted = muted;
    vid.volume = cardUp ? 0.3 : 1;
    const btn = $("mute");
    btn.setAttribute("aria-label", muted ? "Turn sound on" : "Mute");
    btn.innerHTML = `<svg aria-hidden="true"><use href="#${muted ? "i-sound-off" : "i-sound-on"}"/></svg>`;
  }

  $("mute").addEventListener("click", () => setMuted(!vid.muted));
  $("close").addEventListener("click", closeViewer);
  $("skip").addEventListener("click", () => go(1));
  $("keep").addEventListener("click", () => {
    wcard.classList.remove("up");
    dim.classList.remove("on");
    cardUp = false;
    vid.volume = 1;
    $("close").focus({ preventScroll: true });
  });

  document.addEventListener("keydown", (e) => {
    if (viewer.hidden) return;
    if (e.key === "Escape") { e.preventDefault(); closeViewer(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); go(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); go(-1); }
    else if (e.key === "Tab") {                       // keep focus inside the viewer
      const f = [...viewer.querySelectorAll("button")].filter((b) => b.offsetParent !== null &&
                 !(b.closest(".wcard") && !cardUp));
      if (!f.length) return;
      const at = f.indexOf(document.activeElement);
      const next = e.shiftKey ? (at <= 0 ? f.length - 1 : at - 1) : (at === f.length - 1 ? 0 : at + 1);
      e.preventDefault();
      f[next].focus();
    }
  });

  let wheelLock = 0;
  viewer.addEventListener("wheel", (e) => {
    if (e.target.closest(".wcard") && wcard.scrollHeight > wcard.clientHeight) return;   // let a long card scroll
    e.preventDefault();
    const now = Date.now();
    if (Math.abs(e.deltaY) < 25 || now < wheelLock) return;
    wheelLock = now + 700;
    go(e.deltaY > 0 ? 1 : -1);
  }, { passive: false });

  // swipe up / down = next / previous; dragging down from the top bar = back to the grid
  let drag = null;
  viewer.addEventListener("pointerdown", (e) => {
    if (e.target.closest("button") || (e.target.closest(".wcard") && cardUp)) return;
    const top = viewer.getBoundingClientRect().top;
    drag = { y: e.clientY, fromTop: e.clientY - top < 90 };
  });
  viewer.addEventListener("pointerup", (e) => {
    if (!drag) return;
    const dy = e.clientY - drag.y;
    if (drag.fromTop && dy > 60) closeViewer();
    else if (dy < -50) go(1);
    else if (dy > 50) go(-1);
    drag = null;
  });
  viewer.addEventListener("pointercancel", () => { drag = null; });

  // ---------------------------------------------------------------- start
  Promise.all([loadJSON("assets/site.json", "site-data"), loadJSON("assets/clips.json", "clips-data")])
    .then(([site, data]) => {
      clips = data;
      renderSite(site);
      renderPair(clips);
      renderGrid();
    });
})();
