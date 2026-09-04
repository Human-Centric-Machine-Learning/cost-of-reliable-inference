// Page logic: configuration state, the live readout, the worker, and the results.

(() => {
  const DATA = "data/";
  const $ = (id) => document.getElementById(id);
  const TABS = ["overview", "selection", "regret", "payments", "beliefs", "robustness"];

  let CATALOG = null,
    PRESETS = null,
    RESULT = null,
    SWEEP = null;
  let worker = null,
    engineReady = false,
    jobCounter = 0;
  const jobs = new Map();
  let activeTab = "overview";
  const dirty = new Set(TABS);

  const state = {
    benchmark: "GSM8K",
    roster: [],
    theta: 0.5,
    delta: 0.05,
    gamma: 0,
    margin: 0.25,
    c_max: null,
    estimator: { kind: "empirical_mean", k: 10, prior_mean: 0, bias: 0.25 },
    arms: [],
    repetitions: 3,
    t_max: null,
    master_seed: 20260826,
  };

  // ---- helpers ------------------------------------------------------------------

  const fetchJSON = (name) => fetch(DATA + name).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${name}: ${r.status}`))));
  const fmt = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
  const fmtInt = (x) => (x === null || x === undefined ? "–" : Math.round(x).toLocaleString("en-US"));
  const pm = (s, d = 3) => (s ? `${fmt(s.mean, d)} ± ${fmt(s.sd, d)}` : "–");
  const el = (tag, attrs = {}, children = []) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const c of children) e.append(c);
    return e;
  };

  // Render the LaTeX inside a node (or the whole page) once it has been (re)built.
  function typeset(node) {
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise(node ? [node] : undefined).catch(() => {});
  }

  function cands() {
    return Live.candidates(CATALOG, state.benchmark);
  }
  function needsRewards() {
    return state.roster.some((m) => m.includes("@"));
  }
  function questions() {
    const c = CATALOG[state.benchmark];
    return needsRewards() ? c.questions_with_rewards : c.questions;
  }
  function horizon() {
    return state.t_max || questions();
  }
  function liveTruth() {
    return Live.truth(cands(), state.roster, state.theta, state.margin);
  }
  function tokenCap() {
    return PRESETS?.mechanism?.token_cap ?? 512;
  }
  function pinned() {
    return state.roster.length ? Live.pinnedCmax(cands(), state.roster, tokenCap()) : 0;
  }
  function floorC() {
    return state.roster.length ? Live.floorCmax(cands(), state.roster, tokenCap(), state.margin) : 0;
  }
  function cMax() {
    return state.c_max === null ? pinned() : state.c_max;
  }

  function matchPreset() {
    for (const env of PRESETS.environments) {
      const roster = PRESETS.rosters[env.roster];
      if (env.benchmark === state.benchmark && env.theta === state.theta && env.t_max === horizon() && roster.length === state.roster.length && roster.every((m, i) => m === state.roster[i])) return env.id;
    }
    return null;
  }

  function buildRequest(overrides = {}) {
    return {
      benchmark: state.benchmark,
      roster: [...state.roster],
      theta: state.theta,
      delta: state.delta,
      gamma: state.gamma,
      margin: state.margin,
      token_cap: tokenCap(),
      c_max: state.c_max,
      estimator: { ...state.estimator },
      arms: [...state.arms],
      repetitions: state.repetitions,
      t_max: state.t_max,
      master_seed: state.master_seed,
      environment_id: matchPreset(),
      ...overrides,
    };
  }

  // Does the current configuration differ from the one the shown result was run with?
  function isStale() {
    if (!RESULT) return false;
    const r = RESULT.request,
      q = buildRequest();
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    return !(
      r.benchmark === q.benchmark &&
      same(r.roster, q.roster) &&
      r.theta === q.theta &&
      r.delta === q.delta &&
      r.gamma === q.gamma &&
      r.margin === q.margin &&
      r.token_cap === q.token_cap &&
      Math.abs(r.c_max - cMax()) < 1e-9 &&
      same(r.estimator, { kind: q.estimator.kind, k: q.estimator.k, prior_mean: q.estimator.prior_mean, bias: q.estimator.bias }) &&
      same(r.arms, q.arms) &&
      r.repetitions === q.repetitions &&
      r.t_max === horizon() &&
      r.master_seed === q.master_seed
    );
  }

  // ---- URL hash ---------------------------------------------------------------------

  function writeHash() {
    const compact = { ...state };
    delete compact.token_cap;
    if (compact.c_max === null) delete compact.c_max;
    if (compact.t_max === null) delete compact.t_max;
    history.replaceState(null, "", "#" + encodeURIComponent(JSON.stringify(compact)));
  }
  function readHash() {
    if (!location.hash || location.hash.length < 3) return null;
    try {
      const s = JSON.parse(decodeURIComponent(location.hash.slice(1)));
      if (!s.benchmark || !Array.isArray(s.roster) || typeof s.theta !== "number") return null;
      if (!CATALOG[s.benchmark]) return null;
      const known = Live.candidates(CATALOG, s.benchmark);
      s.roster = s.roster.filter((m) => known.has(m));
      delete s.token_cap;
      s.estimator = { ...state.estimator, ...(s.estimator || {}) };
      s.arms = (s.arms || []).filter((a) => CONTENT.arms[a]);
      return s;
    } catch (e) {
      return null;
    }
  }

  // ---- static UI --------------------------------------------------------------------

  function buildStaticUI() {
    const primer = $("primer-body");
    for (const p of CONTENT.primer) primer.append(el("div", {}, [el("h3", { text: p.title }), el("p", { html: p.html })]));
    const gl = $("glossary");
    for (const [term, text] of CONTENT.glossary) gl.append(el("dt", { text: term }), el("dd", { text }));
    document.querySelectorAll(".info[data-param]").forEach((e) => e.append(el("span", { class: "tip", html: CONTENT.params[e.dataset.param] || "" })));
    document.querySelectorAll(".caption[data-plot]").forEach((e) => (e.textContent = CONTENT.plots[e.dataset.plot] || ""));

    const preset = $("preset");
    preset.append(el("option", { value: "", text: "custom configuration" }));
    for (const env of PRESETS.environments) preset.append(el("option", { value: env.id, text: `${env.id}  (${env.benchmark}, ${PRESETS.rosters[env.roster].length} providers, Theta = ${env.theta})` }));
    preset.addEventListener("change", () => {
      if (preset.value) {
        applyPreset(preset.value);
        onChange();
      }
    });
    const bench = $("benchmark");
    for (const b of Object.keys(CATALOG)) bench.append(el("option", { value: b, text: `${b}  (${CATALOG[b].questions} questions)` }));
    bench.addEventListener("change", () => {
      state.benchmark = bench.value;
      onChange();
    });
    const rp = $("roster-presets");
    for (const name of Object.keys(PRESETS.rosters)) {
      rp.append(
        el("button", { class: "small", text: `${name} (${PRESETS.rosters[name].length})`, title: `Use the shipped ${name} roster`, onclick: () => {
          state.roster = [...PRESETS.rosters[name]];
          onChange();
        } }),
        " ",
      );
    }

    const armList = (id, names) => {
      const box = $(id);
      for (const name of names) {
        const a = CONTENT.arms[name];
        const cb = el("input", { type: "checkbox", id: "arm-" + name });
        cb.addEventListener("change", () => {
          state.arms = state.arms.filter((x) => x !== name);
          if (cb.checked) state.arms.push(name);
          state.arms.sort((x, y) => ALL_ARMS.indexOf(x) - ALL_ARMS.indexOf(y));
          onChange();
        });
        box.append(
          el("label", { class: "arm-item" }, [
            cb,
            el("span", { class: "swatch", style: `background:${ARM_COLORS[name]}` }),
            el("span", { class: "name", text: a.label }),
            el("span", { class: "info", tabindex: "0", text: "?" }, [el("span", { class: "tip", html: a.short + " " + a.long })]),
          ]),
        );
      }
    };
    armList("arms-main", PRESETS.main_arms);
    armList("arms-ablation", PRESETS.ablation_arms);
    armList("arms-other", ALL_ARMS.filter((a) => !PRESETS.main_arms.includes(a) && !PRESETS.ablation_arms.includes(a)));

    // numeric inputs
    const bind = (id, get, set, parse = parseFloat) => {
      const input = $(id);
      input.addEventListener("change", () => {
        const v = parse(input.value);
        if (Number.isFinite(v)) set(v);
        onChange();
      });
    };
    bind("delta", () => state.delta, (v) => (state.delta = Math.min(0.999, Math.max(0.001, v))));
    bind("gamma", () => state.gamma, (v) => (state.gamma = Math.max(0, v)));
    bind("margin", () => state.margin, (v) => (state.margin = Math.max(-0.9, v)));
    bind("c_max", () => state.c_max, (v) => (state.c_max = v));
    $("c_max_auto").addEventListener("change", () => {
      state.c_max = $("c_max_auto").checked ? null : pinned();
      onChange();
    });
    $("estimator").addEventListener("change", () => {
      state.estimator.kind = $("estimator").value;
      if (state.estimator.kind === "shrinkage" && state.estimator.prior_mean === 0) state.estimator.prior_mean = Math.round(cMax() / 2);
      onChange();
    });
    bind("est_k", () => state.estimator.k, (v) => (state.estimator.k = Math.max(0.1, v)));
    bind("est_prior", () => state.estimator.prior_mean, (v) => (state.estimator.prior_mean = Math.max(0, v)));
    bind("est_bias", () => state.estimator.bias, (v) => (state.estimator.bias = Math.max(0, v)));
    bind("repetitions", () => state.repetitions, (v) => (state.repetitions = Math.min(200, Math.max(1, Math.round(v)))), parseInt);
    $("t_max").addEventListener("change", () => {
      const v = parseInt($("t_max").value);
      state.t_max = Number.isFinite(v) && v > 0 ? v : null;
      onChange();
    });
    bind("seed", () => state.master_seed, (v) => (state.master_seed = Math.max(0, Math.round(v))), parseInt);

    // threshold
    const theta = $("theta"),
      thetaN = $("theta-number");
    let raf = null;
    theta.addEventListener("input", () => {
      state.theta = parseFloat(theta.value);
      thetaN.value = state.theta.toFixed(3);
      if (!raf) raf = requestAnimationFrame(() => {
        raf = null;
        onChange(false);
      });
    });
    theta.addEventListener("change", () => onChange());
    thetaN.addEventListener("change", () => {
      const v = parseFloat(thetaN.value);
      if (Number.isFinite(v)) state.theta = Math.round(v * 1000) / 1000;
      onChange();
    });
    $("theta-prev").addEventListener("click", () => snapTheta(-1));
    $("theta-next").addEventListener("click", () => snapTheta(1));

    // tooltips: place the bubble under its icon, inside the viewport
    document.querySelectorAll(".info").forEach((icon) => {
      const place = () => {
        const tip = icon.querySelector(".tip");
        if (!tip) return;
        const r = icon.getBoundingClientRect();
        tip.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - 310))}px`;
        tip.style.top = `${r.bottom + 6}px`;
      };
      icon.addEventListener("mouseenter", place);
      icon.addEventListener("focus", place);
    });

    // tabs
    document.querySelectorAll(".tab").forEach((b) =>
      b.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
        document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
        activeTab = b.dataset.tab;
        renderTab(activeTab);
      }),
    );
    $("sel-arm").addEventListener("change", () => renderSelection(true));
    $("pay-arm").addEventListener("change", () => renderPayments(true));
    $("bel-arm").addEventListener("change", () => renderBeliefs(true));

    // buttons
    $("btn-run").addEventListener("click", runExperiment);
    $("btn-cancel").addEventListener("click", () => cancelJob("run"));
    $("btn-sweep").addEventListener("click", runSweep);
    $("btn-sweep-cancel").addEventListener("click", () => cancelJob("sweep"));
    $("btn-share").addEventListener("click", shareLink);
    $("btn-export").addEventListener("click", exportResults);
    $("btn-add-provider").addEventListener("click", openPicker);
    $("picker-done").addEventListener("click", () => $("picker").close());
    $("picker-clear").addEventListener("click", () => {
      state.roster = [];
      renderPicker();
      onChange();
    });
    $("picker-base").addEventListener("change", renderPicker);
    $("picker-frontier").addEventListener("change", renderPicker);
    $("picker").addEventListener("close", () => onChange());
  }

  const ALL_ARMS = Object.keys(CONTENT.arms);

  function applyPreset(envId) {
    const env = PRESETS.environments.find((e) => e.id === envId);
    if (!env) return;
    state.benchmark = env.benchmark;
    state.roster = [...PRESETS.rosters[env.roster]];
    state.theta = env.theta;
    state.t_max = null;
    const m = PRESETS.mechanism;
    state.delta = m.delta;
    state.gamma = m.gamma;
    state.margin = m.margin;
    state.c_max = m.c_max;
    state.estimator = { ...PRESETS.estimator };
    state.master_seed = PRESETS.master_seed;
  }

  function snapTheta(direction) {
    const mids = Live.midpoints(cands(), state.roster);
    if (!mids.length) return;
    const next = direction > 0 ? mids.find((t) => t > state.theta + 1e-9) : [...mids].reverse().find((t) => t < state.theta - 1e-9);
    if (next !== undefined) {
      state.theta = next;
      onChange();
    }
  }

  // ---- controls <-> state ---------------------------------------------------------------

  function syncControls() {
    $("benchmark").value = state.benchmark;
    $("preset").value = matchPreset() || "";
    $("delta").value = state.delta;
    $("gamma").value = state.gamma;
    $("margin").value = state.margin;
    $("c_max_auto").checked = state.c_max === null;
    $("c_max").value = cMax().toFixed(1);
    $("c_max").disabled = state.c_max === null;
    $("estimator").value = state.estimator.kind;
    $("est_k").value = state.estimator.k;
    $("est_prior").value = state.estimator.prior_mean;
    $("est_bias").value = state.estimator.bias;
    $("est-shrinkage").style.display = state.estimator.kind === "shrinkage" ? "" : "none";
    $("est-biased").style.display = state.estimator.kind === "biased" ? "" : "none";
    $("repetitions").value = state.repetitions;
    $("t_max").value = state.t_max === null ? "" : state.t_max;
    $("seed").value = state.master_seed;
    for (const name of ALL_ARMS) {
      const cb = $("arm-" + name);
      if (cb) cb.checked = state.arms.includes(name);
    }
    const c = cands();
    const qs = state.roster.map((m) => c.get(m).quality);
    const theta = $("theta");
    theta.min = qs.length ? Math.max(0, Math.floor((Math.min(...qs) - 0.02) * 1000) / 1000) : 0;
    theta.max = qs.length ? Math.min(1, Math.ceil((Math.max(...qs) + 0.01) * 1000) / 1000) : 1;
    theta.value = state.theta;
    $("theta-number").value = state.theta.toFixed(3);
  }

  function onChange(full = true) {
    if (full) {
      syncControls();
      writeHash();
    }
    refreshLive(full);
  }

  // ---- live readout -----------------------------------------------------------------------

  function refreshLive(full = true) {
    const c = cands();
    const tr = liveTruth();
    const n = state.roster.length;

    // chips, sorted by quality
    const chips = $("roster-chips");
    chips.replaceChildren();
    const sorted = [...state.roster].sort((a, b) => c.get(b).quality - c.get(a).quality);
    for (const m of sorted) {
      const cls = !tr.q ? "none" : m === tr.istar ? "star" : (tr.qualified || []).includes(m) ? "q" : "u";
      const cand = c.get(m);
      chips.append(
        el("span", { class: "chip " + cls, title: `quality ${cand.quality.toFixed(3)}, cost ${Live.cost(cand, state.margin).toFixed(1)} $` }, [
          m + (m === tr.istar ? " ★" : ""),
          el("button", { text: "×", title: "remove", onclick: () => {
            state.roster = state.roster.filter((x) => x !== m);
            onChange();
          } }),
        ]),
      );
    }
    if (!n) chips.append(el("span", { class: "hint", text: "no providers yet" }));

    // truth readout
    const dl = $("truth");
    dl.replaceChildren();
    const row = (k, v) => dl.append(el("dt", { text: k }), el("dd", { html: v }));
    if (n < 2) {
      $("truth-hint").textContent = "add at least two providers";
      $("truth-hint").className = "hint bad";
    } else if (!tr.valid) {
      row("qualified", `${(tr.qualified || []).length} of ${n}`);
      $("truth-hint").innerHTML = tr.reason;
      $("truth-hint").className = "hint bad";
    } else {
      row("qualified", `${tr.qualified.length} of ${n}`);
      row("$i^*$", `<b>${tr.istar}</b> at ${tr.cStar.toFixed(1)} \\$`);
      row("runner-up", `${tr.cSecond.toFixed(1)} \\$ (gap $\\Delta = ${tr.deltaGap.toFixed(1)}$)`);
      row("cheapest overall", `${tr.cheapest}${tr.trap ? " — <b>unqualified</b>: a trap for greedy rules" : " (qualified)"}`);
      $("truth-hint").textContent = "";
      $("truth-hint").className = "hint";
    }

    // constants
    const p = pinned(),
      f = floorC(),
      cm = cMax();
    $("c_max_hint").innerHTML = n ? `pinned value ${p.toFixed(1)} \\$; validity floor ${f.toFixed(1)} \\$` + (cm < f - 1e-9 ? " — $C_{\\max}$ is below the floor" : "") : "";
    $("c_max_hint").className = cm < f - 1e-9 ? "hint bad" : "hint";
    const req = Live.requiredGamma(state.estimator, Math.max(n, 2), state.delta);
    if (state.estimator.kind === "shrinkage") {
      $("est_hint").innerHTML = `shrinkage with $k = ${state.estimator.k}$ needs $\\gamma \\ge ${req.toFixed(3)}$; $\\gamma$ is ${state.gamma}` + (state.gamma < req ? " — the radius is too narrow for these bids" : "");
      $("est_hint").className = state.gamma < req ? "hint warn" : "hint";
    } else if (state.estimator.kind === "biased") {
      $("est_hint").innerHTML = "no finite $\\gamma$ bounds a biased bidder's slack: the assumption is broken on purpose";
      $("est_hint").className = "hint warn";
    } else {
      $("est_hint").innerHTML = "honest empirical-mean bids need no widening: $\\gamma = 0$ is exact";
      $("est_hint").className = "hint";
    }

    // theory
    if (tr.valid && cm > 0) {
      const th = Live.theory(tr, n, state.delta, cm, state.gamma, horizon());
      $("theory").innerHTML = `Theory for this environment: $B_{id}$ allows ${fmtInt(th.bId)} rounds away from $i^*$. The identification certificate requires more than ${fmtInt(th.need)} rounds; one pass has ${fmtInt(horizon())}.`;
    } else $("theory").textContent = "";

    // run controls
    const t = horizon();
    $("t_max_hint").textContent = `of ${questions()} questions`;
    const problems = [];
    if (n < 2) problems.push("add at least two providers");
    else if (!tr.valid) problems.push(tr.reason);
    if (cm < f - 1e-9) problems.push("$C_{\\max}$ is below the validity floor");
    if (state.t_max !== null && (state.t_max > questions() || state.t_max < n)) problems.push(`rounds must lie between ${n} and ${questions()}`);
    if (!state.arms.length) problems.push("select at least one arm");
    const episodes = state.arms.length * state.repetitions;
    $("run-estimate").innerHTML = problems.length ? problems[0] : `${episodes} episodes of ${t} rounds`;
    $("run-estimate").className = problems.length ? "hint bad" : "hint";
    const canRun = engineReady && !problems.length && !jobs.size;
    $("btn-run").disabled = !canRun;
    $("btn-sweep").disabled = !canRun;

    if (full) {
      Plots.landscape("landscape", CATALOG, state.benchmark, state.roster, tr, state.margin, state.theta);
      if (n >= 2) Plots.radii("radii", tr, n, state.delta, cm, state.gamma, t);
    }
    renderStale();
    typeset($("config"));
  }

  // ---- picker -----------------------------------------------------------------------------

  function openPicker() {
    const sel = $("picker-base");
    sel.replaceChildren(el("option", { value: "", text: "all base models" }));
    for (const b of CATALOG[state.benchmark].models) sel.append(el("option", { value: b, text: b }));
    renderPicker();
    $("picker").showModal();
  }

  function renderPicker() {
    const c = CATALOG[state.benchmark].candidates.map((x) => ({ ...x, cost: Live.cost(x, state.margin) }));
    const front = new Set(Live.paretoFrontier(c).map((x) => x.name));
    const base = $("picker-base").value;
    const onlyFront = $("picker-frontier").checked;
    const rows = c.filter((x) => (!base || x.base === base) && (!onlyFront || front.has(x.name)));
    const table = $("picker-table");
    table.replaceChildren(el("tr", {}, ["", "provider", "N", "quality", "cost ($)", "frontier"].map((h) => el("th", { text: h }))));
    for (const x of rows) {
      const on = state.roster.includes(x.name);
      const cb = el("input", { type: "checkbox" });
      cb.checked = on;
      cb.addEventListener("change", () => {
        if (cb.checked && !state.roster.includes(x.name)) state.roster.push(x.name);
        if (!cb.checked) state.roster = state.roster.filter((m) => m !== x.name);
        tr.classList.toggle("on", cb.checked);
        $("picker-count").textContent = `${state.roster.length} on the roster`;
      });
      const tr = el("tr", { class: on ? "on" : "" }, [el("td", {}, [cb]), el("td", { text: x.name }), el("td", { text: String(x.n) }), el("td", { text: x.quality.toFixed(3) }), el("td", { text: x.cost.toFixed(1) }), el("td", { text: front.has(x.name) ? "●" : "" })]);
      table.append(tr);
    }
    $("picker-count").textContent = `${state.roster.length} on the roster`;
  }

  // ---- worker -------------------------------------------------------------------------------

  function setEngineStatus(cls, text) {
    const pill = $("engine-status");
    pill.className = "pill " + cls;
    pill.textContent = "engine: " + text;
  }

  function startWorker() {
    worker = new Worker("worker.js");
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "status") setEngineStatus("loading", msg.message);
      else if (msg.type === "ready") {
        engineReady = true;
        setEngineStatus("ready", "ready");
        refreshLive(false);
      } else if (msg.type === "progress") {
        const job = jobs.get(msg.id);
        if (job) job.onProgress(msg.done, msg.total, msg.label);
        setEngineStatus("loading", job && job.kind === "sweep" ? "sweeping" : "running");
      } else if (msg.type === "result") {
        const job = jobs.get(msg.id);
        jobs.delete(msg.id);
        if (job) job.resolve(msg.result);
      } else if (msg.type === "cancelled") {
        const job = jobs.get(msg.id);
        jobs.delete(msg.id);
        if (job) job.reject(new Error("cancelled"));
      } else if (msg.type === "error") {
        if (msg.id !== undefined && jobs.has(msg.id)) {
          const job = jobs.get(msg.id);
          jobs.delete(msg.id);
          job.reject(new Error(msg.message));
        } else setEngineStatus("error", msg.message);
      }
      if (["result", "cancelled", "error"].includes(msg.type) && engineReady) setEngineStatus("ready", "ready");
    };
    worker.onerror = (e) => setEngineStatus("error", e.message || "worker failed");
    worker.postMessage({ type: "init", base: new URL(DATA, location.href).href });
  }

  function submit(kind, request, thetas, onProgress) {
    const id = ++jobCounter;
    return new Promise((resolve, reject) => {
      jobs.set(id, { kind, resolve, reject, onProgress });
      worker.postMessage({ type: "start", id, kind, request, thetas });
    });
  }
  function cancelJob(kind) {
    for (const [id, job] of jobs) if (job.kind === kind) worker.postMessage({ type: "cancel", id });
  }

  async function runExperiment() {
    const request = buildRequest();
    $("run-error").textContent = "";
    $("btn-run").disabled = true;
    $("btn-cancel").hidden = false;
    $("progress").hidden = false;
    const started = performance.now();
    const bar = $("progress").firstElementChild;
    try {
      const result = await submit("run", request, null, (done, total, label) => {
        bar.style.width = `${(100 * done) / Math.max(1, total)}%`;
        $("progress-text").textContent = `${done} of ${total} episodes` + (label ? ` · ${Plots.plainLabel(label)}` : "") + ` · ${((performance.now() - started) / 1000).toFixed(0)} s`;
      });
      setResult(result);
    } catch (err) {
      $("run-error").textContent = err.message === "cancelled" ? "run cancelled" : err.message;
    } finally {
      $("btn-cancel").hidden = true;
      $("progress").hidden = true;
      $("progress-text").textContent = "";
      refreshLive(false);
    }
  }

  async function runSweep() {
    const c = cands();
    const points = Math.max(2, parseInt($("sweep-points").value) || 10);
    let thetas;
    if ($("sweep-mode").value === "midpoints") {
      thetas = Live.midpoints(c, state.roster);
      if (thetas.length > points) {
        const step = (thetas.length - 1) / (points - 1);
        thetas = Array.from({ length: points }, (_, i) => thetas[Math.round(i * step)]);
      }
    } else {
      const qs = state.roster.map((m) => c.get(m).quality);
      const lo = Math.min(...qs) + 0.005,
        hi = Math.max(...qs) - 0.005;
      thetas = Array.from({ length: points }, (_, i) => Math.round((lo + ((hi - lo) * i) / (points - 1)) * 1e4) / 1e4);
    }
    thetas = [...new Set(thetas)].filter((t) => Live.truth(c, state.roster, t, state.margin).valid);
    if (thetas.length < 2) {
      $("sweep-status").textContent = "fewer than two admissible thresholds for this roster";
      return;
    }
    const request = buildRequest({ repetitions: Math.max(1, parseInt($("sweep-reps").value) || 5), theta: thetas[0] });
    $("btn-sweep").disabled = true;
    $("btn-sweep-cancel").hidden = false;
    $("sweep-progress").hidden = false;
    const bar = $("sweep-progress").firstElementChild;
    const started = performance.now();
    try {
      SWEEP = await submit("sweep", request, thetas, (done, total, label) => {
        bar.style.width = `${(100 * done) / Math.max(1, total)}%`;
        $("sweep-status").textContent = `${done} of ${total} episodes · ${label} · ${((performance.now() - started) / 1000).toFixed(0)} s`;
      });
      $("sweep-status").textContent = `${SWEEP.points.length} thresholds × ${Object.keys(SWEEP.points[0].arms).length} arms × ${request.repetitions} repetitions on ${SWEEP.roster.length} providers (${SWEEP.request.benchmark}), ${SWEEP.seconds.toFixed(0)} s` + (SWEEP.skipped.length ? `; ${SWEEP.skipped.length} threshold(s) skipped` : "");
      $("sweep-card").hidden = false;
      Plots.sweep("plot-sweep", SWEEP);
    } catch (err) {
      $("sweep-status").textContent = err.message === "cancelled" ? "sweep cancelled" : err.message;
    } finally {
      $("btn-sweep-cancel").hidden = true;
      $("sweep-progress").hidden = true;
      refreshLive(false);
    }
  }

  // ---- results ------------------------------------------------------------------------------

  function setResult(result) {
    RESULT = result;
    for (const t of TABS) dirty.add(t);
    renderStatus();
    renderTab(activeTab);
  }

  function renderStatus() {
    const s = $("result-status");
    if (!RESULT) {
      s.textContent = "No results yet.";
      return;
    }
    const r = RESULT.request,
      env = RESULT.environment;
    const what = RESULT.precomputed ? "Precomputed run" : "Run";
    s.replaceChildren(
      el("span", { html: `<b>${what}</b>: ${env.benchmark}, ${env.roster.length} providers, $\\Theta = ${env.theta}$, ${r.repetitions} repetition${r.repetitions > 1 ? "s" : ""} of ${r.t_max} rounds, ${r.arms.length} arms` + (r.environment_id && PRESETS.environments.some((e) => e.id === r.environment_id) ? ` · preset <b>${r.environment_id}</b>` : "") + (RESULT.seconds ? ` · ${RESULT.seconds.toFixed(1)} s` : "") }),
      el("span", { html: `$i^*$ = <b>${env.istar}</b>, ${env.qualified.length} qualified, $C_{\\max} = ${env.c_max.toFixed(1)}$` + (env.trap ? ", cheapest overall is unqualified" : "") }),
    );
    renderStale();
    typeset($("result-status"));
  }

  function renderStale() {
    const n = $("stale-notice");
    const stale = isStale();
    n.hidden = !stale;
    if (stale) n.textContent = "The configuration has changed since these results were computed. Press Run to update them.";
  }

  function renderTab(tab) {
    if (!RESULT || !dirty.has(tab)) return;
    dirty.delete(tab);
    if (tab === "overview") renderOverview();
    else if (tab === "selection") renderSelection();
    else if (tab === "regret") renderRegret();
    else if (tab === "payments") renderPayments();
    else if (tab === "beliefs") renderBeliefs();
  }

  function fillArmSelect(select, names) {
    const previous = select.value;
    select.replaceChildren();
    for (const n of names) select.append(el("option", { value: n, text: Plots.plainLabel(n) }));
    select.value = names.includes(previous) ? previous : names.includes("mechanism") ? "mechanism" : names[0];
  }

  function renderOverview() {
    const cards = $("cards");
    cards.replaceChildren();
    for (const [name, s] of Object.entries(RESULT.summary)) {
      const info = RESULT.arms[name];
      const rows = [
        ["accuracy", pm(s.realized_accuracy)],
        ["cost", pm(s.total_cost, 0)],
        ["$i^*$ share", pm(s.istar_share)],
        ["collapsed", `${(100 * s.collapsed).toFixed(0)}% of episodes`],
      ];
      if (info.paid) rows.push(["payment", pm(s.total_payment, 0)]);
      if (s.good_event_holds !== null) rows.push(["good event", `${(100 * s.good_event_holds).toFixed(0)}% of episodes`]);
      cards.append(
        el("div", { class: "arm-card", style: `border-left-color:${ARM_COLORS[name]}` }, [
          el("h4", { text: Plots.armLabel(name) + (info.group === "ablation" ? " (ablation)" : "") }),
          el("p", { class: "short", text: CONTENT.arms[name].short }),
          el("table", {}, rows.map(([k, v]) => el("tr", {}, [el("td", { html: k }), el("td", { text: v })]))),
        ]),
      );
    }
    typeset($("cards"));
    Plots.overviewScatter("plot-overview-scatter", RESULT);
    Plots.overviewShare("plot-overview-share", RESULT);
  }

  function renderSelection(onlyArm = false) {
    if (!onlyArm) {
      fillArmSelect($("sel-arm"), Object.keys(RESULT.rounds));
      Plots.istarShare("plot-istar-share", RESULT);
      Plots.eligible("plot-eligible", RESULT);
    }
    Plots.providerShare("plot-provider-share", RESULT, $("sel-arm").value);
  }

  function renderRegret() {
    Plots.regret("plot-quality-regret", RESULT, "quality_regret", ["quality_anytime", "quality_gap"], "cumulative quality regret");
    Plots.regret("plot-generation-regret", RESULT, "generation_regret", ["generation_anytime", "generation_gap"], "cumulative cost regret ($)");
  }

  function renderPayments(onlyArm = false) {
    const paid = Object.keys(RESULT.arms).filter((a) => RESULT.arms[a].paid);
    $("pay-none").hidden = paid.length > 0;
    $("pay-body").style.display = paid.length ? "" : "none";
    if (!onlyArm) {
      const table = el("table", { class: "data" }, [el("tr", {}, ["arm", "payment rule", "total payment", "excess payment", "rounds with critical payment below bid"].map((h) => el("th", { text: h })))]);
      for (const [name, s] of Object.entries(RESULT.summary)) {
        table.append(el("tr", {}, [el("td", { text: Plots.armLabel(name) }), el("td", { text: RESULT.arms[name].payment || "unpaid" }), el("td", { text: pm(s.total_payment, 0) }), el("td", { text: pm(s.excess_payment, 0) }), el("td", { text: pm(s.critical_below_bid_rounds, 1) })]));
      }
      $("pay-table").replaceChildren(table);
      if (paid.length) fillArmSelect($("pay-arm"), paid);
    }
    if (!paid.length) return;
    const arm = $("pay-arm").value;
    const s = RESULT.summary[arm];
    $("pay-numbers").textContent = `total payment ${pm(s.total_payment, 0)} $, excess over the runner-up ${pm(s.excess_payment, 0)} $, lowest provider payoff ${pm(s.min_provider_payoff, 0)} $`;
    Plots.envelope("plot-envelope", RESULT, arm);
    const split = Plots.payoff("plot-payoff", RESULT, arm);
    $("payoff-numbers").textContent = `First repetition: ${split.negative} of ${split.rounds} rounds paid less than the realised cost; gained ${fmtInt(split.gain)}, lost ${fmtInt(-split.loss)}, net ${fmtInt(split.net)} $.`;
    Plots.payoffProviders("plot-payoff-providers", RESULT, arm);
  }

  function renderBeliefs(onlyArm = false) {
    if (!onlyArm) fillArmSelect($("bel-arm"), Object.keys(RESULT.beliefs));
    const arm = $("bel-arm").value;
    Plots.qucb("plot-qucb", RESULT, arm);
    Plots.slack("plot-slack", RESULT, arm);
  }

  // ---- share and export ---------------------------------------------------------------------

  async function shareLink() {
    writeHash();
    const url = location.href;
    try {
      await navigator.clipboard.writeText(url);
      $("btn-share").textContent = "Link copied";
      setTimeout(() => ($("btn-share").textContent = "Share link"), 1500);
    } catch (e) {
      window.prompt("Copy this link:", url);
    }
  }

  function exportResults() {
    if (!RESULT) return;
    const blob = new Blob([JSON.stringify({ run: RESULT, sweep: SWEEP })], { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: `cri-${RESULT.environment.benchmark}-theta${RESULT.environment.theta}.json` });
    document.body.append(a);
    a.click();
    a.remove();
  }

  // ---- boot ---------------------------------------------------------------------------------------

  async function boot() {
    try {
      [PRESETS, CATALOG] = await Promise.all([fetchJSON("presets.json"), fetchJSON("catalog.json")]);
    } catch (err) {
      $("result-error").hidden = false;
      $("result-error").textContent = `The page's data could not be loaded (${err.message}). Serve the app directory over HTTP after running web/build.py.`;
      return;
    }
    buildStaticUI();
    typeset();
    const fromHash = readHash();
    if (fromHash) Object.assign(state, fromHash);
    else {
      applyPreset(PRESETS.default_environment);
      state.arms = [...PRESETS.policies];
    }
    if (!state.arms.length) state.arms = [...PRESETS.policies];
    syncControls();
    refreshLive();
    startWorker();
    try {
      const preset = await fetchJSON("default.json");
      if (!fromHash) {
        // a fresh visit adopts the precomputed run's repetitions and arms, so nothing is stale
        state.repetitions = preset.request.repetitions;
        state.arms = [...preset.request.arms];
        syncControls();
        refreshLive(false);
      }
      setResult(preset);
    } catch (err) {
      $("result-status").textContent = "No precomputed run found; press Run once the engine is ready.";
    }
  }

  boot();
})();
