// Plotly figures built from an engine result. Each function draws into a
// container id. Colours are fixed per arm so an arm looks the same everywhere.

const ARM_COLORS = {
  mechanism: "#1f6fb4",
  uniform_random: "#c9312c",
  cheapest_bid: "#2e9e4f",
  quality_greedy: "#8e5bc4",
  oracle_cheapest_qualified: "#7a5230",
  no_quality_filter: "#d65db1",
  greedy_cheapest_qualified: "#f28c1e",
  pay_your_bid: "#1aa8b8",
  gamma_zero: "#a3a51a",
  biased_beliefs: "#6f6f6f",
  independent_cost_stream: "#7fb3e0",
  cheapest_price: "#93d68f",
  oracle_quality_random: "#c3aee6",
};

const Plots = (() => {
  const FONT = { family: "Inter, system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: "#2b2f36" };
  const BASE = {
    margin: { l: 58, r: 16, t: 34, b: 48 },
    font: FONT,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#ffffff",
    hovermode: "closest",
    legend: { font: { size: 10 }, bgcolor: "rgba(255,255,255,0.7)" },
    xaxis: { gridcolor: "#eef0f3", zerolinecolor: "#dde1e6" },
    yaxis: { gridcolor: "#eef0f3", zerolinecolor: "#dde1e6" },
  };
  const CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

  function draw(id, traces, layout) {
    const el = document.getElementById(id);
    if (!el) return;
    const merged = { ...BASE, ...layout, xaxis: { ...BASE.xaxis, ...(layout.xaxis || {}) }, yaxis: { ...BASE.yaxis, ...(layout.yaxis || {}) } };
    Plotly.react(el, traces, merged, CONFIG);
  }

  function armLabel(name) {
    return CONTENT.arms[name] ? CONTENT.arms[name].label : name;
  }
  // Labels for places that cannot render LaTeX: select options, hover text, progress lines.
  function plainLabel(name) {
    return armLabel(name).replace(/\$/g, "").replace(/\\/g, "");
  }
  function armColor(name) {
    return ARM_COLORS[name] || "#444";
  }

  // Provider colours: i* green, other qualified blues, unqualified oranges, shaded by rank.
  function providerColors(roster, env) {
    const qualified = roster.filter((m) => env.qualified.includes(m) && m !== env.istar).sort((a, b) => env.c[a] - env.c[b]);
    const unqualified = roster.filter((m) => !env.qualified.includes(m)).sort((a, b) => env.q[b] - env.q[a]);
    const out = {};
    qualified.forEach((m, i) => (out[m] = `hsl(212, 60%, ${38 + (28 * i) / Math.max(1, qualified.length - 1)}%)`));
    unqualified.forEach((m, i) => (out[m] = `hsl(28, 85%, ${45 + (25 * i) / Math.max(1, unqualified.length - 1)}%)`));
    if (env.istar) out[env.istar] = "#2e9e4f";
    return out;
  }

  const fmt = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));

  // ---- configuration panel ---------------------------------------------------

  function landscape(id, catalog, benchmark, roster, tr, margin, theta) {
    const all = catalog[benchmark].candidates.map((c) => ({ ...c, cost: Live.cost(c, margin) }));
    const front = Live.paretoFrontier(all);
    const onRoster = new Set(roster);
    const others = all.filter((c) => !onRoster.has(c.name));
    const mine = all.filter((c) => onRoster.has(c.name));
    const color = (c) => (!tr.valid ? "#5b8def" : c.name === tr.istar ? "#2e9e4f" : tr.qualified.includes(c.name) ? "#3b7dd8" : "#f28c1e");
    const traces = [
      { x: others.map((c) => c.cost), y: others.map((c) => c.quality), text: others.map((c) => c.name), mode: "markers", type: "scatter", name: "other candidates", marker: { color: "#c9ced6", size: 6 }, hovertemplate: "%{text}<br>cost %{x:.1f}, quality %{y:.3f}<extra></extra>" },
      { x: front.map((c) => c.cost), y: front.map((c) => c.quality), mode: "lines", type: "scatter", name: "frontier", line: { color: "#9aa3ad", dash: "dash", width: 1 }, hoverinfo: "skip" },
      { x: mine.map((c) => c.cost), y: mine.map((c) => c.quality), text: mine.map((c) => c.name), mode: "markers", type: "scatter", name: "roster", marker: { color: mine.map(color), size: 10, line: { color: "#fff", width: 1 } }, hovertemplate: "%{text}<br>cost %{x:.1f}, quality %{y:.3f}<extra></extra>" },
    ];
    draw(id, traces, {
      margin: { l: 44, r: 8, t: 8, b: 38 },
      showlegend: false,
      height: 230,
      xaxis: { title: { text: "cost ($ per query)", standoff: 4 }, type: "log" },
      yaxis: { title: { text: "quality", standoff: 4 }, range: [0, 1.02] },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: theta, y1: theta, line: { color: "#2b2f36", dash: "dot", width: 1 } }],
    });
  }

  function radii(id, tr, n, delta, cMax, gamma, horizon) {
    const ms = [];
    for (let e = 0; e <= 6; e += 0.05) ms.push(Math.round(10 ** e));
    const uniq = [...new Set(ms)];
    const traces = [{ x: uniq, y: uniq.map((m) => Live.beta(m, n, delta)), mode: "lines", name: "$\\beta(m)$", line: { color: "#2b2f36", width: 2.5 }, hovertemplate: "m = %{x}<br>beta = %{y:.3f}<extra></extra>" }];
    const lines = [];
    if (tr.valid) {
      for (const [m, e] of Object.entries(tr.eps)) lines.push({ name: m, level: e / 2, color: "#f28c1e", what: "half the quality gap" });
      for (const [m, g] of Object.entries(tr.deltaJ)) lines.push({ name: m, level: g / (2 * (1 + gamma) * cMax), color: "#3b7dd8", what: "half the cost gap, in radius units" });
    }
    for (const l of lines) {
      const cross = uniq.find((m) => Live.beta(m, n, delta) < l.level);
      traces.push({ x: [1, 1e6], y: [l.level, l.level], mode: "lines", line: { color: l.color, width: 1, dash: "dot" }, showlegend: false, hoverinfo: "skip" });
      traces.push({ x: [cross || 1e6], y: [l.level], mode: "markers", marker: { color: l.color, size: 7 }, showlegend: false, text: [l.name], hovertemplate: `%{text}<br>${l.what} = %{y:.3f}<br>resolved after about %{x} selections<extra></extra>` });
    }
    draw(id, traces, {
      margin: { l: 44, r: 8, t: 8, b: 38 },
      height: 230,
      showlegend: false,
      xaxis: { title: { text: "selections m", standoff: 4 }, type: "log" },
      yaxis: { title: { text: "radius", standoff: 4 }, type: "log" },
      shapes: [{ type: "line", yref: "paper", x0: horizon, x1: horizon, y0: 0, y1: 1, line: { color: "#9aa3ad", width: 1 } }],
    });
  }

  // ---- overview ----------------------------------------------------------------

  function overviewScatter(id, result) {
    const traces = [];
    for (const [name, s] of Object.entries(result.summary)) {
      traces.push({
        x: [s.total_cost.mean],
        y: [s.realized_accuracy.mean],
        error_x: { type: "data", array: [s.total_cost.sd], visible: true, thickness: 1 },
        error_y: { type: "data", array: [s.realized_accuracy.sd], visible: true, thickness: 1 },
        mode: "markers",
        type: "scatter",
        name: armLabel(name),
        marker: { color: armColor(name), size: 11 },
        hovertemplate: `${plainLabel(name)}<br>cost %{x:,.0f} $<br>accuracy %{y:.3f}<extra></extra>`,
      });
    }
    draw(id, traces, {
      showlegend: true,
      legend: { font: { size: 10 }, orientation: "h", y: -0.25 },
      height: 380,
      xaxis: { title: "total cost of the episode ($)" },
      yaxis: { title: "accuracy", range: [0, 1.02] },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: result.environment.theta, y1: result.environment.theta, line: { color: "#2b2f36", dash: "dot", width: 1 } }],
    });
  }

  function overviewShare(id, result) {
    const env = result.environment;
    const roster = env.roster;
    const colors = providerColors(roster, env);
    const arms = Object.keys(result.summary);
    const order = [...roster].sort((a, b) => env.q[a] - env.q[b]);
    const traces = order.map((m) => ({
      type: "bar",
      orientation: "h",
      name: m,
      y: arms.map(armLabel),
      x: arms.map((a) => result.summary[a].selection_share[roster.indexOf(m)]),
      marker: { color: colors[m], line: { color: "#fff", width: 0.5 } },
      hovertemplate: `${m}${m === env.istar ? " (i*)" : env.qualified.includes(m) ? " (qualified)" : " (unqualified)"}<br>%{x:.1%} of rounds<extra></extra>`,
    }));
    draw(id, traces, {
      barmode: "stack",
      height: 60 + 34 * arms.length,
      margin: { l: 170, r: 16, t: 10, b: 40 },
      showlegend: false,
      xaxis: { title: "share of rounds", range: [0, 1], tickformat: ".0%" },
      yaxis: { autorange: "reversed" },
    });
  }

  // ---- selection ---------------------------------------------------------------

  function istarShare(id, result) {
    const traces = Object.entries(result.series).map(([name, s]) => ({
      x: s.t,
      y: s.istar_share,
      mode: "lines",
      name: armLabel(name),
      line: { color: armColor(name), width: name === "mechanism" ? 3 : 1.5 },
      hovertemplate: `${plainLabel(name)}<br>round %{x}<br>i* share %{y:.3f}<extra></extra>`,
    }));
    draw(id, traces, { height: 340, xaxis: { title: "round t", type: "log" }, yaxis: { title: "$\\text{share of rounds given to } i^*$", range: [0, 1.02] } });
  }

  function providerShare(id, result, arm) {
    const env = result.environment;
    const r = result.rounds[arm];
    const roster = env.roster;
    const colors = providerColors(roster, env);
    const t = r.t;
    const traces = roster.map((m, j) => {
      let count = 0;
      const y = r.winner.map((w, i) => {
        if (w === j) count += 1;
        return count / t[i];
      });
      const istar = m === env.istar;
      return { x: t, y, mode: "lines", name: m + (istar ? " (i*)" : env.qualified.includes(m) ? "" : " (unqualified)"), line: { color: colors[m], width: istar ? 3 : 1.3 }, hovertemplate: `${m}<br>round %{x}<br>share %{y:.3f}<extra></extra>` };
    });
    draw(id, traces, { height: 340, xaxis: { title: "round t", type: "log" }, yaxis: { title: "running selection share", range: [0, 1.02] } });
  }

  function eligible(id, result) {
    const env = result.environment;
    const traces = Object.entries(result.rounds).map(([name, r]) => ({
      x: r.t,
      y: r.n_eligible,
      mode: "lines",
      line: { shape: "hv", color: armColor(name), width: name === "mechanism" ? 3 : 1.3 },
      name: armLabel(name),
      hovertemplate: `${plainLabel(name)}<br>round %{x}<br>%{y} eligible<extra></extra>`,
    }));
    draw(id, traces, {
      height: 300,
      xaxis: { title: "round t", type: "log" },
      yaxis: { title: "eligible providers", rangemode: "tozero" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: env.qualified.length, y1: env.qualified.length, line: { color: "#2b2f36", dash: "dot", width: 1 } }],
    });
  }

  // ---- regret --------------------------------------------------------------------

  function regret(id, result, key, boundKeys, title) {
    const traces = [];
    for (const [name, s] of Object.entries(result.series)) {
      traces.push({ x: s.t, y: s[key].map((v) => (v > 0 ? v : null)), mode: "lines", name: armLabel(name), line: { color: armColor(name), width: name === "mechanism" ? 3 : 1.5 }, hovertemplate: `${plainLabel(name)}<br>round %{x}<br>%{y:,.1f}<extra></extra>` });
    }
    const ref = result.series.mechanism || Object.values(result.series)[0];
    traces.push({ x: ref.t, y: ref[boundKeys[0]], mode: "lines", name: "bound (any horizon)", line: { color: "#2b2f36", dash: "dash", width: 1 }, hovertemplate: "round %{x}<br>bound %{y:,.0f}<extra></extra>" });
    traces.push({ x: ref.t, y: ref[boundKeys[1]], mode: "lines", name: "bound (gap-dependent)", line: { color: "#2b2f36", dash: "dot", width: 1 }, hovertemplate: "round %{x}<br>bound %{y:,.0f}<extra></extra>" });
    draw(id, traces, { height: 360, xaxis: { title: "round t", type: "log" }, yaxis: { title, type: "log" } });
  }

  // ---- payments -------------------------------------------------------------------

  function envelope(id, result, arm) {
    const env = result.environment;
    const r = result.rounds[arm];
    const main = r.t.map((_, i) => i).filter((i) => i >= r.init_rounds);
    const t = main.map((i) => r.t[i]);
    const cTrue = main.map((i) => env.c[env.roster[r.winner[i]]]);
    const lower = main.map((i, k) => Math.max(0, cTrue[k] - r.rho_m[i]));
    const upper = main.map((i) => Math.min(env.c_max, env.c_second + r.rho_m[i]));
    const traces = [
      { x: t, y: upper, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x: t, y: lower, mode: "lines", fill: "tonexty", fillcolor: "rgba(31,111,180,0.12)", line: { width: 0 }, name: "allowed band", hoverinfo: "skip" },
      { x: t, y: main.map((i) => r.payment[i]), mode: "lines", name: "payment", line: { color: armColor(arm), width: 1 }, hovertemplate: "round %{x}<br>paid %{y:.1f}<extra></extra>" },
      { x: [t[0], t[t.length - 1]], y: [env.c_star, env.c_star], mode: "lines", name: "$i^* \\text{ cost}$", line: { color: "#2e9e4f", dash: "dot", width: 1 }, hoverinfo: "skip" },
      { x: [t[0], t[t.length - 1]], y: [env.c_second, env.c_second], mode: "lines", name: "second qualified cost", line: { color: "#2b2f36", dash: "dot", width: 1 }, hoverinfo: "skip" },
    ];
    draw(id, traces, { height: 340, xaxis: { title: "round t", type: "log" }, yaxis: { title: "payment ($)", rangemode: "tozero" } });
  }

  function payoffSplit(result, arm) {
    const env = result.environment;
    const r = result.rounds[arm];
    const payoff = r.payment.map((p, i) => (p === null ? null : p - r.cost[i]));
    const perProvider = env.roster.map(() => ({ gain: 0, loss: 0 }));
    let net = 0,
      gain = 0,
      loss = 0;
    const cumNet = [],
      cumGain = [],
      cumLoss = [];
    let negative = 0;
    payoff.forEach((v, i) => {
      if (v !== null) {
        net += v;
        if (v >= 0) gain += v;
        else {
          loss += v;
          negative += 1;
        }
        if (v >= 0) perProvider[r.winner[i]].gain += v;
        else perProvider[r.winner[i]].loss += v;
      }
      cumNet.push(net);
      cumGain.push(gain);
      cumLoss.push(loss);
    });
    return { t: r.t, cumNet, cumGain, cumLoss, perProvider, negative, rounds: payoff.length, gain, loss, net };
  }

  function payoff(id, result, arm) {
    const s = payoffSplit(result, arm);
    const traces = [
      { x: s.t, y: s.cumNet, mode: "lines", name: "net", line: { color: "#2b2f36", width: 2.5 } },
      { x: s.t, y: s.cumGain, mode: "lines", name: "gains (payment ≥ cost)", line: { color: "#2e9e4f", width: 1.2 } },
      { x: s.t, y: s.cumLoss, mode: "lines", name: "losses (payment < cost)", line: { color: "#c9312c", width: 1.2 } },
    ];
    draw(id, traces, { height: 320, xaxis: { title: "round t", type: "log" }, yaxis: { title: "cumulative payoff ($)" } });
    return s;
  }

  function payoffProviders(id, result, arm) {
    const env = result.environment;
    const s = payoffSplit(result, arm);
    const rows = env.roster.map((m, j) => ({ m, gain: s.perProvider[j].gain, loss: s.perProvider[j].loss, net: s.perProvider[j].gain + s.perProvider[j].loss })).sort((a, b) => a.net - b.net);
    const traces = [
      { type: "bar", orientation: "h", y: rows.map((r) => r.m), x: rows.map((r) => r.gain), name: "gains", marker: { color: "#2e9e4f" }, hovertemplate: "%{y}<br>gains %{x:,.0f}<extra></extra>" },
      { type: "bar", orientation: "h", y: rows.map((r) => r.m), x: rows.map((r) => r.loss), name: "losses", marker: { color: "#c9312c" }, hovertemplate: "%{y}<br>losses %{x:,.0f}<extra></extra>" },
      { type: "scatter", mode: "markers", y: rows.map((r) => r.m), x: rows.map((r) => r.net), name: "net", marker: { color: "#2b2f36", size: 8 }, hovertemplate: "%{y}<br>net %{x:,.0f}<extra></extra>" },
    ];
    draw(id, traces, { barmode: "overlay", height: 60 + 26 * rows.length, margin: { l: 150, r: 16, t: 10, b: 40 }, xaxis: { title: "payoff ($)" } });
  }

  // ---- beliefs -----------------------------------------------------------------------

  function qucb(id, result, arm) {
    const env = result.environment;
    const b = result.beliefs[arm];
    const colors = providerColors(env.roster, env);
    const traces = env.roster.map((m, j) => ({
      x: b.t,
      y: b.q_ucb.map((row) => row[j]),
      mode: "lines",
      name: `${m} (q = ${env.q[m].toFixed(2)})`,
      line: { color: colors[m], width: m === env.istar ? 3 : 1.3 },
      hovertemplate: `${m}<br>round %{x}<br>optimistic quality %{y:.3f}<extra></extra>`,
    }));
    draw(id, traces, {
      height: 360,
      xaxis: { title: "round t", type: "log" },
      yaxis: { title: "optimistic quality estimate", range: [0, 1.05] },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: env.theta, y1: env.theta, line: { color: "#2b2f36", dash: "dot", width: 1 } }],
    });
  }

  function slack(id, result, arm) {
    const env = result.environment;
    const s = result.slack[arm];
    const gamma = result.arms[arm].gamma;
    const colors = providerColors(env.roster, env);
    const traces = env.roster.map((m) => ({ x: s[m].m, y: s[m].ratio, mode: "lines", name: m, line: { color: colors[m], width: 1.2 }, hovertemplate: `${m}<br>selection %{x}<br>slack ratio %{y:.3f}<extra></extra>` }));
    draw(id, traces, {
      height: 320,
      xaxis: { title: "selection count m", type: "log" },
      yaxis: { title: "belief slack ratio", rangemode: "tozero" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: gamma, y1: gamma, line: { color: "#2b2f36", dash: "dot", width: 1 } }],
    });
  }

  // ---- threshold sweep ------------------------------------------------------------------

  function sweep(id, sw) {
    const panels = [
      ["istar_share", "$i^* \\text{ share}$", "y", "x"],
      ["realized_accuracy", "accuracy", "y2", "x2"],
      ["total_cost", "cost ($)", "y3", "x3"],
      ["collapsed", "collapse rate", "y4", "x4"],
    ];
    const arms = Object.keys(sw.points[0].arms);
    const traces = [];
    panels.forEach(([key, , yaxis, xaxis], p) => {
      arms.forEach((name, k) => {
        const thetas = sw.points.map((pt) => pt.theta);
        const mean = sw.points.map((pt) => {
          const v = pt.arms[name][key].map(Number);
          return v.reduce((a, b) => a + b, 0) / v.length;
        });
        traces.push({ x: thetas, y: mean, mode: "lines+markers", name: armLabel(name), legendgroup: name, showlegend: p === 0, line: { color: armColor(name), width: name === "mechanism" ? 3 : 1.5 }, marker: { size: 5 }, xaxis, yaxis, hovertemplate: `${plainLabel(name)}<br>Theta = %{x}<br>%{y:.3f}<extra></extra>` });
        if (key !== "collapsed") {
          const xs = [],
            ys = [];
          sw.points.forEach((pt) => pt.arms[name][key].forEach((v) => {
            xs.push(pt.theta + (k - arms.length / 2) * 0.003);
            ys.push(v);
          }));
          traces.push({ x: xs, y: ys, mode: "markers", legendgroup: name, showlegend: false, marker: { color: armColor(name), size: 4, opacity: 0.35 }, xaxis, yaxis, hoverinfo: "skip" });
        }
      });
    });
    const thetas = sw.points.map((pt) => pt.theta);
    const annotations = sw.points.map((pt) => ({ x: pt.theta, y: 1, xref: "x", yref: "paper", yanchor: "bottom", text: String(pt.qualified), showarrow: false, font: { size: 9, color: "#6b7280" } }));
    const layout = {
      height: 620,
      grid: { rows: 2, columns: 2, pattern: "independent", roworder: "top to bottom" },
      margin: { l: 58, r: 16, t: 40, b: 48 },
      annotations,
      legend: { orientation: "h", y: -0.1 },
    };
    panels.forEach(([key, title, yaxis, xaxis]) => {
      const yk = yaxis === "y" ? "yaxis" : "yaxis" + yaxis.slice(1);
      const xk = xaxis === "x" ? "xaxis" : "xaxis" + xaxis.slice(1);
      layout[yk] = { title: { text: title }, gridcolor: "#eef0f3", ...(key === "istar_share" || key === "collapsed" ? { range: [-0.03, 1.03] } : key === "realized_accuracy" ? { range: [0, 1.03] } : {}) };
      layout[xk] = { title: { text: "$\\text{threshold } \\Theta$" }, gridcolor: "#eef0f3", tickvals: thetas };
    });
    draw(id, traces, layout);
  }

  return { draw, armLabel, plainLabel, armColor, providerColors, fmt, landscape, radii, overviewScatter, overviewShare, istarShare, providerShare, eligible, regret, envelope, payoff, payoffProviders, payoffSplit, qucb, slack, sweep };
})();
