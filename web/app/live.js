// Pure functions the page evaluates instantly from the catalog, without the
// engine: ground truth for a roster and threshold, the radii, the theory's
// counts, and the constants pinned to the price scale. They mirror the
// formulas in cri/; web/tools/live_check.mjs compares them with the engine.

const Live = (() => {
  function candidates(catalog, benchmark) {
    const map = new Map();
    for (const c of catalog[benchmark].candidates) map.set(c.name, c);
    return map;
  }

  function cost(cand, margin) {
    return (cand.mean_tokens * cand.price_per_1M) / (1 + margin);
  }

  function pinnedCmax(cands, roster, tokenCap) {
    return tokenCap * Math.max(...roster.map((m) => cands.get(m).rate));
  }

  function floorCmax(cands, roster, tokenCap, margin) {
    return pinnedCmax(cands, roster, tokenCap) / (1 + margin);
  }

  // Ground truth as cri.metrics.ground_truth computes it; valid=false carries the reason.
  function truth(cands, roster, theta, margin) {
    const q = {},
      c = {};
    for (const m of roster) {
      const cand = cands.get(m);
      if (!cand) return { valid: false, reason: `no data for ${m} on this benchmark` };
      q[m] = cand.quality;
      c[m] = cost(cand, margin);
    }
    const qualified = roster.filter((m) => q[m] >= theta);
    if (qualified.length < 2) {
      return { valid: false, reason: `only ${qualified.length} provider(s) reach $\\Theta = ${theta}$; at least two must qualify`, q, c, qualified };
    }
    const istar = qualified.reduce((a, b) => (c[b] < c[a] ? b : a));
    const contenders = qualified.filter((m) => m !== istar);
    if (contenders.some((m) => Math.abs(c[m] - c[istar]) <= 1e-12 * Math.abs(c[istar]))) {
      return { valid: false, reason: "two qualified providers share the lowest cost", q, c, qualified };
    }
    const cSecond = Math.min(...contenders.map((m) => c[m]));
    const cheapest = roster.reduce((a, b) => (c[b] < c[a] ? b : a));
    const eps = {},
      deltaJ = {};
    for (const m of roster) if (q[m] < theta) eps[m] = theta - q[m];
    for (const m of contenders) deltaJ[m] = c[m] - c[istar];
    return {
      valid: true,
      theta,
      q,
      c,
      qualified,
      istar,
      cStar: c[istar],
      cSecond,
      deltaGap: cSecond - c[istar],
      eps,
      deltaJ,
      cheapest,
      trap: !qualified.includes(cheapest),
    };
  }

  function Lambda(m, n, delta) {
    return Math.log((2 * Math.PI * Math.PI * n * m * m) / (3 * delta));
  }
  function beta(m, n, delta) {
    return Math.sqrt(Lambda(m, n, delta) / (2 * m));
  }
  function rho(m, n, delta, cMax, gamma) {
    return (1 + gamma) * cMax * beta(m, n, delta);
  }

  function Mcount(eps, n, delta) {
    return Math.ceil((4 / (eps * eps)) * Math.log((57 * n) / (delta * eps ** 4)));
  }
  function Lcount(gap, n, delta, cMax, gamma) {
    const g = 1 + gamma;
    return Math.ceil(((4 * g * g * cMax * cMax) / (gap * gap)) * Math.log((57 * n * g ** 4 * cMax ** 4) / (delta * gap ** 4)));
  }

  function theory(tr, n, delta, cMax, gamma, horizon) {
    let sumM = 0,
      sumL = 0;
    for (const e of Object.values(tr.eps)) sumM += Mcount(e, n, delta);
    for (const g of Object.values(tr.deltaJ)) sumL += Lcount(g, n, delta, cMax, gamma);
    const bId = sumM + sumL;
    return { sumM, sumL, bId, need: n + 2 * bId, reachable: n + 2 * bId < horizon };
  }

  // Smallest gamma that bounds the shrinkage estimator's slack, as cri.confidence computes it.
  function gammaMinShrinkage(k, n, delta, mMax = 200000) {
    let worst = 0;
    for (let m = 1; m <= mMax; m++) {
      const v = k / ((m + k) * beta(m, n, delta));
      if (v > worst) worst = v;
    }
    return worst;
  }
  function requiredGamma(est, n, delta) {
    if (est.kind === "shrinkage") return gammaMinShrinkage(est.k, n, delta);
    if (est.kind === "biased") return Infinity;
    return 0;
  }

  // Midpoints between adjacent roster qualities, the natural thresholds.
  function midpoints(cands, roster, minGap = 0.01) {
    const qs = roster.map((m) => cands.get(m).quality).sort((a, b) => a - b);
    const out = [];
    for (let i = 0; i + 1 < qs.length; i++) if (qs[i + 1] - qs[i] > minGap) out.push(Math.round(((qs[i] + qs[i + 1]) / 2) * 1e4) / 1e4);
    return out;
  }

  function paretoFrontier(list) {
    const sorted = [...list].sort((a, b) => a.cost - b.cost);
    let best = -1;
    const front = [];
    for (const p of sorted)
      if (p.quality > best) {
        front.push(p);
        best = p.quality;
      }
    return front;
  }

  return { candidates, cost, pinnedCmax, floorCmax, truth, Lambda, beta, rho, Mcount, Lcount, theory, requiredGamma, midpoints, paretoFrontier };
})();

if (typeof module !== "undefined") module.exports = Live;
