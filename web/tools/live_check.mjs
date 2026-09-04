// Compare the page's instant calculations (web/app/live.js) with the engine.
//   node web/tools/live_check.mjs web/app/data/catalog.json describe.json
// where describe.json is Engine.describe(request) for the same roster; the
// test in tests/test_web_engine.py drives this.
import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const Live = require("../app/live.js");
const [catalogPath, describePath] = process.argv.slice(2);
const catalog = JSON.parse(fs.readFileSync(catalogPath, "utf8"));
const d = JSON.parse(fs.readFileSync(describePath, "utf8"));
const env = d.environment,
  req = d.request;

const cands = Live.candidates(catalog, env.benchmark);
const tr = Live.truth(cands, env.roster, env.theta, env.margin);
const problems = [];
const close = (a, b, tol, what) => {
  if (Math.abs(a - b) > tol * Math.max(1, Math.abs(b))) problems.push(`${what}: ${a} vs ${b}`);
};
if (!tr.valid) problems.push("live truth invalid: " + tr.reason);
else {
  if (tr.istar !== env.istar) problems.push(`i*: ${tr.istar} vs ${env.istar}`);
  if (JSON.stringify(tr.qualified) !== JSON.stringify(env.qualified)) problems.push("qualified set differs");
  for (const m of env.roster) {
    close(tr.q[m], env.q[m], 2e-3, `q[${m}]`); // catalog qualities use the reward-aligned questions
    close(tr.c[m], env.c[m], 2e-3, `c[${m}]`);
  }
  close(Live.pinnedCmax(cands, env.roster, req.token_cap), env.pinned_c_max, 1e-9, "pinned C_max");
  close(Live.floorCmax(cands, env.roster, req.token_cap, env.margin), env.c_max_floor, 1e-9, "C_max floor");
  const th = Live.theory(tr, env.n, env.delta, env.c_max, env.gamma, env.questions);
  close(th.bId, env.theory.B_id, 2e-2, "B_id"); // gaps differ slightly with the qualities
  for (let i = 0; i < env.radius_table.m.length; i += 20) {
    const m = env.radius_table.m[i];
    close(Live.beta(m, env.n, env.delta), env.radius_table.beta[i], 2e-6, `beta(${m})`); // the table is rounded to six decimals
    close(Live.rho(m, env.n, env.delta, env.c_max, env.gamma), env.radius_table.rho[i], 2e-6, `rho(${m})`);
  }
  if (env.required_gamma !== null) close(Live.requiredGamma(env.estimator, env.n, env.delta), env.required_gamma, 1e-6, "required gamma");
}
if (problems.length) {
  console.log(problems.join("\n"));
  process.exit(1);
}
console.log("OK: live.js agrees with the engine");
