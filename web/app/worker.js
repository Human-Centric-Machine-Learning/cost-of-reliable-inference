// The simulation worker: loads Pyodide and the engine, fetches benchmarks on
// demand, and runs jobs one episode at a time so the page can show progress
// and cancel between episodes. Messages:
//   in : {type:"init", base}  {type:"start", id, kind:"run"|"sweep", request, thetas}  {type:"cancel", id}
//   out: {type:"status", message}  {type:"ready"}  {type:"progress", id, done, total, label}
//        {type:"result", id, result}  {type:"cancelled", id}  {type:"error", id, message}

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE_URL + "pyodide.js");

const BOOT = `
import json, sys, traceback
sys.path.insert(0, "/engine")
from engine.api import Engine, EngineError
ENGINE = Engine.from_packs({})
JOBS = {}

def _ok(**kw):
    return json.dumps({"ok": True, **kw})

def _err(exc):
    if isinstance(exc, EngineError):
        return json.dumps({"ok": False, "error": str(exc)})
    last = "".join(traceback.format_exception(exc)).strip().splitlines()[-1]
    return json.dumps({"ok": False, "error": "internal error: " + last})

def add_pack(benchmark, path):
    ENGINE.add_pack(benchmark, path)

def start(job_id, kind, request_json, thetas_json):
    try:
        request = json.loads(request_json)
        JOBS[job_id] = ENGINE.run_job(request) if kind == "run" else ENGINE.sweep_job(request, json.loads(thetas_json))
        return _ok(total=JOBS[job_id].total)
    except Exception as exc:
        return _err(exc)

def step(job_id):
    try:
        job = JOBS[job_id]
        more = job.step()
        return _ok(more=more, done=job.done, total=job.total, label=job.label)
    except Exception as exc:
        JOBS.pop(job_id, None)
        return _err(exc)

def finish(job_id):
    try:
        return _ok(result=JOBS.pop(job_id).result())
    except Exception as exc:
        return _err(exc)

def cancel(job_id):
    JOBS.pop(job_id, None)
`;

let py = null;
let base = "";
const fn = {};
const packs = new Set();
const cancelled = new Set();
let busy = false;

function status(message) {
  postMessage({ type: "status", message });
}

async function init(dataBase) {
  base = dataBase;
  try {
    status("loading the Python runtime");
    py = await loadPyodide({ indexURL: PYODIDE_URL });
    status("loading numpy");
    await py.loadPackage("numpy");
    status("loading the engine");
    const zip = await (await fetch(base + "engine.zip")).arrayBuffer();
    py.unpackArchive(zip, "zip", { extractDir: "/engine" });
    py.runPython(BOOT);
    for (const name of ["add_pack", "start", "step", "finish", "cancel"]) fn[name] = py.globals.get(name);
    postMessage({ type: "ready" });
  } catch (err) {
    postMessage({ type: "error", message: "the engine could not be loaded: " + (err && err.message ? err.message : err) });
  }
}

async function ensurePack(benchmark) {
  if (packs.has(benchmark)) return;
  status(`loading the ${benchmark} benchmark`);
  const bytes = await (await fetch(`${base}${benchmark}.npz`)).arrayBuffer();
  try {
    py.FS.mkdir("/data");
  } catch (e) {
    /* exists */
  }
  const path = `/data/${benchmark}.npz`;
  py.FS.writeFile(path, new Uint8Array(bytes));
  fn.add_pack(benchmark, path);
  packs.add(benchmark);
}

function call(name, ...args) {
  const out = JSON.parse(fn[name](...args));
  if (!out.ok) throw new Error(out.error);
  return out;
}

async function run(msg) {
  const { id, kind, request, thetas } = msg;
  busy = true;
  try {
    await ensurePack(request.benchmark);
    const started = call("start", id, kind, JSON.stringify(request), JSON.stringify(thetas || []));
    postMessage({ type: "progress", id, done: 0, total: started.total, label: "" });
    for (;;) {
      if (cancelled.has(id)) {
        cancelled.delete(id);
        fn.cancel(id);
        postMessage({ type: "cancelled", id });
        return;
      }
      const s = call("step", id);
      postMessage({ type: "progress", id, done: s.done, total: s.total, label: s.label });
      if (!s.more) break;
      await new Promise((resolve) => setTimeout(resolve, 0)); // let a cancel message through
    }
    const finished = call("finish", id);
    postMessage({ type: "result", id, result: finished.result });
  } catch (err) {
    postMessage({ type: "error", id, message: err && err.message ? err.message : String(err) });
  } finally {
    busy = false;
  }
}

onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "init") init(msg.base);
  else if (msg.type === "start") {
    if (busy) postMessage({ type: "error", id: msg.id, message: "a job is already running" });
    else run(msg);
  } else if (msg.type === "cancel") cancelled.add(msg.id);
};
