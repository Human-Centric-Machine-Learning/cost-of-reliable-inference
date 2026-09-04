// Run one engine request under Pyodide in Node, the runtime the page uses.
//
//   PYODIDE_MODULE=/path/to/node_modules/pyodide/pyodide.mjs \
//   node web/tools/pyodide_run.mjs web/app/data request.json out.json
//
// Loads engine.zip and <benchmark>.npz from the data directory exactly as the
// page's worker does, runs Engine.run, and writes the JSON result.
import fs from "node:fs";
import path from "node:path";

const { loadPyodide } = await import(process.env.PYODIDE_MODULE ?? "pyodide");
const [dataDir, requestPath, outPath] = process.argv.slice(2);
if (!outPath) {
  console.error("usage: pyodide_run.mjs <data dir> <request.json> <out.json>");
  process.exit(2);
}
const request = JSON.parse(fs.readFileSync(requestPath, "utf8"));

const t0 = Date.now();
const py = await loadPyodide();
await py.loadPackage("numpy");
const zip = fs.readFileSync(path.join(dataDir, "engine.zip"));
py.unpackArchive(new Uint8Array(zip.buffer, zip.byteOffset, zip.byteLength), "zip", { extractDir: "/engine" });
py.FS.mkdir("/data");
const pack = `${request.benchmark}.npz`;
const packBytes = fs.readFileSync(path.join(dataDir, pack));
py.FS.writeFile(`/data/${pack}`, new Uint8Array(packBytes.buffer, packBytes.byteOffset, packBytes.byteLength));
console.error(`pyodide ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

py.globals.set("REQUEST_JSON", JSON.stringify(request));
py.globals.set("PACK_PATH", `/data/${pack}`);
const t1 = Date.now();
const out = py.runPython(`
import json, sys, time
sys.path.insert(0, "/engine")
from engine.api import Engine
engine = Engine.from_packs({json.loads(REQUEST_JSON)["benchmark"]: PACK_PATH})
result = engine.run(json.loads(REQUEST_JSON))
result["backend"] = {"runtime": "pyodide", "python": sys.version, "numpy": __import__("numpy").__version__}
json.dumps(result)
`);
console.error(`run finished in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
fs.writeFileSync(outPath, out);
