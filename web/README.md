# Interactive companion

A static page for configuring an environment, comparing the mechanism with
baselines and ablations, and inspecting the results. It runs the `cri`
package in the browser through Pyodide, so nothing is installed beyond the
Python environment of the repository.

The overview reports generation cost, query price and amount paid separately:
the amount paid is the critical payment for the mechanism and the query price
for the listed-price baselines. Oracles, further baselines and the remaining
ablations are under the collapsed "More baselines" section.

```
web/
  engine/     the JSON API over cri (api.py) and the binary benchmark packs (pack.py)
  app/        the static site
  app/data/   built assets: benchmark packs, catalog, presets, engine.zip, a precomputed default run
  build.py    builds app/data/ from datasets/, rewards/ and configs/default.toml
  tools/      checks: Pyodide against CPython, live.js against the engine, a headless browser run
```

## Run locally

Build the datasets first if `datasets/` and `rewards/` are not present (see the
repository README), then:

```bash
python web/build.py                       # writes web/app/data/, about 6 MB
python -m http.server -d web/app 8000     # open http://localhost:8000
```

The first visit downloads the Python runtime and numpy from a CDN (about
12 MB, cached by the browser). The page shows the precomputed default run
immediately and enables Run once the engine is ready. Build again whenever
`cri/`, `configs/default.toml` or the data change; `build.py --repetitions N`
sets the repetitions of the precomputed run.

## Changing the text

Every explanation, tooltip, caption and glossary entry lives in
`web/app/content.js`. Nothing else reads it.
