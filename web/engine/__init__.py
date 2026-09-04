"""The simulation engine behind the interactive companion page.

This JSON layer over ``cri`` is independent of the user interface and runs
under both CPython and Pyodide.

    pack.py   compact binary benchmarks (the page fetches these)
    api.py    catalog, describe, run and sweep requests
"""
