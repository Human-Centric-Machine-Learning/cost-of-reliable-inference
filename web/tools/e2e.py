#!/usr/bin/env python3
"""Drive the built page in a headless browser: load, run, sweep, switch presets.

    pip install playwright && playwright install chromium
    python web/tools/e2e.py [--screenshots DIR]

Serves web/app on a local port, opens it in Chromium, waits for the engine
(the runtime comes from a CDN), runs a small experiment and a small sweep,
visits every tab, and fails on any page error. With --screenshots each tab is
captured for inspection.
"""

from __future__ import annotations

import argparse
import http.server
import socket
import sys
import threading
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
APP = ROOT / "web" / "app"


def serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(APP))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screenshots", type=Path, help="directory for one screenshot per tab")
    ap.add_argument("--engine-timeout", type=int, default=240, help="seconds to wait for the engine")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    if not (APP / "data" / "engine.zip").is_file():
        print("build the assets first: python web/build.py", file=sys.stderr)
        return 2
    server, port = serve()
    url = f"http://127.0.0.1:{port}/"
    errors: list[str] = []
    shots = args.screenshots
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    def shot(name: str) -> None:
        if shots:
            page.screenshot(path=str(shots / f"{name}.png"), full_page=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
        page.on("console", lambda m: errors.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
        page.goto(url)

        # precomputed default run is shown before the engine is ready
        page.wait_for_selector("#cards .arm-card", timeout=30_000)
        assert page.locator("#cards .arm-card").count() == 8, "expected 8 arm cards from default.json"
        assert "Precomputed run" in page.inner_text("#result-status")
        assert page.locator("#stale-notice").is_hidden(), "the default configuration must not be stale"
        assert page.locator("#preset").input_value() == "GSM8K-ladder"
        print("default run rendered")
        shot("01-overview-default")

        page.wait_for_function("document.getElementById('engine-status').textContent.includes('ready')", timeout=args.engine_timeout * 1000)
        print("engine ready")

        # a small run: three arms, one repetition, 200 rounds
        for name in ["uniform_random", "quality_greedy", "oracle_cheapest_qualified", "no_quality_filter", "greedy_cheapest_qualified"]:
            page.uncheck(f"#arm-{name}")
        page.fill("#repetitions", "1")
        page.press("#repetitions", "Tab")
        page.fill("#t_max", "200")
        page.press("#t_max", "Tab")
        assert page.locator("#stale-notice").is_visible(), "a changed configuration must be flagged"
        page.click("#btn-run")
        page.wait_for_function("document.getElementById('result-status').textContent.startsWith('Run:')", timeout=120_000)
        assert page.locator("#stale-notice").is_hidden()
        assert page.locator("#cards .arm-card").count() == 3
        assert page.inner_text("#run-error").strip() == ""
        print("run finished:", page.inner_text("#result-status").splitlines()[0])
        shot("02-overview-run")

        for tab in ["selection", "regret", "payments", "beliefs"]:
            page.click(f".tab[data-tab={tab}]")
            page.wait_for_selector(f"#tab-{tab} .plot .main-svg", timeout=20_000)
            shot(f"03-{tab}")
        print("tabs render")

        # payments: switch the paid arm
        page.click(".tab[data-tab=payments]")
        page.select_option("#pay-arm", "pay_your_bid")
        page.wait_for_timeout(500)
        assert "pay your bid" in page.inner_text("#pay-table").lower()

        # a small sweep
        page.click(".tab[data-tab=robustness]")
        page.fill("#sweep-points", "3")
        page.fill("#sweep-reps", "1")
        page.click("#btn-sweep")
        page.wait_for_selector("#sweep-card:not([hidden]) .main-svg", timeout=180_000)
        print("sweep finished:", page.inner_text("#sweep-status"))
        shot("04-robustness")

        # an invalid threshold disables Run with a reason
        page.fill("#theta-number", "0.95")
        page.press("#theta-number", "Tab")
        assert page.locator("#btn-run").is_disabled()
        assert "qualif" in page.inner_text("#run-estimate")
        page.click("#theta-prev")
        assert not page.locator("#btn-run").is_disabled()

        # the roster picker and a shipped roster
        page.click("#btn-add-provider")
        page.wait_for_selector("#picker[open]")
        rows = page.locator("#picker-table tr").count()
        assert rows == 9 * 7 + 1, f"expected every candidate in the picker, got {rows - 1}"
        page.click("#picker-done")
        page.click("#roster-presets button:nth-child(1)")  # the full roster
        assert page.locator("#roster-chips .chip").count() == 9
        shot("05-config-full-roster")

        # another benchmark through a preset: the worker fetches its pack on demand
        page.select_option("#preset", "GPQA-full")  # one full pass, so the run counts as the preset
        page.uncheck("#arm-pay_your_bid")
        page.click(".tab[data-tab=overview]")
        page.click("#btn-run")
        page.wait_for_function("document.getElementById('result-status').textContent.includes('GPQA')", timeout=120_000)
        assert "preset GPQA-full" in page.inner_text("#result-status")
        print("second benchmark ran:", page.inner_text("#result-status").splitlines()[0])
        shot("06-gpqa")

        # share link reproduces the configuration
        page.click("#btn-share")
        href = page.evaluate("location.href")
        page2 = browser.new_page(viewport={"width": 1500, "height": 1000})
        page2.on("pageerror", lambda e: errors.append(f"page2 error: {e}"))
        page2.goto(href)
        page2.wait_for_selector("#roster-chips .chip", timeout=30_000)
        assert page2.locator("#preset").input_value() == "GPQA-full"
        assert page2.locator("#stale-notice").is_visible()
        print("share link restores the configuration")
        browser.close()
    server.shutdown()

    if errors:
        print("\n".join(errors))
        print(f"{len(errors)} browser error(s)")
        return 1
    print("OK: the page loads, runs, sweeps and navigates without errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
