"""Capture genuine full-page screenshots of the running CASSAVISION Streamlit application."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = ROOT_DIR / "assets" / "screenshots"


def main() -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    app_path = ROOT_DIR / "app" / "streamlit_app.py"
    port = 8519

    cmd = [
        str(venv_python),
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"Starting Streamlit server on port {port}...")
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        import urllib.request
        url = f"http://localhost:{port}"
        for _ in range(30):
            try:
                urllib.request.urlopen(url, timeout=2)
                print("Streamlit server is ready!")
                break
            except Exception:
                time.sleep(1.0)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(viewport={"width": 1480, "height": 2050})
            page = context.new_page()

            url = f"http://localhost:{port}"
            print(f"Navigating to {url}...", flush=True)
            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            # Wait for prediction and Grad-CAM images to finish rendering on Tab 1
            page.wait_for_selector("text=Visual Explainability & Spatial Saliency Inspection", timeout=60000)
            time.sleep(3.5)

            # 1. Screenshot Tab 1: Classification & Grad-CAM Studio
            p1 = SCREENSHOTS_DIR / "01_classification_and_gradcam.png"
            page.screenshot(path=str(p1))
            print(f"Saved {p1.name} ({p1.stat().st_size} bytes)", flush=True)

            # 2. Switch to Tab 2: Model Evaluation Centre
            page.get_by_text("2. Model Evaluation Centre", exact=False).first.click()
            time.sleep(2.5)
            p2 = SCREENSHOTS_DIR / "03_model_evaluation_centre.png"
            page.screenshot(path=str(p2))
            print(f"Saved {p2.name} ({p2.stat().st_size} bytes)", flush=True)

            # 3. Switch to Tab 3: Explainability & Dataset Audit
            page.get_by_text("3. Explainability & Dataset Audit", exact=False).first.click()
            time.sleep(2.5)
            p3 = SCREENSHOTS_DIR / "04_explainability_and_dataset_audit.png"
            page.screenshot(path=str(p3))
            print(f"Saved {p3.name} ({p3.stat().st_size} bytes)", flush=True)

            # 4. Switch to Tab 4: Model Card & Session History
            page.get_by_text("4. Model Card & Session History", exact=False).first.click()
            page.wait_for_selector("text=Model Card & System Specification", timeout=30000)
            time.sleep(2.0)
            p4 = SCREENSHOTS_DIR / "05_model_card_and_session_history.png"
            page.screenshot(path=str(p4))
            print(f"Saved {p4.name} ({p4.stat().st_size} bytes)", flush=True)

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("Streamlit server stopped cleanly.")


if __name__ == "__main__":
    main()
