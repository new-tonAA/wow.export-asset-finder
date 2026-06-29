"""
WoW Asset Finder - Step 1: Extract feature vectors from wow.export model previews.

This script automates wow.export (nw.js app) via Selenium/ChromeDriver to:
1. Launch wow.export and connect to WoW CDN
2. Navigate to the Models tab
3. Iterate through all models in the listfile
4. For each model: load preview -> capture canvas screenshot -> extract CLIP feature vector
5. Save all vectors + paths to a .npz file

No thumbnails are stored on disk - images are processed in memory only.

Usage:
    python extract_features.py [--resume] [--limit N] [--batch-size N]

Prerequisites:
    - wow.export debug build at C:/wow.export/wow.export/bin/win-x64-debug/
    - pip install -r requirements.txt
    - Must manually connect to a WoW CDN source in wow.export first time
"""

import os
import sys
import time
import json
import argparse
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Selenium for controlling nw.js via chromedriver
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# CLIP for feature extraction
import torch
import open_clip


# --- Config ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_DIR, "config.json")

def _load_config():
    """Load wow.export path from config.json, or auto-detect."""
    # 1. Try config.json
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
            path = cfg.get("wow_export_dir", "")
            if path and os.path.isdir(path):
                return path

    # 2. Try common locations
    candidates = [
        os.path.join(PROJECT_DIR, "wow.export"),  # Alongside this project
        r"C:\wow.export\wow.export\bin\win-x64-debug",
        os.path.join(os.environ.get("USERPROFILE", ""), "wow.export", "wow.export", "bin", "win-x64-debug"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "nw.exe")):
            return c

    # 3. Not found - will fail at connect time with clear error
    return ""

WOW_EXPORT_DIR = _load_config()
NW_EXE = os.path.join(WOW_EXPORT_DIR, "nw.exe")
CHROMEDRIVER_EXE = os.path.join(WOW_EXPORT_DIR, "chromedriver.exe")
OUTPUT_DIR = PROJECT_DIR
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wow_model_features.npz")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

# How long to wait for a model to load and render (seconds)
MODEL_LOAD_TIMEOUT = 15
# Minimum wait after model load for render to complete
RENDER_MIN_WAIT = 0.5
# Max time to wait for isBusy to clear
BUSY_WAIT_TIMEOUT = 30
# Canvas screenshot size (will be resized for CLIP)
CLIP_IMAGE_SIZE = 224
# Minimum pixel variance to consider an image non-empty
# wow.export grid-only background is ~37, so 40 filters empty without killing dark models
MIN_IMAGE_VARIANCE = 40.0


def load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load CLIP model for feature extraction."""
    print(f"Loading CLIP model on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()
    print("CLIP model loaded.")
    return model, preprocess, device


def extract_feature_vector(image_bytes, model, preprocess, device):
    """Extract CLIP feature vector from PNG image bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def connect_to_wow_export():
    """
    Launch wow.export via chromedriver and return the Selenium driver.
    nw.js ships with its own chromedriver that speaks the same protocol.
    """
    print("Connecting to wow.export via ChromeDriver...")

    if not WOW_EXPORT_DIR or not os.path.isfile(NW_EXE):
        raise FileNotFoundError(
            f"wow.export not found. Please set the path in config.json.\n"
            f"  Expected: nw.exe at '{NW_EXE}'\n"
            f"  Copy config.example.json to config.json and update the path."
        )

    service = Service(executable_path=CHROMEDRIVER_EXE)

    options = Options()
    options.binary_location = NW_EXE
    # nw.js specific: point to the app directory
    options.add_argument(f"--nwapp={WOW_EXPORT_DIR}")
    options.add_argument("--disable-gpu-sandbox")
    options.add_argument("--no-sandbox")
    # Use a dedicated user data dir inside the project to avoid lock conflicts
    user_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wow_export_data")
    os.makedirs(user_data_dir, exist_ok=True)
    options.add_argument(f"--user-data-dir={user_data_dir}")

    driver = webdriver.Chrome(service=service, options=options)
    # Set script timeout to 120s (some models take a long time to download from CDN)
    driver.set_script_timeout(120)
    print("wow.export launched successfully.")
    return driver


def wait_for_app_ready(driver, timeout=120):
    """Wait for wow.export to finish initialization."""
    print("Waiting for wow.export to initialize...")

    # nw.js debug builds open DevTools in a separate window.
    # ChromeDriver may initially attach to the wrong one.
    # Switch to the actual wow.export app window.
    import time as _time
    for _ in range(10):
        handles = driver.window_handles
        for handle in handles:
            driver.switch_to.window(handle)
            if "devtools://" not in driver.current_url:
                break
        if "devtools://" not in driver.current_url:
            break
        _time.sleep(1)

    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script(
            "return typeof core !== 'undefined' && core.view !== undefined"
        )
    )
    print("App initialized.")


def wait_for_casc_ready(driver, timeout=120):
    """Wait for the user to connect to a CASC source (CDN)."""
    print("Waiting for CASC source connection...")
    print(">>> Please connect to a WoW CDN source in the wow.export window <<<")
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script(
            "return core.view.casc !== null && core.view.casc !== undefined"
        )
    )
    print("CASC source connected.")


def get_model_list(driver):
    """Get the full list of model paths from the listfile."""
    print("Retrieving model list from listfile...")
    model_list = driver.execute_script("""
        const models = core.view.listfileModels;
        if (!models || models.length === 0) return null;
        // Each entry is like "path/to/model.m2 [12345]" or just "path/to/model.m2"
        return models.map(entry => {
            // Strip the [fileDataId] suffix if present
            const match = entry.match(/^(.+?)\\s*(?:\\[\\d+\\])?$/);
            return match ? match[1].trim() : entry.trim();
        });
    """)

    if not model_list:
        print("ERROR: No models found. Is the CASC source connected and Models tab loaded?")
        return []

    print(f"Found {len(model_list)} models in listfile.")
    return model_list


def switch_to_models_tab(driver):
    """Navigate to the Models tab in wow.export."""
    print("Switching to Models tab...")
    driver.execute_script("""
        const modules = require('./js/modules');
        modules.set_active('tab_models');
    """)
    time.sleep(2)


def load_model_and_capture(driver, model_path, timeout=MODEL_LOAD_TIMEOUT):
    """
    Load a model in wow.export and capture the canvas as PNG bytes.
    Returns PNG bytes or None if failed or model is empty.
    """
    # Trigger model preview via internal function
    success = driver.execute_script(f"""
        return (async () => {{
            try {{
                // Disable background and grid for clean capture
                core.view.config.modelViewerShowBackground = false;
                core.view.config.modelViewerShowGrid = false;

                const listfile = require('./js/casc/listfile');
                const file_data_id = listfile.getByFilename("{model_path}");
                if (!file_data_id) return 'no_id';

                const file = await core.view.casc.getFile(file_data_id);
                if (!file) return 'no_file';

                const modelViewerUtils = require('./js/ui/model-viewer-utils');
                const gl_context = core.view.modelViewerContext?.gl_context;
                if (!gl_context) return 'no_gl';

                const model_type = modelViewerUtils.detect_model_type_by_name("{model_path}")
                    ?? modelViewerUtils.detect_model_type(file);

                // Dispose previous renderer
                if (window.__autoRenderer) {{
                    window.__autoRenderer.dispose();
                    window.__autoRenderer = null;
                }}

                const renderer = modelViewerUtils.create_renderer(
                    file, model_type, gl_context,
                    true, "{model_path}"
                );
                await renderer.load();

                // Check if model has actual geometry
                const has_content = (renderer.draw_calls && renderer.draw_calls.length > 0)
                    || (renderer.groups && renderer.groups.length > 0);
                if (!has_content) return 'empty_model';

                window.__autoRenderer = renderer;

                // Set as active renderer so the render loop draws it
                core.view.modelViewerContext.getActiveRenderer = () => renderer;

                // Fit camera
                if (core.view.modelViewerContext.fitCamera)
                    core.view.modelViewerContext.fitCamera();

                // Wait for one frame to actually render
                await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

                return 'ok';
            }} catch (e) {{
                return 'error:' + e.message;
            }}
        }})();
    """)

    if not success or not success.startswith("ok"):
        # Log skip reason
        skip_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features", "skip_log.txt")
        with open(skip_log, "a", encoding="utf-8") as f:
            f.write(f"{model_path}\t{success}\n")
        return None

    # Wait for isBusy to clear (model fully loaded including async textures)
    for _ in range(int(BUSY_WAIT_TIMEOUT / 0.5)):
        is_busy = driver.execute_script("return core.view.isBusy > 0")
        if not is_busy:
            break
        time.sleep(0.5)

    # Extra frame wait for render stability
    time.sleep(RENDER_MIN_WAIT)

    # Capture canvas as base64 PNG
    canvas_data = driver.execute_script("""
        const canvas = document.querySelector('.gl-canvas');
        if (!canvas) return null;
        return canvas.toDataURL('image/png');
    """)

    if not canvas_data:
        return None

    # Strip data URL prefix "data:image/png;base64,"
    base64_str = canvas_data.split(",", 1)[1]
    png_bytes = base64.b64decode(base64_str)

    # Use alpha channel to mask out background
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        img_array = np.array(img)
        alpha = img_array[:, :, 3]

        # If almost no visible pixels, model is empty
        visible_pixels = (alpha > 10).sum()
        total_pixels = alpha.shape[0] * alpha.shape[1]
        if visible_pixels < total_pixels * 0.01:
            skip_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features", "skip_log.txt")
            with open(skip_log, "a", encoding="utf-8") as f:
                f.write(f"{model_path}\tblank_image (visible={visible_pixels}/{total_pixels})\n")
            return None

        # Composite onto white background for CLIP (CLIP expects RGB)
        rgb = img_array[:, :, :3].astype(np.float32)
        alpha_f = (alpha / 255.0)[:, :, np.newaxis]
        white_bg = np.ones_like(rgb) * 255.0
        composited = (rgb * alpha_f + white_bg * (1 - alpha_f)).astype(np.uint8)

        # Re-encode as PNG bytes for CLIP processing
        comp_img = Image.fromarray(composited, "RGB")
        buf = io.BytesIO()
        comp_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

    except Exception:
        return None

    return png_bytes


def save_progress(index, paths, vectors):
    """Save current progress to allow resume."""
    np.savez_compressed(
        OUTPUT_FILE,
        paths=np.array(paths, dtype=object),
        vectors=np.array(vectors, dtype=np.float32),
    )
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_index": index, "total_extracted": len(paths)}, f)


def load_progress():
    """Load previous progress if resuming."""
    if not os.path.exists(PROGRESS_FILE) or not os.path.exists(OUTPUT_FILE):
        return 0, [], []

    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)

    data = np.load(OUTPUT_FILE, allow_pickle=True)
    paths = data["paths"].tolist()
    vectors = data["vectors"].tolist()

    print(f"Resuming from index {progress['last_index']}, {len(paths)} vectors already extracted.")
    return progress["last_index"] + 1, paths, vectors


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features from WoW models via wow.export")
    parser.add_argument("--resume", action="store_true", help="Resume from last saved progress")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of models to process (0=all)")
    parser.add_argument("--batch-size", type=int, default=100, help="Save progress every N models")
    args = parser.parse_args()

    # Load CLIP model
    clip_model, clip_preprocess, device = load_clip_model()

    # Load progress if resuming
    start_index = 0
    extracted_paths = []
    extracted_vectors = []
    if args.resume:
        start_index, extracted_paths, extracted_vectors = load_progress()

    # Connect to wow.export
    driver = connect_to_wow_export()

    try:
        # Wait for app to be ready
        wait_for_app_ready(driver)

        # Wait for CASC connection (user must connect manually first time)
        wait_for_casc_ready(driver)

        # Switch to models tab and get model list
        switch_to_models_tab(driver)
        time.sleep(3)  # Let the tab fully load

        model_list = get_model_list(driver)
        if not model_list:
            print("No models found. Exiting.")
            return

        # Apply limit
        if args.limit > 0:
            model_list = model_list[: args.limit]

        # Main extraction loop
        total = len(model_list)
        failed_count = 0
        pbar = tqdm(range(start_index, total), initial=start_index, total=total, desc="Extracting features")

        for i in pbar:
            model_path = model_list[i]
            pbar.set_postfix_str(f"{os.path.basename(model_path)[:30]}", refresh=False)

            # Load model and capture screenshot
            png_bytes = load_model_and_capture(driver, model_path)

            if png_bytes is None:
                failed_count += 1
                pbar.set_postfix_str(f"SKIP (failed:{failed_count})", refresh=True)
                continue

            # Extract CLIP feature vector (in memory, no disk write)
            try:
                vector = extract_feature_vector(png_bytes, clip_model, clip_preprocess, device)
                extracted_paths.append(model_path)
                extracted_vectors.append(vector)
            except Exception as e:
                failed_count += 1
                continue

            # Periodic save
            if len(extracted_paths) % args.batch_size == 0:
                save_progress(i, extracted_paths, extracted_vectors)
                pbar.set_postfix_str(f"saved ({len(extracted_paths)} vectors)", refresh=True)

        # Final save
        save_progress(total - 1, extracted_paths, extracted_vectors)
        print(f"\nDone! Extracted {len(extracted_paths)} feature vectors. Failed: {failed_count}")
        print(f"Saved to: {OUTPUT_FILE}")

    finally:
        # Clean up renderer
        try:
            driver.execute_script("""
                if (window.__autoRenderer) {
                    window.__autoRenderer.dispose();
                    window.__autoRenderer = null;
                }
            """)
        except:
            pass
        driver.quit()


if __name__ == "__main__":
    main()
