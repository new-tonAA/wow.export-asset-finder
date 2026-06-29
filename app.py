"""
WoW Asset Finder - Web UI

Flask + SocketIO backend providing:
- Image search (CLIP semantic similarity)
- Text search (CLIP text-image matching)
- Color search (HSV histogram similarity)
- Combined mode with adjustable weights
- Real-time thumbnail generation from wow.export
- Feature vector extraction with live progress

Usage:
    python app.py
    Open http://localhost:5000 in browser
"""

import os
import sys
import time
import json
import base64
import io
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

import numpy as np
import faiss
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

import torch
import open_clip

from extract_features import (
    connect_to_wow_export,
    wait_for_app_ready,
    wait_for_casc_ready,
    get_model_list,
    switch_to_models_tab,
    load_model_and_capture,
)

# --- App Setup ---
app = Flask(__name__)
app.config["SECRET_KEY"] = "wow-asset-finder-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

import threading

# --- Global State ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

clip_model = None
clip_preprocess = None
clip_device = None

faiss_index_clip = None
faiss_index_color = None
model_paths = None

driver = None
driver_lock = None  # Not needed with gevent (single-threaded)
driver_connected = False

extraction_running = False
_stop_extraction = False

# Color histogram config
COLOR_HIST_BINS = (12, 8, 8)  # H, S, V bins
COLOR_VECTOR_DIM = 12 * 8 * 8  # 768-dim


def get_clip():
    global clip_model, clip_preprocess, clip_device
    if clip_model is None:
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on {clip_device}...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        clip_model = clip_model.to(clip_device)
        clip_model.eval()
        print("CLIP model loaded.")
    return clip_model, clip_preprocess, clip_device


def extract_color_histogram(image_bytes):
    """Extract normalized HSV color histogram from image bytes, ignoring transparent/white background."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img_array = np.array(image)

    # Use alpha channel to mask out background if available
    alpha = img_array[:, :, 3]
    mask = alpha > 10  # Only count visible pixels

    if mask.sum() < 100:
        # Almost no visible pixels, return zero vector
        return np.zeros(COLOR_VECTOR_DIM, dtype=np.float32)

    # Extract only visible pixels
    r = img_array[:, :, 0][mask] / 255.0
    g = img_array[:, :, 1][mask] / 255.0
    b = img_array[:, :, 2][mask] / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    hmask = diff != 0
    mask_r = hmask & (cmax == r)
    mask_g = hmask & (cmax == g)
    mask_b = hmask & (cmax == b)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

    # Saturation & Value
    s = np.where(cmax == 0, 0, diff / cmax)
    v = cmax

    # Quantize to bins
    h_bins = np.clip((h / 360.0 * COLOR_HIST_BINS[0]).astype(int), 0, COLOR_HIST_BINS[0] - 1)
    s_bins = np.clip((s * COLOR_HIST_BINS[1]).astype(int), 0, COLOR_HIST_BINS[1] - 1)
    v_bins = np.clip((v * COLOR_HIST_BINS[2]).astype(int), 0, COLOR_HIST_BINS[2] - 1)

    # Build histogram
    flat_idx = h_bins * (COLOR_HIST_BINS[1] * COLOR_HIST_BINS[2]) + s_bins * COLOR_HIST_BINS[2] + v_bins
    hist = np.bincount(flat_idx.flatten(), minlength=COLOR_VECTOR_DIM).astype(np.float32)

    # L2 normalize for cosine similarity via inner product
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm

    return hist


def get_npz_path(category):
    """Get the npz file path for a specific category."""
    return os.path.join(FEATURES_DIR, f"features_{category}.npz")


def get_progress_path(category):
    """Get the progress file path for a specific category."""
    return os.path.join(FEATURES_DIR, f"progress_{category}.json")


def load_faiss_index():
    """Load or reload FAISS indices from all npz files in features dir."""
    global faiss_index_clip, faiss_index_color, model_paths

    all_paths = []
    all_clip_vectors = []
    all_color_vectors = []

    import glob
    npz_files = glob.glob(os.path.join(FEATURES_DIR, "features_*.npz"))

    if not npz_files:
        faiss_index_clip = None
        faiss_index_color = None
        model_paths = None
        return 0

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        all_paths.extend(data["paths"].tolist())
        all_clip_vectors.append(data["vectors"].astype(np.float32))
        if "color_vectors" in data:
            all_color_vectors.append(data["color_vectors"].astype(np.float32))

    model_paths = np.array(all_paths, dtype=object)
    clip_vectors = np.vstack(all_clip_vectors)

    dim_clip = clip_vectors.shape[1]
    faiss_index_clip = faiss.IndexFlatIP(dim_clip)
    faiss_index_clip.add(clip_vectors)

    if all_color_vectors and sum(len(c) for c in all_color_vectors) == len(model_paths):
        color_vectors = np.vstack(all_color_vectors)
        dim_color = color_vectors.shape[1]
        faiss_index_color = faiss.IndexFlatIP(dim_color)
        faiss_index_color.add(color_vectors)
        print(f"Loaded FAISS indices: {len(model_paths)} models from {len(npz_files)} files (CLIP {dim_clip}D + Color {dim_color}D)")
    else:
        faiss_index_color = None
        print(f"Loaded FAISS index: {len(model_paths)} models from {len(npz_files)} files (CLIP {dim_clip}D)")

    return len(model_paths)


def extract_image_features(image_bytes):
    model, preprocess, device = get_clip()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().astype(np.float32)


def extract_text_features(text):
    model, _, device = get_clip()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().astype(np.float32)


def search_index(query_vector, index, top_k=12):
    if index is None or model_paths is None:
        return []
    query_vector = query_vector.reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0:
            continue
        results.append({
            "rank": i + 1,
            "path": str(model_paths[idx]),
            "similarity": round(float(dist), 4),
        })
    return results


def capture_thumbnail(model_path):
    global driver
    if driver is None:
        return None
    png_bytes = load_model_and_capture(driver, model_path)
    if png_bytes is None:
        return None
    return base64.b64encode(png_bytes).decode("utf-8")


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    has_image = "image" in request.files and request.files["image"].filename != ""
    text_query = request.form.get("text", "").strip()
    search_mode = request.form.get("mode", "semantic")  # semantic, color, text, combined

    if not has_image and not text_query:
        return jsonify({"error": "Please provide an image or text description"}), 400

    try:
        image_bytes = None
        if has_image:
            image_bytes = request.files["image"].read()

        if search_mode == "color":
            if not image_bytes:
                return jsonify({"error": "Color mode requires an image"}), 400
            if faiss_index_color is None:
                return jsonify({"error": "Color index not available. Re-run extraction to build it."}), 400
            query_vector = extract_color_histogram(image_bytes)
            results = search_index(query_vector, faiss_index_color)

        elif search_mode == "text":
            if not text_query:
                return jsonify({"error": "Text mode requires a description"}), 400
            query_vector = extract_text_features(text_query)
            results = search_index(query_vector, faiss_index_clip)

        elif search_mode == "combined":
            vectors = []
            weights = []

            if image_bytes:
                vectors.append(extract_image_features(image_bytes))
                weights.append(float(request.form.get("w_semantic", "0.5")))

            if text_query:
                vectors.append(extract_text_features(text_query))
                weights.append(float(request.form.get("w_text", "0.3")))

            if image_bytes and faiss_index_color is not None:
                color_vec = extract_color_histogram(image_bytes)
                # Project color to CLIP dimension for combined search
                # We search CLIP index with semantic+text blend only
                pass

            if not vectors:
                return jsonify({"error": "Combined mode needs at least one input"}), 400

            # Weighted sum in CLIP space
            total_w = sum(weights)
            query_vector = sum(v * w for v, w in zip(vectors, weights)) / total_w
            query_vector = query_vector / np.linalg.norm(query_vector)
            results = search_index(query_vector, faiss_index_clip)

        else:  # semantic (default)
            if image_bytes:
                query_vector = extract_image_features(image_bytes)
            else:
                query_vector = extract_text_features(text_query)
            results = search_index(query_vector, faiss_index_clip)

    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

    return jsonify({
        "results": results,
        "index_size": len(model_paths) if model_paths is not None else 0,
        "has_color_index": faiss_index_color is not None,
    })


@app.route("/api/status")
def api_status():
    import glob
    index_size = len(model_paths) if model_paths is not None else 0
    npz_files = glob.glob(os.path.join(FEATURES_DIR, "features_*.npz"))

    # Collect per-category progress
    categories_done = {}
    for f in npz_files:
        name = os.path.basename(f).replace("features_", "").replace(".npz", "")
        data = np.load(f, allow_pickle=True)
        categories_done[name] = len(data["paths"])

    return jsonify({
        "index_size": index_size,
        "has_npz": len(npz_files) > 0,
        "has_color_index": faiss_index_color is not None,
        "driver_connected": driver_connected,
        "extraction_running": extraction_running,
        "categories": categories_done,
    })


@app.route("/api/delete_index", methods=["POST"])
def api_delete_index():
    global faiss_index_clip, faiss_index_color, model_paths
    try:
        import glob
        for f in glob.glob(os.path.join(FEATURES_DIR, "features_*.npz")):
            os.remove(f)
        for f in glob.glob(os.path.join(FEATURES_DIR, "progress_*.json")):
            os.remove(f)
        faiss_index_clip = None
        faiss_index_color = None
        model_paths = None
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["GET"])
def api_config_get():
    config_path = os.path.join(PROJECT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"wow_export_dir": ""})


@app.route("/api/config", methods=["POST"])
def api_config_set():
    import extract_features
    try:
        data = request.get_json()
        config_path = os.path.join(PROJECT_DIR, "config.json")
        with open(config_path, "w") as f:
            json.dump(data, f, indent=4)
        # Reload the path in extract_features module
        new_dir = data.get("wow_export_dir", "")
        extract_features.WOW_EXPORT_DIR = new_dir
        extract_features.NW_EXE = os.path.join(new_dir, "nw.exe")
        extract_features.CHROMEDRIVER_EXE = os.path.join(new_dir, "chromedriver.exe")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- WebSocket Events ---

@socketio.on("request_thumbnails")
def handle_thumbnail_request(data):
    paths = data.get("paths", [])
    if driver is None:
        emit("thumbnail_error", {"error": "wow.export not connected"})
        return
    for i, model_path in enumerate(paths):
        try:
            b64_png = capture_thumbnail(model_path)
            emit("thumbnail_result", {"index": i, "path": model_path, "image": b64_png})
        except Exception as e:
            emit("thumbnail_result", {"index": i, "path": model_path, "image": None, "error": str(e)})


@socketio.on("connect_wow_export")
def handle_connect(data=None):
    """Launch connection in background thread."""
    threading.Thread(target=_do_connect, args=(data,), daemon=True).start()


def _do_connect(data=None):
    global driver, driver_connected
    region = (data or {}).get("region", "tw")
    product = (data or {}).get("product", "wow")

    # Close existing driver if any
    if driver is not None:
        logger.info("Closing previous wow.export session...")
        try:
            driver.quit()
        except Exception:
            pass
        driver = None
        driver_connected = False
        # Kill any lingering nw.exe/chromedriver processes
        import subprocess
        subprocess.run(["taskkill", "/F", "/IM", "nw.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["taskkill", "/F", "/IM", "chromedriver.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

    socketio.emit("connection_status", {"connected": False, "message": "Launching wow.export..."})

    try:
        logger.info("Launching wow.export...")
        driver = connect_to_wow_export()
        logger.info("wow.export launched, waiting for init...")
        socketio.emit("connection_status", {"connected": False, "message": "Waiting for app init..."})
        wait_for_app_ready(driver)
        logger.info("App ready. Connecting to CDN region=%s product=%s", region, product)

        socketio.emit("connection_status", {"connected": False, "message": f"Connecting to CDN ({region})..."})

        driver.execute_script(f"""
            (async () => {{
                const CASCRemote = require('./js/casc/casc-source-remote');

                core.view.selectedCDNRegion = {{tag: '{region}', name: '{region}', delay: null}};
                core.view.lockCDNRegion = true;
                core.view.config.sourceSelectUserRegion = '{region}';

                const casc = new CASCRemote('{region}');
                await casc.init();

                window.__cascSource = casc;
                window.__builds = casc.getProductList();
                core.view.availableRemoteBuilds = window.__builds;
                core.view.sourceSelectShowBuildSelect = true;
            }})().then(() => {{ window.__cdnReady = true; }}).catch(e => {{ window.__cdnError = e.message; }});
        """)

        for _ in range(90):
            ready = driver.execute_script("return window.__cdnReady === true")
            error = driver.execute_script("return window.__cdnError || null")
            if error:
                logger.error("CDN init error: %s", error)
                raise Exception(f"CDN error: {error}")
            if ready:
                break
            time.sleep(2)
        else:
            raise Exception("CDN connection timeout")

        logger.info("CDN connected. Available builds fetched. Loading product: %s", product)
        socketio.emit("connection_status", {"connected": False, "message": f"Loading {product}..."})

        driver.execute_script(f"""
            (async () => {{
                const builds = window.__builds;
                let idx = 0;
                for (let i = 0; i < builds.length; i++) {{
                    if (builds[i].product === '{product}') {{ idx = builds[i].buildIndex; break; }}
                }}
                await window.__cascSource.load(idx);
                core.view.casc = window.__cascSource;
                core.view.installType = 1;
            }})().then(() => {{ window.__cascLoaded = true; }}).catch(e => {{ window.__cascLoadError = e.message; }});
        """)

        socketio.emit("connection_status", {"connected": False, "message": "Loading game data (1-2 min)..."})
        for _ in range(120):
            loaded = driver.execute_script("return window.__cascLoaded === true")
            error = driver.execute_script("return window.__cascLoadError || null")
            if error:
                logger.error("CASC load error: %s", error)
                raise Exception(f"CASC load error: {error}")
            if loaded:
                break
            time.sleep(2)
        else:
            raise Exception("CASC load timeout")

        logger.info("CASC loaded successfully.")
        socketio.emit("connection_status", {"connected": False, "message": "Switching to Models tab..."})
        switch_to_models_tab(driver)
        time.sleep(3)
        driver_connected = True
        socketio.emit("connection_status", {"connected": True, "message": "Connected!"})
    except Exception as e:
        logger.error("Connection failed: %s", str(e))
        driver = None
        driver_connected = False
        socketio.emit("connection_status", {"connected": False, "message": f"Failed: {str(e)}"})


@socketio.on("disconnect")
def handle_disconnect():
    pass


@socketio.on("start_extraction")
def handle_start_extraction(data):
    global extraction_running, _stop_extraction
    logger.info("Received start_extraction: %s", data)
    if extraction_running:
        emit("extraction_status", {"running": True, "message": "Already running"})
        return
    resume = data.get("resume", False)
    limit = data.get("limit", 0)
    category = data.get("category", "all")
    _stop_extraction = False
    extraction_running = True
    threading.Thread(target=_run_extraction, args=(resume, limit, category), daemon=True).start()
    emit("extraction_status", {"running": True, "message": "Started"})


@socketio.on("stop_extraction")
def handle_stop_extraction():
    global _stop_extraction
    _stop_extraction = True
    emit("extraction_status", {"running": False, "message": "Stopping..."})


CATEGORY_FILTERS = {
    "all": None,
    "wmo": ["world/wmo/", "world/minimaps/wmo/"],
    "m2_creatures": ["creature/"],
    "m2_items": ["item/"],
    "m2_characters": ["character/"],
    "m2_spells": ["spells/", "spell/"],
    "m2_environment": ["world/expansion", "world/azeroth", "world/kalimdor", "world/northrend",
                       "world/maps/", "environments/"],
    "m2_doodads": ["world/doodads/", "world/nodxt/detail/"],
}


def _run_extraction(resume, limit, category="all"):
    global extraction_running, _stop_extraction, driver, driver_connected

    try:
        model, preprocess, device = get_clip()

        start_index = 0
        extracted_paths = []
        extracted_vectors = []
        extracted_colors = []

        if resume:
            progress_path = get_progress_path(category)
            npz_path = get_npz_path(category)
            if os.path.exists(progress_path) and os.path.exists(npz_path):
                with open(progress_path, "r") as f:
                    prev_progress = json.load(f)
                start_index = prev_progress.get("last_index", 0) + 1
                data = np.load(npz_path, allow_pickle=True)
                extracted_paths = data["paths"].tolist()
                extracted_vectors = data["vectors"].tolist()
                if "color_vectors" in data:
                    extracted_colors = data["color_vectors"].tolist()
                logger.info("Resuming category '%s' from index %d (%d already done)", category, start_index, len(extracted_paths))
            else:
                logger.info("No previous progress for category '%s', starting fresh", category)

        if driver is None:
            socketio.emit("extraction_progress", {"status": "error", "message": "wow.export not connected."})
            extraction_running = False
            return
        all_models = get_model_list(driver)

        if not all_models:
            socketio.emit("extraction_progress", {"status": "error", "message": "No models found."})
            extraction_running = False
            return

        # Apply category filter
        prefixes = CATEGORY_FILTERS.get(category)
        if prefixes:
            all_models = [m for m in all_models if any(m.lower().startswith(p) for p in prefixes)]
            logger.info("Category filter '%s': %d models after filtering", category, len(all_models))

        if limit > 0:
            all_models = all_models[:limit]

        total = len(all_models)
        failed_count = 0
        start_time = time.time()

        socketio.emit("extraction_progress", {
            "status": "running", "current": start_index, "total": total,
            "extracted": len(extracted_paths), "failed": failed_count, "model": "",
            "eta": "",
        })

        last_processed = start_index
        for i in range(start_index, total):
            if _stop_extraction:
                break

            model_path = all_models[i]
            last_processed = i

            try:
                png_bytes = load_model_and_capture(driver, model_path)
            except Exception as e:
                logger.warning("Model %s capture exception: %s", model_path, str(e)[:80])
                png_bytes = None

            if png_bytes is None:
                failed_count += 1
            else:
                try:
                    # CLIP features
                    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    image_tensor = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = model.encode_image(image_tensor)
                        features = features / features.norm(dim=-1, keepdim=True)
                    clip_vec = features.cpu().numpy().flatten()

                    # Color histogram
                    color_vec = extract_color_histogram(png_bytes)

                    extracted_paths.append(model_path)
                    extracted_vectors.append(clip_vec)
                    extracted_colors.append(color_vec)
                except Exception:
                    failed_count += 1

            if i % 5 == 0 or i == total - 1:
                elapsed = time.time() - start_time
                done_count = (i - start_index + 1)
                if done_count > 0:
                    avg_per_item = elapsed / done_count
                    remaining = (total - i - 1) * avg_per_item
                    hours, rem = divmod(int(remaining), 3600)
                    mins, secs = divmod(rem, 60)
                    if hours > 0:
                        eta_str = f"{hours}h {mins}m"
                    elif mins > 0:
                        eta_str = f"{mins}m {secs}s"
                    else:
                        eta_str = f"{secs}s"
                else:
                    eta_str = "calculating..."

                socketio.emit("extraction_progress", {
                    "status": "running", "current": i + 1, "total": total,
                    "extracted": len(extracted_paths), "failed": failed_count,
                    "model": os.path.basename(model_path), "eta": eta_str,
                })

            if len(extracted_paths) > 0 and len(extracted_paths) % 100 == 0:
                _save_all(last_processed, extracted_paths, extracted_vectors, extracted_colors, category)
                load_faiss_index()

        # Final save with actual last processed index
        if extracted_paths:
            _save_all(last_processed, extracted_paths, extracted_vectors, extracted_colors, category)
            load_faiss_index()

        socketio.emit("extraction_progress", {
            "status": "done" if not _stop_extraction else "stopped",
            "current": last_processed + 1,
            "total": total, "extracted": len(extracted_paths),
            "failed": failed_count, "model": "",
        })

    except Exception as e:
        socketio.emit("extraction_progress", {"status": "error", "message": str(e)})
    finally:
        extraction_running = False
        _stop_extraction = False


def _save_all(index, paths, vectors, colors, category="all"):
    """Save both CLIP and color vectors for a specific category."""
    npz_path = get_npz_path(category)
    progress_path = get_progress_path(category)

    save_data = {
        "paths": np.array(paths, dtype=object),
        "vectors": np.array(vectors, dtype=np.float32),
    }
    if colors:
        save_data["color_vectors"] = np.array(colors, dtype=np.float32)
    np.savez_compressed(npz_path, **save_data)
    with open(progress_path, "w") as f:
        json.dump({"last_index": index, "total_extracted": len(paths), "category": category}, f)


# --- Startup ---

if __name__ == "__main__":
    import webbrowser
    import subprocess

    print("="*50)
    print("  Loading CLIP model (takes ~40s first time)...")
    print("="*50)
    get_clip()
    print("  CLIP ready!")

    count = load_faiss_index()
    if count > 0:
        print(f"  Index ready with {count} models.")
    else:
        print("  No existing index found. Run extraction first.")

    # Close splash screen
    try:
        subprocess.run(["taskkill", "/F", "/IM", "powershell.exe"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    print("\n  Server ready at http://localhost:5001")
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5001")).start()
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)
