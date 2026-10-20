"""BoardMath: barebones live whiteboard -> LaTeX demo using Chandra OCR.

camera -> latest frame -> preprocess -> change check -> Chandra -> LaTeX -> web page
"""

import argparse
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent

# ---------------------------------------------------------------- args

p = argparse.ArgumentParser(description="BoardMath live demo")
p.add_argument("--backend", choices=["hf", "server", "mock"], default="hf",
               help="hf = Chandra via transformers (MPS on Mac); server = Chandra via an "
                    "OpenAI-compatible server (LM Studio / mlx_vlm / vLLM); mock = canned output")
p.add_argument("--camera", default="0", help="camera index or video file/URL")
p.add_argument("--interval", type=float, default=0.0,
               help="min seconds between OCR calls (0 = back-to-back, as fast as the model allows)")
p.add_argument("--diff-threshold", type=float, default=4.0,
               help="mean abs pixel diff (0-255) needed to re-OCR; lower = more sensitive")
p.add_argument("--alpha", type=float, default=1.3, help="contrast gain")
p.add_argument("--beta", type=float, default=10, help="brightness offset")
p.add_argument("--no-bw", dest="bw", action="store_false", help="skip adaptive threshold to B&W")
p.add_argument("--invert", action="store_true", help="invert (chalk on blackboard)")
p.add_argument("--max-side", type=int, default=1600, help="resize long side before OCR")
p.add_argument("--max-tokens", type=int, default=2048, help="max output tokens per frame")
p.add_argument("--api-base", default=os.environ.get("VLLM_API_BASE", "http://localhost:1234/v1"),
               help="server backend: OpenAI-compatible base URL")
p.add_argument("--model-name", default=os.environ.get("VLLM_MODEL_NAME"),
               help="server backend: model id (default: first model the server lists)")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--once", metavar="IMAGE", help="OCR a single image, print result, exit")
args = p.parse_args()

# ---------------------------------------------------------------- shared state

lock = threading.Lock()
state = {
    "raw_jpg": None,         # latest camera frame as jpeg bytes
    "proc_jpg": None,        # latest preprocessed frame as jpeg bytes
    "latest": None,          # latest snapshot dict
    "equations": [],         # ordered unique equations across the lecture
    "seen": set(),
    "snap_count": 0,
    "skipped": 0,
    "last_diff": None,
    "last_latency": None,
    "cam_fps": 0.0,
    "status": "starting",
    "paused": False,
}
settings = {
    "bw": args.bw,
    "invert": args.invert,
    "interval": args.interval,
    "diff_threshold": args.diff_threshold,
}
latest_frame = None  # raw BGR ndarray, written by capture thread

TRANSCRIPTS = ROOT / "transcripts"
transcript_path = TRANSCRIPTS / f"lecture_{datetime.now():%Y%m%d_%H%M%S}.md"

# ---------------------------------------------------------------- preprocessing

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = args.max_side / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    g = cv2.convertScaleAbs(g, alpha=args.alpha, beta=args.beta)
    g = clahe.apply(g)
    g = cv2.medianBlur(g, 3)
    if settings["invert"]:
        g = 255 - g
    th = thumb(g)  # change detection on gray, before thresholding amplifies sensor noise
    if settings["bw"]:
        # dark ink on light board -> black on white
        g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)
    return g, th


def thumb(gray):
    t = cv2.resize(gray, (160, int(160 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(t, (5, 5), 0).astype(np.int16)


# ---------------------------------------------------------------- OCR backends

class MockOCR:
    samples = [
        "Newton's second law\n\n$$F = ma$$\n\nwith $a = \\frac{dv}{dt}$",
        "$$\\int_0^1 x^2\\,dx = \\frac{1}{3}$$",
        "Euler: $$e^{i\\pi} + 1 = 0$$",
        "$$\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0}$$",
    ]

    def __init__(self):
        self.i = 0

    def __call__(self, pil):
        time.sleep(1.0)  # pretend inference
        md = self.samples[self.i % len(self.samples)]
        self.i += 1
        return md, md


class ChandraOCR:
    def __init__(self, backend):
        # chandra reads its settings from env at import time
        if backend == "hf":
            try:
                import torch
                if torch.backends.mps.is_available():
                    os.environ.setdefault("TORCH_DEVICE", "mps")
            except ImportError:
                pass
        else:
            os.environ["VLLM_API_BASE"] = args.api_base
            if args.model_name:
                os.environ["VLLM_MODEL_NAME"] = args.model_name
        os.environ.setdefault("MAX_OUTPUT_TOKENS", str(args.max_tokens))
        from chandra.model import InferenceManager
        from chandra.model.schema import BatchInputItem
        self.Item = BatchInputItem
        print(f"[ocr] loading chandra ({backend}) ...", flush=True)
        self.mgr = InferenceManager(method="hf" if backend == "hf" else "vllm")
        print("[ocr] ready", flush=True)

    def __call__(self, pil):
        kw = {"max_output_tokens": args.max_tokens}
        if args.backend == "server":
            kw["vllm_api_base"] = args.api_base
        out = self.mgr.generate([self.Item(image=pil, prompt_type="ocr_layout")], **kw)[0]
        if out.error:
            raise RuntimeError("chandra returned error")
        return out.markdown, out.raw


MATH_RE = re.compile(
    r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$|<math[^>]*>(.+?)</math>",
    re.S,
)


def extract_equations(*texts):
    eqs = []
    for t in texts:
        for m in MATH_RE.finditer(t or ""):
            e = next(g for g in m.groups() if g is not None).strip()
            if e and e not in eqs:
                eqs.append(e)
        if eqs:
            break  # markdown had math; don't double count from raw
    return eqs


def norm(e):
    return re.sub(r"\s+", "", e)


def run_ocr(ocr, gray):
    pil = Image.fromarray(gray).convert("RGB")
    t0 = time.time()
    md, raw = ocr(pil)
    dt = time.time() - t0
    return md, extract_equations(md, raw), dt


# ---------------------------------------------------------------- threads

def capture_loop():
    global latest_frame
    src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    if not cap.isOpened():
        state["status"] = f"cannot open camera {args.camera}"
        print("[cam]", state["status"], flush=True)
        return
    print(f"[cam] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
          flush=True)
    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            if not isinstance(src, int):  # loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue
        latest_frame = frame
        n += 1
        if n % 3 == 0:  # preview jpeg at reduced rate/size
            small = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
            with lock:
                state["raw_jpg"] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        if time.time() - t0 >= 1:
            state["cam_fps"] = round(n / (time.time() - t0), 1)
            n, t0 = 0, time.time()


def ocr_loop(ocr):
    last_thumb = None
    last_sent = 0.0
    state["status"] = "running"
    while True:
        if state["paused"] or latest_frame is None:
            time.sleep(0.1)
            continue
        wait = settings["interval"] - (time.time() - last_sent)
        if wait > 0:
            time.sleep(min(wait, 0.1))
            continue
        gray, th = preprocess(latest_frame)
        with lock:
            state["proc_jpg"] = cv2.imencode(".jpg", gray)[1].tobytes()
        if last_thumb is not None and th.shape == last_thumb.shape:
            diff = float(np.abs(th - last_thumb).mean())
            state["last_diff"] = round(diff, 2)
            if diff < settings["diff_threshold"]:
                state["skipped"] += 1
                time.sleep(0.2)
                continue
        last_sent = time.time()
        state["status"] = "ocr..."
        try:
            md, eqs, dt = run_ocr(ocr, gray)
        except Exception as e:
            state["status"] = f"ocr error: {e}"
            print("[ocr]", state["status"], flush=True)
            time.sleep(1)
            continue
        last_thumb = th
        add_snapshot(md, eqs, dt)
        state["status"] = "running"


def add_snapshot(md, eqs, dt):
    with lock:
        state["snap_count"] += 1
        snap = {
            "id": state["snap_count"],
            "time": datetime.now().strftime("%H:%M:%S"),
            "latency": round(dt, 2),
            "markdown": md,
            "equations": eqs,
        }
        state["latest"] = snap
        state["last_latency"] = snap["latency"]
        new = []
        for e in eqs:
            k = norm(e)
            if k not in state["seen"]:
                state["seen"].add(k)
                item = {"n": len(state["equations"]) + 1, "time": snap["time"], "latex": e}
                state["equations"].append(item)
                new.append(item)
    print(f"[ocr] #{snap['id']} {dt:.2f}s, {len(eqs)} eqs, {len(new)} new", flush=True)
    TRANSCRIPTS.mkdir(exist_ok=True)
    with open(transcript_path, "a", encoding="utf-8") as f:
        f.write(f"## Snapshot {snap['id']} @ {snap['time']} ({dt:.2f}s)\n\n{md}\n\n")


# ---------------------------------------------------------------- one-shot mode

def make_ocr():
    return MockOCR() if args.backend == "mock" else ChandraOCR(args.backend)


if args.once:
    img = cv2.imread(args.once)
    if img is None:
        sys.exit(f"cannot read {args.once}")
    ocr = make_ocr()
    gray, _ = preprocess(img)
    cv2.imwrite(str(ROOT / "once_processed.png"), gray)
    md, eqs, dt = run_ocr(ocr, gray)
    print(f"--- markdown ({dt:.2f}s) ---\n{md}\n--- equations ---")
    for e in eqs:
        print(e)
    sys.exit(0)

# ---------------------------------------------------------------- web server

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/video")
def video():
    def gen():
        while True:
            with lock:
                jpg = state["raw_jpg"]
            if jpg:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.07)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/processed.jpg")
def processed():
    with lock:
        jpg = state["proc_jpg"]
    return Response(jpg or b"", media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def api_state():
    with lock:
        return {
            "latest": state["latest"],
            "equations": state["equations"],
            "stats": {
                "status": state["status"],
                "backend": args.backend,
                "snapshots": state["snap_count"],
                "skipped": state["skipped"],
                "last_diff": state["last_diff"],
                "last_latency": state["last_latency"],
                "cam_fps": state["cam_fps"],
                "transcript": str(transcript_path.relative_to(ROOT)),
            },
            "paused": state["paused"],
            "settings": settings,
        }


@app.post("/api/pause")
def api_pause():
    state["paused"] = not state["paused"]
    return {"paused": state["paused"]}


@app.post("/api/settings")
async def api_settings(req: Request):
    body = await req.json()
    for k in ("bw", "invert"):
        if k in body:
            settings[k] = bool(body[k])
    for k in ("interval", "diff_threshold"):
        if k in body:
            settings[k] = float(body[k])
    return settings


if __name__ == "__main__":
    import uvicorn
    ocr = make_ocr()
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=ocr_loop, args=(ocr,), daemon=True).start()
    print(f"open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
