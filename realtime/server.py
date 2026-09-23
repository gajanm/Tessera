"""
Pi3X Real-Time Streaming Server
FastAPI + WebSocket server for incremental 3D reconstruction.
"""

import asyncio
import base64
import json
import queue
import ssl
import struct
import threading
import time
import os
import sys
import argparse
import tempfile

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from pipeline import IncrementalPi3
from object_labeler import ObjectLabeler, FrameData

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(title="Pi3X Real-Time Reconstruction")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────
# Global State
# ─────────────────────────────────────────
pipeline: IncrementalPi3 = None
object_labeler: ObjectLabeler = None
frame_queue: queue.Queue = queue.Queue(maxsize=300)
connected_viewers: set[WebSocket] = set()
processing_lock = threading.Lock()
is_processing = False
video_feeder_stop = threading.Event()

# ─── Global counters ───
frames_received_total = 0
frames_dropped_total = 0
chunks_sent_total = 0        # chunks whose point‑cloud was broadcast to viewers
processing_alive = True      # False if the processing thread dies
_counter_lock = threading.Lock()

def frames_received_total_inc():
    global frames_received_total
    with _counter_lock:
        frames_received_total += 1

def frames_dropped_total_inc():
    global frames_dropped_total
    with _counter_lock:
        frames_dropped_total += 1


# ─────────────────────────────────────────
# WebSocket broadcast helper
# ─────────────────────────────────────────
async def _fan_out(targets: list[WebSocket], sends) -> None:
    """Await every send concurrently and drop the sockets that failed.

    Two reasons this is not a plain `for ws in connected_viewers: await ...`:
    the await yields, so a viewer connecting or disconnecting mid-loop mutates
    the set we are iterating and raises "Set changed size during iteration",
    killing the broadcast partway through; and sending serially means one slow
    viewer delays every viewer queued behind it.
    """
    results = await asyncio.gather(*sends, return_exceptions=True)
    dead = {ws for ws, r in zip(targets, results) if isinstance(r, BaseException)}
    if dead:
        connected_viewers.difference_update(dead)


async def broadcast_to_viewers(message: dict):
    """Send JSON message to all connected viewer WebSockets."""
    targets = list(connected_viewers)
    if not targets:
        return
    await _fan_out(targets, [ws.send_json(message) for ws in targets])


async def broadcast_binary(data: bytes):
    """Send binary data to all connected viewer WebSockets."""
    targets = list(connected_viewers)
    if not targets:
        return
    await _fan_out(targets, [ws.send_bytes(data) for ws in targets])


# ─────────────────────────────────────────
# Blocking pipeline operations
#
# Every one of these takes `processing_lock`, which the worker thread holds for
# the whole duration of a chunk. Acquiring it directly inside an `async def`
# parks the event loop thread, so a single /configure or /reset froze frame
# ingest, chunk broadcasts, acks and /health for seconds. Handlers call these
# through asyncio.to_thread so the wait happens on a worker thread and the loop
# keeps serving.
# ─────────────────────────────────────────
def _drain_frame_queue() -> None:
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break


def _locked_reset() -> None:
    """Drain queued frames and reset the pipeline. Blocking."""
    with processing_lock:
        _drain_frame_queue()
        pipeline.reset()


def _locked_configure(**kwargs) -> None:
    """Hot-reconfigure the pipeline. Blocking."""
    with processing_lock:
        pipeline.configure(**kwargs)


def _locked_save() -> None:
    """Persist cloud and frames to disk. Blocking."""
    with processing_lock:
        pipeline.save_to_disk()


# Requests hitting the LLM endpoints. The SDK default is 600s, so without an
# explicit timeout one stalled connection blocks that request for ten minutes.
OPENAI_TIMEOUT_S = 20.0
_openai_client = None


def _get_openai():
    """Process-wide AsyncOpenAI client, or None if no key is configured.

    One client for the process, not one per request: each construction opens a
    fresh connection pool that is never closed, so every call paid a full TLS
    handshake and leaked the pool afterwards. Safe without a lock because this
    never awaits, so it cannot be interleaved on the event loop.
    """
    global _openai_client
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key or api_key == 'your-api-key-here':
        return None
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=api_key,
                                     timeout=OPENAI_TIMEOUT_S,
                                     max_retries=2)
    return _openai_client


def _decode_frame(img_b64: str):
    """base64 JPEG -> BGR ndarray. Blocking (CPU), call via to_thread."""
    img_bytes = base64.b64decode(img_b64)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


def _encode_current_cloud():
    """Snapshot and encode the accumulated cloud, or None if empty.

    Concatenating millions of points and packing them is blocking CPU work,
    so this runs via to_thread rather than inside the WebSocket handler.
    """
    if pipeline is None or pipeline.total_points <= 0:
        return None
    cloud = pipeline.get_global_cloud(max_points=100_000)
    return encode_point_cloud_binary_fast(
        cloud['points'], cloud['colors'],
        max(pipeline.chunks_processed - 1, 0),
        pipeline.total_points,
        0.0,
    )


def _wipe_saved_state() -> None:
    """Delete the persisted cloud/frames. Blocking (filesystem)."""
    base_dir = os.path.dirname(__file__)
    for fname in ['saved_cloud.npz', 'saved_frames.npz']:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                print(f"[Server] Deleted {fname}")
            except Exception:
                pass


def encode_point_cloud_binary(points: np.ndarray, colors: np.ndarray, chunk_id: int, total: int, elapsed: float) -> bytes:
    """
    Encode point cloud as binary for efficient transfer.
    Format:
        Header: chunk_id(i32) + total_points(i32) + num_new(i32) + elapsed(f32) = 16 bytes
        Body:   N * (x f32 + y f32 + z f32 + r u8 + g u8 + b u8) = N * 15 bytes
    """
    n = len(points)
    header = struct.pack('<iiif', chunk_id, total, n, elapsed)

    if n == 0:
        return header

    pts = points.astype(np.float32)
    cols = colors.astype(np.uint8)

    # Interleave: for each point, 12 bytes xyz + 3 bytes rgb
    body = bytearray(n * 15)
    for i in range(n):
        offset = i * 15
        struct.pack_into('<fff', body, offset, pts[i, 0], pts[i, 1], pts[i, 2])
        body[offset + 12] = cols[i, 0]
        body[offset + 13] = cols[i, 1]
        body[offset + 14] = cols[i, 2]

    return header + bytes(body)


def encode_point_cloud_binary_fast(points: np.ndarray, colors: np.ndarray, chunk_id: int, total: int, elapsed: float) -> bytes:
    """Fast version: header + flat xyz floats + flat rgb bytes."""
    n = len(points)
    header = struct.pack('<iiif', chunk_id, total, n, elapsed)

    if n == 0:
        return header

    pts = np.ascontiguousarray(points.astype(np.float32))   # (N, 3)
    cols = np.ascontiguousarray(colors.astype(np.uint8))     # (N, 3)

    # Layout: header(16) + positions(N*12 f32) + colors(N*3 u8)
    return header + pts.tobytes() + cols.tobytes()


# ─────────────────────────────────────────
# Processing Loop (runs in background thread)
# ─────────────────────────────────────────
def _broadcast_chunk_result(result: dict, loop: asyncio.AbstractEventLoop):
    """Helper: encode & broadcast a single chunk result to all viewers."""
    if result is None or len(result['points']) == 0:
        if result is not None:
            skip_msg = {
                'type': 'chunk_skipped',
                'chunk_id': result['chunk_id'],
                'total_points': result['total_points'],
                'reason': 'No valid points produced',
            }
            asyncio.run_coroutine_threadsafe(broadcast_to_viewers(skip_msg), loop)
        return

    # Downsample for streaming if needed
    pts = result['points']
    cols = result['colors']
    if len(pts) > 100_000:
        s = len(pts) // 100_000
        pts = pts[::s]
        cols = cols[::s]

    binary_data = encode_point_cloud_binary_fast(
        pts, cols,
        result['chunk_id'],
        result['total_points'],
        result.get('elapsed', 0.0)
    )
    asyncio.run_coroutine_threadsafe(broadcast_binary(binary_data), loop)

    done_msg = {
        'type': 'chunk_done',
        'chunk_id': result['chunk_id'],
        'new_points': len(result['points']),
        'total_points': result['total_points'],
        'elapsed': result.get('elapsed', 0.0),
    }
    asyncio.run_coroutine_threadsafe(broadcast_to_viewers(done_msg), loop)


def processing_loop(loop: asyncio.AbstractEventLoop):
    """Background thread that pulls frames and runs Pi3X inference.

    In parallel mode (num_workers > 1), uses an *accumulation window*:
    after the first chunk becomes ready, keeps pulling frames from the
    queue for a short time so that multiple chunks can be batched together
    and processed on the GPU in parallel.
    """
    global is_processing, processing_alive, chunks_sent_total

    use_parallel = pipeline.num_workers > 1
    ACCUMULATION_WINDOW = 3.0 if use_parallel else 0.0
    stride = pipeline.chunk_size - pipeline.overlap

    mode_str = (f"parallel ×{pipeline.num_workers}, accum {ACCUMULATION_WINDOW}s"
                if use_parallel else "sequential")
    print(f"[Server] Processing loop started ({mode_str})")

    def _count_available_chunks() -> int:
        buf = len(pipeline.frame_buffer)
        if pipeline.is_first_chunk:
            if buf < pipeline.chunk_size:
                return 0
            return 1 + (buf - pipeline.chunk_size) // stride
        else:
            return buf // stride

    def _send_status():
        remaining = pipeline.frames_until_ready()
        status_msg = {
            'type': 'status',
            'frames_buffered': len(pipeline.frame_buffer),
            'frames_needed': remaining,
            'total_points': pipeline.total_points,
            'chunks_processed': pipeline.chunks_processed,
            'frames_received': frames_received_total,
            'frames_dropped': frames_dropped_total,
            'chunks_sent': chunks_sent_total,
            'queue_size': frame_queue.qsize(),
            'processing_alive': processing_alive,
        }
        asyncio.run_coroutine_threadsafe(broadcast_to_viewers(status_msg), loop)

    while True:
        try:
            # ── Phase 1: get at least one new frame ────────────────
            try:
                frame_bgr = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with processing_lock:
                pipeline.add_frame(frame_bgr)

                # Drain anything else that already arrived
                while not frame_queue.empty():
                    try:
                        pipeline.add_frame(frame_queue.get_nowait())
                    except queue.Empty:
                        break

                _send_status()

                remaining = pipeline.frames_until_ready()
                if remaining > 0:
                    continue

                # ── Phase 2: at least 1 chunk is ready ─────────────
                is_processing = True

                if use_parallel:
                    t_accum_start = time.time()
                    last_frame_time = time.time()

                    while time.time() - t_accum_start < ACCUMULATION_WINDOW:
                        n_chunks = _count_available_chunks()
                        if n_chunks >= pipeline.num_workers:
                            break
                        try:
                            extra = frame_queue.get(timeout=0.25)
                            pipeline.add_frame(extra)
                            last_frame_time = time.time()
                        except queue.Empty:
                            if time.time() - last_frame_time > 0.6:
                                break

                    n_final = _count_available_chunks()
                    accum_elapsed = time.time() - t_accum_start
                    print(f"[Server] Accumulated {n_final} chunk(s) in "
                          f"{accum_elapsed:.1f}s (buffer={len(pipeline.frame_buffer)})")

                processing_msg = {'type': 'processing', 'chunk_id': pipeline.chunks_processed}
                asyncio.run_coroutine_threadsafe(broadcast_to_viewers(processing_msg), loop)

                # ── Phase 3: inference ─────────────────────────────
                try:
                    if use_parallel:
                        results = pipeline.process_chunks_parallel()
                        for r in results:
                            _broadcast_chunk_result(r, loop)
                            chunks_sent_total += 1
                    else:
                        result = pipeline.process_chunk()
                        _broadcast_chunk_result(result, loop)
                        chunks_sent_total += 1

                except Exception as e:
                    print(f"[Server] ERROR in processing: {e}")
                    import traceback; traceback.print_exc()
                    err_msg = {'type': 'chunk_error', 'error': str(e),
                               'chunk_id': pipeline.chunks_processed}
                    asyncio.run_coroutine_threadsafe(broadcast_to_viewers(err_msg), loop)

                is_processing = False
                _send_status()  # immediate status update after processing

        except Exception as e:
            # Protect the processing thread from dying
            print(f"[Server] FATAL error in processing loop: {e}")
            import traceback; traceback.print_exc()
            is_processing = False
            time.sleep(1)  # avoid tight error loop

    # Should never reach here, but just in case
    processing_alive = False
    print("[Server] ⚠️  Processing loop EXITED")


# ─────────────────────────────────────────
# Video file feeder (testing mode)
# ─────────────────────────────────────────
def feed_video_file(video_path: str, target_fps: float = 2.0, fast: bool = False):
    """Read a video file and push frames into the frame_queue."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VideoFeeder] Cannot open: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(round(video_fps / target_fps)))
    effective_fps = video_fps / skip
    delay = 0.0 if fast else (1.0 / effective_fps)

    print(f"[VideoFeeder] {video_path}: {total_frames} frames @ {video_fps:.1f} FPS")
    print(f"[VideoFeeder] Feeding every {skip} frame(s) → ~{effective_fps:.1f} effective FPS")

    idx = 0
    fed = 0

    while not video_feeder_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % skip != 0:
            continue

        if not frame_queue.full():
            frame_queue.put(frame)
            fed += 1
            frames_received_total_inc()
        else:
            frames_dropped_total_inc()

        if delay > 0:
            time.sleep(delay)

    cap.release()
    print(f"[VideoFeeder] Done: fed {fed} frames")


# ─────────────────────────────────────────
# HTTP Endpoints
# ─────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "total_points": pipeline.total_points if pipeline else 0,
        "chunks_processed": pipeline.chunks_processed if pipeline else 0,
    }


@app.get("/stats")
async def stats():
    """Return detailed pipeline stats for the frontend."""
    return {
        "frames_received": frames_received_total,
        "frames_dropped": frames_dropped_total,
        "frames_buffered": len(pipeline.frame_buffer) if pipeline else 0,
        "queue_size": frame_queue.qsize(),
        "queue_max": frame_queue.maxsize,
        "chunks_processed": pipeline.chunks_processed if pipeline else 0,
        "chunks_sent": chunks_sent_total,
        "total_points": pipeline.total_points if pipeline else 0,
        "is_processing": is_processing,
        "processing_alive": processing_alive,
        "workers": pipeline.num_workers if pipeline else 1,
    }


@app.post("/save")
async def save_state():
    """Manually save point cloud and frame data to disk."""
    if not pipeline:
        return JSONResponse(status_code=400, content={"error": "Pipeline not initialized"})
    if not pipeline.global_points and not pipeline.frame_data:
        return JSONResponse(status_code=400, content={
            "error": "No data to save",
            "total_points": pipeline.total_points,
            "global_points_len": len(pipeline.global_points),
            "frame_data_len": len(pipeline.frame_data),
        })

    try:
        await asyncio.to_thread(_locked_save)
        return {
            "status": "saved",
            "total_points": pipeline.total_points,
            "chunks_processed": pipeline.chunks_processed,
            "frames_saved": len(pipeline.frame_data),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/reset")
async def reset():
    """Reset pipeline state and wipe saved frames/cloud from disk."""
    await asyncio.to_thread(_locked_reset)
    await asyncio.to_thread(_wipe_saved_state)

    await broadcast_to_viewers({'type': 'reset'})
    return {"status": "reset_complete"}


@app.post("/configure")
async def configure(config: dict):
    """Hot-reconfigure pipeline parameters."""
    await asyncio.to_thread(
        _locked_configure,
        chunk_size=config.get('chunk_size'),
        overlap=config.get('overlap'),
        conf_thre=config.get('conf_thre'),
    )
    return {
        "status": "configured",
        "chunk_size": pipeline.chunk_size,
        "overlap": pipeline.overlap,
        "conf_thre": pipeline.conf_thre,
    }


@app.get("/config")
async def get_config():
    """Get current pipeline configuration."""
    return {
        "chunk_size": pipeline.chunk_size,
        "overlap": pipeline.overlap,
        "conf_thre": pipeline.conf_thre,
    }


@app.get("/cloud")
async def get_full_cloud():
    """Get the full accumulated point cloud (downsampled)."""
    cloud = pipeline.get_global_cloud(max_points=200_000)
    # Return as binary
    pts = cloud['points']
    cols = cloud['colors']
    return JSONResponse({
        "total_points": pipeline.total_points,
        "returned_points": len(pts),
        "chunks_processed": pipeline.chunks_processed,
    })


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...), fps: float = 2.0, fast: bool = True):
    """Upload a video file for testing (simulates live camera feed)."""
    video_feeder_stop.clear()

    # Save uploaded file. Streamed in chunks rather than file.read() into one
    # bytes object, so a large video does not spike memory, and each write goes
    # to a worker thread so disk I/O never stalls the loop.
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_path = tmp.name
    tmp.close()
    with open(tmp_path, 'wb') as out:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            await asyncio.to_thread(out.write, chunk)

    # Reset pipeline
    await asyncio.to_thread(_locked_reset)

    await broadcast_to_viewers({'type': 'reset'})

    # Start feeding in background
    t = threading.Thread(target=feed_video_file, args=(tmp_path, fps, fast), daemon=True)
    t.start()

    return {
        "status": "feeding",
        "filename": file.filename,
        "fps": fps,
        "fast": fast,
    }


@app.post("/stop-video")
async def stop_video():
    """Stop the video feeder."""
    video_feeder_stop.set()
    return {"status": "stopped"}


# ─────────────────────────────────────────
# Natural Language → Object Prompts (OpenAI)
# ─────────────────────────────────────────
class NLParseRequest(BaseModel):
    text: str

@app.post("/parse-nl-prompt")
async def parse_nl_prompt(req: NLParseRequest):
    """
    Use OpenAI to extract comma-separated object names from a
    natural language description.
    """
    client = _get_openai()
    if client is None:
        return JSONResponse({"error": "OPENAI_API_KEY not configured in .env"}, status_code=500)

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helper that extracts physical object names from user descriptions. "
                        "The user will describe a scene or a task involving physical objects in a room. "
                        "Extract ONLY the distinct physical object types mentioned or implied. "
                        "Return them as a comma-separated list of short, simple nouns (1-2 words each). "
                        "Do NOT include actions, adjectives, or abstract concepts. "
                        "Example: user says 'I want to figure out where the shelves are so we can place the boxes into the shelves' "
                        "→ you return: shelves, boxes"
                    ),
                },
                {"role": "user", "content": req.text},
            ],
            max_tokens=200,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        # Clean up: split, strip, deduplicate, lowercase
        objects = list(dict.fromkeys(
            o.strip().lower() for o in raw.split(',') if o.strip()
        ))
        return {"objects": objects, "raw": raw}

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────
# Spatial Q&A over labeled objects (OpenAI)
# ─────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    question: str
    objects: list[dict] = []
    history: list[dict] = []

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Answer a natural-language question about the labeled objects in the scene.

    The point cloud itself is far too large to send to an LLM, so we send only
    the structured summary the labeler already produced: each object's label,
    centroid, extent and confidence. That is enough to answer the spatial
    questions people actually ask ("what is left of the desk?", "will the box
    fit on the shelf?") while keeping the prompt small and cheap.
    """
    client = _get_openai()
    if client is None:
        return JSONResponse({"error": "OPENAI_API_KEY not configured in .env"}, status_code=500)

    if not req.objects:
        return {"answer": "No objects are labeled yet. Run a label pass first, then ask."}

    try:
        # Compact, deterministic scene description. Units are world units, and
        # the axes are the viewer's convention (x right, y up, z toward camera).
        lines = []
        for o in req.objects[:60]:
            c = o.get('center') or [0, 0, 0]
            e = o.get('extent') or [0, 0, 0]
            lines.append(
                f"- {o.get('label','?')}: center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
                f"size=({e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f}) "
                f"confidence={o.get('confidence', 0):.2f} points={o.get('point_count', 0)}"
            )
        scene = "\n".join(lines)

        messages = [{
            "role": "system",
            "content": (
                "You answer questions about a 3D scene that was reconstructed from a phone camera. "
                "You are given a list of detected objects with their 3D centers and bounding box sizes. "
                "Axes: +x is right, +y is up, +z is toward the camera. Units are approximate meters. "
                "Reason about relative positions, distances and whether things fit. "
                "Be concise, two or three sentences. If the object list does not contain enough "
                "information to answer, say so plainly instead of guessing."
            ),
        }, {
            "role": "user",
            "content": f"Objects in the scene:\n{scene}",
        }]

        # Replay prior turns so follow-ups like "what about the other one?" work.
        for turn in req.history[-8:]:
            role = "assistant" if turn.get("role") == "model" else "user"
            text = turn.get("text", "")
            if text:
                messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": req.question})

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.2,
        )
        return {"answer": response.choices[0].message.content.strip()}

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────
# Object Labeling Endpoint (fire-and-forget)
# ─────────────────────────────────────────
class LabelRequest(BaseModel):
    prompts: list[str]
    confidence: float = 0.3
    max_frames: int = 20

_labeling_in_progress = False

@app.post("/label-objects")
async def label_objects_endpoint(req: LabelRequest):
    """
    Run SAM3 object labeling in the background (non-blocking).
    Returns 202 immediately. Results are broadcast via WebSocket
    when labeling completes.
    """
    global _labeling_in_progress

    prompts = req.prompts
    confidence = req.confidence
    max_frames = req.max_frames

    if not prompts:
        return JSONResponse({"error": "No prompts provided"}, status_code=400)

    if pipeline is None or not pipeline.frame_data:
        return JSONResponse({"error": "No frames available yet. Process some chunks first."},
                            status_code=400)

    if _labeling_in_progress:
        return JSONResponse({"error": "Labeling already in progress"}, status_code=409)

    # Claim the slot here rather than inside the worker. Handlers run on the
    # single event-loop thread, so this check-and-set cannot be interleaved;
    # setting it after the thread starts left a window in which a second
    # request also passed the guard and loaded a second SAM3 onto a GPU that
    # auto_detect_workers has already sized to be full.
    _labeling_in_progress = True

    # Notify viewers that labeling is starting
    await broadcast_to_viewers({'type': 'labeling_start', 'prompts': prompts})

    # Capture the event loop for broadcasting from the background thread.
    # get_running_loop, not get_event_loop: inside a coroutine the latter is
    # deprecated and only happens to work.
    loop = asyncio.get_running_loop()

    # Copy frame_data snapshot so background thread doesn't fight with pipeline
    frame_data_snapshot = list(pipeline.frame_data)

    def _run_labeling_background():
        """Runs in a daemon thread. Broadcasts results via WebSocket when done."""
        global object_labeler, _labeling_in_progress

        try:
            if object_labeler is None:
                object_labeler = ObjectLabeler(
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )

            # Convert pipeline frame_data dicts to FrameData objects
            frame_data_list = [
                FrameData(
                    frame_idx=fd['frame_idx'],
                    image=fd['image'],
                    point_map=fd['point_map'],
                    conf_mask=fd['conf_mask'],
                )
                for fd in frame_data_snapshot
            ]

            results = object_labeler.label_objects(
                frames=frame_data_list,
                prompts=prompts,
                confidence_threshold=confidence,
                max_frames=max_frames,
                include_debug_images=True,
                max_debug_images=5,
            )

            # Serialize results (including OBB corners, rotation, debug images)
            objects_json = [
                {
                    'label': obj.label,
                    'instance_id': obj.instance_id,
                    'bbox_min': obj.bbox_min,
                    'bbox_max': obj.bbox_max,
                    'center': obj.center,
                    'extent': obj.extent,
                    'rotation': obj.rotation,
                    'corners': obj.corners,
                    'confidence': round(obj.confidence, 3),
                    'point_count': obj.point_count,
                    'debug_images': obj.debug_images,
                }
                for obj in results
            ]

            # Broadcast to all viewers from the event loop
            asyncio.run_coroutine_threadsafe(
                broadcast_to_viewers({
                    'type': 'labeled_objects',
                    'objects': objects_json,
                }),
                loop,
            )

            print(f"[Server] Labeling complete: {len(results)} objects broadcast to viewers")

        except Exception as e:
            import traceback; traceback.print_exc()
            asyncio.run_coroutine_threadsafe(
                broadcast_to_viewers({'type': 'labeling_error', 'error': str(e)}),
                loop,
            )
        finally:
            _labeling_in_progress = False

    # Fire-and-forget: start background thread, return 202 immediately.
    # If the thread cannot start, release the slot we claimed above, otherwise
    # labeling is wedged at "already in progress" until the server restarts.
    try:
        t = threading.Thread(target=_run_labeling_background, daemon=True)
        t.start()
    except BaseException:
        _labeling_in_progress = False
        raise

    return JSONResponse(
        {"status": "labeling_started", "prompts": prompts, "max_frames": max_frames},
        status_code=202,
    )


# ─────────────────────────────────────────
# WebSocket Endpoints
# ─────────────────────────────────────────
@app.websocket("/ws/viewer")
async def viewer_ws(ws: WebSocket):
    """Viewer WebSocket: receives point cloud updates."""
    await ws.accept()
    connected_viewers.add(ws)
    print(f"[Server] Viewer connected ({len(connected_viewers)} total)")

    # Send current state
    await ws.send_json({
        'type': 'init',
        'total_points': pipeline.total_points,
        'chunks_processed': pipeline.chunks_processed,
        'config': {
            'chunk_size': pipeline.chunk_size,
            'overlap': pipeline.overlap,
            'conf_thre': pipeline.conf_thre,
        }
    })

    # If we already have a cloud, send it
    binary = await asyncio.to_thread(_encode_current_cloud)
    if binary is not None:
        await ws.send_bytes(binary)

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get('type') == 'configure':
                await asyncio.to_thread(
                    _locked_configure,
                    chunk_size=msg.get('chunk_size'),
                    overlap=msg.get('overlap'),
                    conf_thre=msg.get('conf_thre'),
                )
                await ws.send_json({
                    'type': 'configured',
                    'config': {
                        'chunk_size': pipeline.chunk_size,
                        'overlap': pipeline.overlap,
                        'conf_thre': pipeline.conf_thre,
                    }
                })
            elif msg.get('type') == 'reset':
                await asyncio.to_thread(_locked_reset)
                await asyncio.to_thread(_wipe_saved_state)
                await broadcast_to_viewers({'type': 'reset'})

            elif msg.get('type') == 'get_cloud':
                binary = await asyncio.to_thread(_encode_current_cloud)
                if binary is not None:
                    await ws.send_bytes(binary)

    except WebSocketDisconnect:
        pass
    finally:
        connected_viewers.discard(ws)
        print(f"[Server] Viewer disconnected ({len(connected_viewers)} total)")


@app.websocket("/ws/sender")
async def sender_ws(ws: WebSocket):
    """Sender WebSocket: receives camera frames as base64 JPEG."""
    await ws.accept()
    print("[Server] Sender connected")
    frames_received = 0

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get('type') == 'frame':
                # base64 + JPEG decode is 5-15ms of CPU per frame; at 10fps that
                # is a permanent tax on the event loop, stalling every other
                # connection. Hand it to a worker thread.
                frame = await asyncio.to_thread(_decode_frame, msg['image'])

                if frame is not None:
                    frames_received += 1
                    try:
                        frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Drop oldest frame to make room
                        try:
                            frame_queue.get_nowait()
                            frames_dropped_total_inc()
                        except queue.Empty:
                            pass
                        try:
                            frame_queue.put_nowait(frame)
                        except queue.Full:
                            frames_dropped_total_inc()
                    frames_received_total_inc()

                await ws.send_json({
                    'type': 'frame_ack',
                    'frame': frames_received,
                    'queue_size': frame_queue.qsize(),
                    'queue_full': frame_queue.full(),
                    'total_received': frames_received_total,
                    'total_dropped': frames_dropped_total,
                })

    except WebSocketDisconnect:
        pass
    finally:
        print(f"[Server] Sender disconnected (received {frames_received} frames)")


# ─────────────────────────────────────────
# Static file serving
# ─────────────────────────────────────────
webserver_dir = os.path.join(os.path.dirname(__file__), 'webserver', 'dist')
if os.path.exists(webserver_dir):
    app.mount("/", StaticFiles(directory=webserver_dir, html=True), name="static")
else:
    # Fallback: serve from webserver root during dev
    dev_dir = os.path.join(os.path.dirname(__file__), 'webserver')
    if os.path.exists(dev_dir):
        @app.get("/")
        async def serve_index():
            return FileResponse(os.path.join(dev_dir, 'index.html'))

        @app.get("/viewer.html")
        async def serve_viewer():
            return FileResponse(os.path.join(dev_dir, 'viewer.html'))

        @app.get("/sender.html")
        async def serve_sender():
            return FileResponse(os.path.join(dev_dir, 'sender.html'))


# ─────────────────────────────────────────
# Startup
# ─────────────────────────────────────────
@app.on_event("shutdown")
async def _close_openai():
    """Release the shared HTTP pool so shutdown is clean."""
    if _openai_client is not None:
        try:
            await _openai_client.close()
        except Exception:
            pass


@app.on_event("startup")
async def startup():
    # The worker thread hands broadcasts back with run_coroutine_threadsafe, so
    # it needs the loop that is actually running, not whatever get_event_loop
    # would resolve to.
    loop = asyncio.get_running_loop()
    t = threading.Thread(target=processing_loop, args=(loop,), daemon=True)
    t.start()
    print("[Server] Background processing thread started")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pi3X Real-Time Server')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--ckpt', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--video', type=str, default=None, help='Video file for testing')
    parser.add_argument('--video-fps', type=float, default=2.0, help='Effective FPS for video')
    parser.add_argument('--fast', action='store_true', help='Feed video as fast as possible')
    parser.add_argument('--chunk-size', type=int, default=16, help='Chunk size')
    parser.add_argument('--overlap', type=int, default=6, help='Overlap size')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parallel GPU workers (0=auto-detect from GPU memory)')
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = IncrementalPi3(device='cuda' if torch.cuda.is_available() else 'cpu',
                              num_workers=1)  # set workers after model load
    pipeline.load_model(ckpt=args.ckpt)
    pipeline.configure(chunk_size=args.chunk_size, overlap=args.overlap)

    # Determine worker count (auto-detect needs model loaded to know memory usage)
    if args.workers == 0:
        pipeline.num_workers = pipeline.auto_detect_workers()
    else:
        pipeline.num_workers = max(1, args.workers)
    print(f"[Server] GPU workers: {pipeline.num_workers}")

    # Try to load persisted point cloud
    try:
        if pipeline.load_from_disk():
            print(f"[Server] Loaded {pipeline.total_points} persisted points from disk")
    except Exception as e:
        print(f"[Server] Could not load persisted cloud: {e}")

    # Start video feeder if provided
    if args.video:
        if not os.path.isfile(args.video):
            print(f"Video file not found: {args.video}")
            sys.exit(1)
        t = threading.Thread(target=feed_video_file, args=(args.video, args.video_fps, args.fast), daemon=True)
        t.start()

    # SSL
    cert_file = os.path.join(os.path.dirname(__file__), 'webserver', 'server.cert')
    key_file = os.path.join(os.path.dirname(__file__), 'webserver', 'server.key')

    ssl_config = {}
    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_config = {
            'ssl_certfile': cert_file,
            'ssl_keyfile': key_file,
        }
        print(f"[Server] HTTPS enabled")
    else:
        print(f"[Server] No SSL certs found, running HTTP only")

    print("=" * 60)
    print("  Pi3X Real-Time Reconstruction Server")
    print("=" * 60)
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Port: {args.port}")
    print(f"  Chunk: {pipeline.chunk_size} frames, Overlap: {pipeline.overlap}")
    print(f"  Workers: {pipeline.num_workers} "
          f"({'auto-detected' if args.workers == 0 else 'manual'})")
    if args.video:
        print(f"  Video: {args.video} ({args.video_fps} fps, {'fast' if args.fast else 'realtime'})")
    print(f"  Frontend: https://0.0.0.0:{args.port}")
    print("=" * 60)

    uvicorn.run(app, host='0.0.0.0', port=args.port, **ssl_config)
