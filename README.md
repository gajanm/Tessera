# Tessera

Real-time 3D reconstruction from a phone camera, with text-prompted object labeling.

Point your phone at a room. Frames stream to a GPU server over WebSocket, get
reconstructed into 3D in overlapping windows, stitched into one continuous world,
and streamed back to a browser that renders the point cloud as it grows. Then type
"find the shelves so I can put boxes in them" and it draws labeled 3D bounding
boxes around the shelves and the boxes.

## Attribution

3D reconstruction uses **Pi3X** (https://github.com/yyfz/Pi3), vendored unmodified
in `third_party/pi3/`. That is not my work.

**My contribution is `realtime/`** — the streaming system built on top of it:
sliding-window inference with Sim3 stitching, the server and concurrency model,
the binary wire protocol, the frontend, and the object labeling.

## How it works

    phone camera ──ws──> FastAPI ──> bounded queue ──> GPU worker thread
                                                              │
                              browser <──ws── binary encode <──┘

Pi3X is a **batch** model: it takes N frames and returns a point cloud in an
arbitrary coordinate frame at an arbitrary scale. Consecutive runs don't line up.

To make it continuous:

1. **Overlapping windows.** 16 frames per chunk, stride 10, so each chunk shares
   6 frames with the last one. Those shared frames are predicted twice, which gives
   dense 3D-to-3D correspondences for free — no feature detection, no RANSAC.
2. **Sim3 alignment.** Umeyama/Kabsch over those correspondences solves the 7-DoF
   similarity transform (rotation, translation, scale) that maps the new chunk into
   the existing world, with a determinant fix so SVD can't hand back a reflection.
3. **Conditioning.** The previous chunk's poses, depths and rays are fed back into
   Pi3X as inputs for the overlap frames, so each prediction starts out already
   near the right coordinate frame and Sim3 only corrects a small residual.
4. **Validation.** Recovered scale is sanity-checked, depth-discontinuity pixels are
   masked, non-finite and far-outlier points are dropped, and a bad solve falls back
   to identity rather than flinging the room across the map.

Inference blocks for seconds, so it runs on a worker thread with a bounded frame
queue that drops *oldest* frames under pressure — in a live scan a stale frame is
worthless because the camera has already moved.

Object labeling composes SAM3 with the dense point map. SAM3 gives 2D masks from
text; Pi3X already knows the 3D position of every pixel; so lifting 2D to 3D is one
fancy-index. DBSCAN with adaptive eps removes mask bleed, and the box is *oriented*
so a chair at an angle gets a tight fit.

## Known limitations

- **No loop closure.** Transforms are chained relative to the previous window, so
  drift compounds. This is visual odometry, not SLAM. Pose-graph optimization is
  the next thing to build.
- **Unbounded memory.** Points accumulate with no voxel dedup, so overlapping views
  duplicate surfaces.
- Needs a CUDA GPU with ~24GB VRAM. Will not run usefully on a laptop.

## Setup

Requires an NVIDIA GPU (24GB VRAM recommended), CUDA 12.x.

    git clone <this repo> && cd tessera
    bash setup_gpu.sh
    cp realtime/.env.example realtime/.env    # add your OpenAI key
    cd realtime && python server.py

To reach it from a phone (camera access requires HTTPS):

    cloudflared tunnel --url http://localhost:5000

Open the printed URL on your phone for the sender, and on your laptop for the viewer.
