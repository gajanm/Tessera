#!/bin/bash
# Run on the GPU box from the repo root BEFORE a demo:   bash preflight.sh
# Checks everything the demo depends on and prints PASS / WARN / FAIL for each.
# Exit code is the number of FAILs, so 0 means ready.

cd "$(dirname "$0")"
FAILS=0; WARNS=0
pass() { echo "  PASS  $1"; }
warn() { echo "  WARN  $1"; WARNS=$((WARNS+1)); }
fail() { echo "  FAIL  $1"; FAILS=$((FAILS+1)); }

echo "== GPU"
GPU=$(python - <<'EOF' 2>&1
import torch
if not torch.cuda.is_available():
    print("NOCUDA"); raise SystemExit
p = torch.cuda.get_device_properties(0)
print(f"{p.name}|{p.total_memory/1e9:.0f}|{torch.__version__}")
EOF
)
if [[ "$GPU" == *"|"* ]]; then
  IFS='|' read -r NAME MEM TV <<< "$GPU"
  pass "$NAME, ${MEM}GB VRAM, torch $TV"
  [ "$MEM" -lt 20 ] && warn "under 20GB VRAM, Pi3X + SAM3 together may OOM"
else
  fail "no CUDA GPU visible to torch ($(echo "$GPU" | tail -1))"
fi

echo "== Python packages"
for mod in numpy cv2 einops fastapi uvicorn open3d sklearn dotenv openai huggingface_hub; do
  python -c "import $mod" 2>/dev/null && pass "$mod" || fail "$mod not importable (pip install -r realtime/requirements.txt)"
done
python -c "import sam3.model_builder" 2>/dev/null && pass "sam3" \
  || fail "sam3 not importable (pip install git+https://github.com/facebookresearch/sam3.git)"
python -c "import sys; sys.path.insert(0,'third_party'); from pi3.models.pi3x import Pi3X" 2>/dev/null \
  && pass "vendored pi3x imports" || fail "third_party/pi3 does not import"

NP=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
[[ "$NP" == 1.* ]] && pass "numpy $NP (sam3 needs <2)" || fail "numpy $NP, sam3 needs <2: pip install 'numpy<2' 'opencv-python<4.12'"
# cv2 compiled against the wrong numpy only fails when it touches an array
python -c "import cv2, numpy as np; cv2.imdecode(np.zeros(1,np.uint8), 1)" 2>/dev/null \
  && pass "opencv works with this numpy" || fail "opencv broken with numpy $NP: pip install 'opencv-python<4.12'"
pip check >/dev/null 2>&1 && pass "pip check clean" || warn "pip check reports conflicts (run 'pip check' to read them)"

echo "== Hugging Face"
WHO=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null)
[ -n "$WHO" ] && pass "logged in as $WHO" || fail "not logged in: python -c \"from huggingface_hub import login; login()\""
CACHED=$(python -c "from huggingface_hub import scan_cache_dir; print(' '.join(r.repo_id for r in scan_cache_dir().repos))" 2>/dev/null)
[[ " $CACHED " == *" yyfz233/Pi3X "* ]] && pass "Pi3X weights cached" || fail "Pi3X weights not cached (would download at startup)"
[[ " $CACHED " == *" facebook/sam3 "* ]] && pass "SAM3 weights cached" || fail "SAM3 weights not cached (labeling will fail or stall)"

echo "== OpenAI key"
if [ -f realtime/.env ]; then
  pass "realtime/.env exists"
  git ls-files --error-unmatch realtime/.env >/dev/null 2>&1 && fail "realtime/.env is TRACKED BY GIT, remove it now"
  KEYCHECK=$(cd realtime && python - <<'EOF' 2>&1
import os
from dotenv import load_dotenv
load_dotenv()
k = os.environ.get("OPENAI_API_KEY", "")
if not k or k.startswith("sk-...") or k == "your-api-key-here":
    print("PLACEHOLDER"); raise SystemExit
from openai import OpenAI
OpenAI(api_key=k).models.list()   # free call, just proves the key is live
print("OK")
EOF
)
  case "$KEYCHECK" in
    *OK) pass "key is live" ;;
    *PLACEHOLDER*) fail "key is still the placeholder" ;;
    *) fail "key rejected or unreachable: $(echo "$KEYCHECK" | tail -1)" ;;
  esac
else
  fail "realtime/.env missing: cp realtime/.env.example realtime/.env and add your key"
fi

echo "== Frontend + networking"
[ -f realtime/webserver/dist/viewer.html ] && [ -f realtime/webserver/dist/sender.html ] \
  && pass "frontend built" || fail "frontend not built: cd realtime/webserver && npm install && npx vite build"
[ -f realtime/webserver/server.cert ] || [ -f realtime/webserver/server.key ] \
  && fail "self-signed certs present, server will run HTTPS and the tunnel breaks: rm realtime/webserver/server.{cert,key}" \
  || pass "no self-signed certs (server runs plain HTTP behind tunnel)"
command -v cloudflared >/dev/null && pass "cloudflared installed" || fail "cloudflared missing"
command -v tmux >/dev/null && pass "tmux installed" || warn "tmux missing, server dies if your SSH drops"
if python -c "import socket; s=socket.socket(); s.bind(('0.0.0.0',5000))" 2>/dev/null; then
  pass "port 5000 free"
elif curl -s localhost:5000/health >/dev/null; then
  pass "server already running and /health answers"
else
  fail "port 5000 taken by something that isn't the server"
fi

echo "== Disk + state"
FREE=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$FREE" -ge 10 ] && pass "${FREE}GB disk free" || warn "only ${FREE}GB free, saves and caches may fail"
ls realtime/saved_*.npz >/dev/null 2>&1 \
  && warn "saved_*.npz present, server will auto-load an OLD scan (delete for a clean demo)" \
  || pass "no saved scan, demo starts clean"

echo
echo "$FAILS fail(s), $WARNS warning(s)"
[ "$FAILS" -eq 0 ] && echo "READY" || echo "NOT READY, fix the FAILs above top to bottom"
exit $FAILS
