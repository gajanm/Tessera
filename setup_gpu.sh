#!/bin/bash
# Run this on a fresh rented GPU box (vast.ai / RunPod, CUDA 12.x image).
set -e

apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 tmux curl nodejs npm

export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME
grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install -r realtime/requirements.txt
pip install flash-attn --no-build-isolation || echo "!! flash-attn failed - model falls back to standard attention (slower but works)"

# pre-download model weights so nothing downloads mid-demo
python -c "from huggingface_hub import snapshot_download; snapshot_download('yyfz233/Pi3'); print('pi3 weights cached')"

# cloudflare tunnel, so the phone can reach this box over real HTTPS
curl -L -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared

cd realtime/webserver && npm install && npx vite build && cd ../..

echo ""
echo "setup complete."
echo "  tmux new -s server  -> cd realtime && python server.py"
echo "  tmux new -s tunnel  -> cloudflared tunnel --url http://localhost:5000"
