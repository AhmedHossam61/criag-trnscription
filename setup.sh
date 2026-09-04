#!/usr/bin/env bash
# Create an isolated venv for one ASR backend.
#
#   bash setup.sh qwencleo          # recommended for Egyptian + code-switching
#   bash setup.sh cohere
#   bash setup.sh qwen3asr
#   bash setup.sh faster-whisper
#   CUDA=cu124 bash setup.sh qwencleo   # override the wheel index
#
# One venv per backend on purpose: QwenCleo pins torch 2.5.1 while the newer
# transformers releases the other backends need may pull a different torch.
# Sharing one environment is how you end up reinstalling torch at 3am.

set -euo pipefail

BACKEND="${1:-}"
CUDA="${CUDA:-cu121}"
VENV=".venv-${BACKEND}"

case "$BACKEND" in
    qwencleo|qwen3asr|cohere|faster-whisper) ;;
    "")      echo "usage: bash setup.sh {qwencleo|qwen3asr|cohere|faster-whisper}" >&2; exit 1 ;;
    *)       echo "unknown backend: $BACKEND" >&2; exit 1 ;;
esac

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "warning: ffmpeg not found on PATH -- the pipeline cannot decode audio without it." >&2
    echo "         Ubuntu/Debian: sudo apt install ffmpeg" >&2
fi

echo ">> creating $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip --quiet

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA}"

case "$BACKEND" in
    qwencleo)
        # Order matters: torch first (so the package can't re-resolve it),
        # then QwenCleo with --no-deps, then its remaining runtime deps.
        echo ">> installing torch 2.5.1 (${CUDA})"
        pip install torch==2.5.1 torchaudio==2.5.1 --index-url "$TORCH_INDEX"
        echo ">> installing qwencleo-asr (--no-deps)"
        pip install qwencleo-asr --no-deps
        echo ">> installing remaining deps"
        pip install -r requirements-qwencleo.txt
        ;;
    qwen3asr)
        echo ">> installing torch (${CUDA})"
        pip install torch torchaudio --index-url "$TORCH_INDEX"
        pip install -r requirements-qwen3asr.txt
        ;;
    cohere)
        echo ">> installing torch (${CUDA})"
        pip install torch torchaudio --index-url "$TORCH_INDEX"
        pip install -r requirements-cohere.txt
        echo
        echo ">> NOTE: these weights are gated."
        echo "   1. accept the terms at"
        echo "      https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026"
        echo "   2. run: hf auth login"
        ;;
    faster-whisper)
        # CTranslate2 based -- no torch needed at all.
        pip install -r requirements-whisper.txt
        ;;
esac

echo
echo ">> verifying"
python craig_transcribe.py --check-env --backend "$BACKEND" || true

echo
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Then:                 python craig_transcribe.py recording.zip -o out/ --backend $BACKEND"
