#!/usr/bin/env bash
# Start LLAMA_SERVER on MODEL (port 8099), run repro.py against it, stop it.
# usage: run.sh /path/to/llama-server /path/to/model.gguf [extra llama-server args...]
set -euo pipefail
bin=$1 model=$2; shift 2
"$bin" -m "$model" --ctx-size 42000 --port 8099 -fa on --parallel 1 "$@" > server.log 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:8099/health > /dev/null; do sleep 1; kill -0 $pid; done
python3 "$(dirname "$0")/repro.py"
