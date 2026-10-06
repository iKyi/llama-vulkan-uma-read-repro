#!/usr/bin/env bash
# perf profile of llama-server while it serves short requests over a cached prefix.
# usage: profile.sh /path/to/llama-server /path/to/model.gguf   (needs sudo for perf)
set -euo pipefail
bin=$1 model=$2
"$bin" -m "$model" --ctx-size 42000 --port 8099 -fa on --parallel 1 > server.log 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:8099/health > /dev/null; do sleep 1; kill -0 $pid; done
load() {
  python3 - "$1" <<'PY'
import json, sys, urllib.request
p = ' '.join(('apples pears plums apricots grapes ' * 100).split()[:170])
def c(x):
    body = json.dumps({'prompt': x, 'n_predict': 1, 'cache_prompt': True}).encode()
    urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8099/completion', body, {'Content-Type': 'application/json'})).read()
for i in range(int(sys.argv[1])):
    c(p); c(p + ' x%d' % i)
PY
}
load 2
sudo perf record -F 999 -g -p $pid -o perf.data -- sleep 12 > /dev/null 2>&1 &
perfpid=$!
sleep 0.5
load 20
wait $perfpid || true
sudo chown "$USER" perf.data
perf report -i perf.data --no-children --sort dso,symbol --stdio 2>/dev/null | grep -v '^#' | grep '%' | head -12
