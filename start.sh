#!/usr/bin/env bash
# Start the API, wait until it is healthy, then start the web page.
set -euo pipefail

uvicorn handover.api:app --host 127.0.0.1 --port 8000 &

for _ in $(seq 1 120); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; then
    break
  fi
  sleep 1
done

exec streamlit run src/handover/app.py \
  --server.port 7860 --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false
