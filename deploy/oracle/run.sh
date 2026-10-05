#!/usr/bin/env bash
# Runs on the Oracle VM: pulls the image and (re)starts the container with the env file in ~/market-research-agent.env.
set -euo pipefail

IMAGE="${1:?usage: run.sh <image>}"
NAME=market-research-agent
ENV_FILE="$HOME/market-research-agent.env"

docker pull "$IMAGE"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped \
  --env-file "$ENV_FILE" -e GIT_SHA="${GIT_SHA:-unknown}" \
  -p 80:7860 "$IMAGE"

for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:80/health >/dev/null; then
    echo "healthy"
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 3
done

docker logs --tail 50 "$NAME"
exit 1
