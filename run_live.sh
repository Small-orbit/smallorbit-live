#!/usr/bin/env bash
# STREAM_KEY=... SECONDS_TO_RUN=21300 bash run_live.sh
# Uses the backup ingest only while the previous run is still streaming (hand-off overlap), then primary.
set -u
cd "$(dirname "$0")"
FF=${FFMPEG:-ffmpeg}
PRIMARY="rtmp://a.rtmp.youtube.com/live2/${STREAM_KEY}"
BACKUP="rtmp://b.rtmp.youtube.com/live2?backup=1/${STREAM_KEY}"
command -v "$FF" >/dev/null || { echo "[run] ffmpeg missing: $FF"; exit 1; }
echo "[run] using $FF"
W=${W:-1280}; H=${H:-720}
END=$(( $(date +%s) + ${SECONDS_TO_RUN:-21300} ))
others() {  # other live runs still in progress
  [ -z "${GH_TOKEN:-}" ] && { echo 0; return; }
  gh api "repos/${GITHUB_REPOSITORY}/actions/workflows/live.yml/runs?status=in_progress" \
    --jq "[.workflow_runs[] | select(.id != ${GITHUB_RUN_ID:-0})] | length" 2>/dev/null || echo 0
}
while [ "$(date +%s)" -lt "$END" ]; do
  LEFT=$(( END - $(date +%s) ))
  if [ "$(others)" -gt 0 ]; then URL="$BACKUP"; SEG=$(( LEFT < 240 ? LEFT : 240 )); echo "[run] previous run still live -> backup ingest for ${SEG}s"
  else URL="$PRIMARY"; SEG=$LEFT; echo "[run] primary ingest for ${SEG}s"; fi
  if [ -n "${STREAM_KEY_2:-}" ]; then  # same encode, also pushed to a second channel; its failure never stops the main one
    OUT=(-flags +global_header -f tee "[f=flv:onfail=ignore]${URL}|[f=flv:onfail=ignore]${URL//${STREAM_KEY}/${STREAM_KEY_2}}")
  else OUT=(-f flv "$URL"); fi
  rm -f audio.fifo; mkfifo audio.fifo
  python3 stream.py --seconds "$SEG" --audio audio.fifo | "$FF" -hide_banner -loglevel error -stats -stats_period 60 \
    -re -f rawvideo -pix_fmt bgr24 -s ${W}x${H} -r 30 -i - \
    -f s16le -ar 48000 -ac 2 -i audio.fifo \
    -map 0:v -map 1:a \
    -c:v libx264 -preset superfast -b:v 6000k -maxrate 6000k -bufsize 12000k -g 60 -keyint_min 60 -sc_threshold 0 -pix_fmt yuv420p \
    -c:a aac -b:a 160k -ar 48000 "${OUT[@]}" 2>&1 | stdbuf -oL tr '\r' '\n'
  echo "[run] ffmpeg exited ($?), next segment in 2s"; sleep 2
done
