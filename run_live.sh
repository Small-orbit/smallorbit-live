#!/usr/bin/env bash
# Usage: STREAM_KEY=... INGEST=primary|backup SECONDS_TO_RUN=20400 bash run_live.sh
# stream.py renders video to stdout and step-synced music to a FIFO.
set -u
cd "$(dirname "$0")"
if [ "${INGEST:-primary}" = "backup" ]; then
  URL="rtmp://b.rtmp.youtube.com/live2?backup=1/${STREAM_KEY}"
else
  URL="rtmp://a.rtmp.youtube.com/live2/${STREAM_KEY}"
fi
W=${W:-1280}; H=${H:-720}
END=$(( $(date +%s) + ${SECONDS_TO_RUN:-20400} ))
while [ "$(date +%s)" -lt "$END" ]; do
  LEFT=$(( END - $(date +%s) ))
  rm -f audio.fifo; mkfifo audio.fifo
  python3 stream.py --seconds "$LEFT" --audio audio.fifo | ffmpeg -loglevel warning -stats_period 60 \
    -thread_queue_size 64 -re -f rawvideo -pix_fmt bgr24 -s ${W}x${H} -r 30 -i - \
    -thread_queue_size 256 -f s16le -ar 48000 -ac 2 -i audio.fifo \
    -map 0:v -map 1:a \
    -c:v libx264 -preset veryfast -tune zerolatency -b:v 4500k -maxrate 4500k -bufsize 9000k -g 60 -keyint_min 60 -sc_threshold 0 -pix_fmt yuv420p \
    -c:a aac -b:a 160k -ar 48000 -f flv "$URL"
  echo "ffmpeg exited, restarting in 3s"; sleep 3
done
