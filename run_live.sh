#!/usr/bin/env bash
# Usage: STREAM_KEY=... INGEST=primary|backup SECONDS_TO_RUN=20400 ./run_live.sh
set -u
cd "$(dirname "$0")"
if [ ! -f lofi_all.wav ]; then
  for s in 11 22 33; do python3 lofi.py lofi_$s.wav $s; done
  printf "file 'lofi_11.wav'\nfile 'lofi_22.wav'\nfile 'lofi_33.wav'\n" > l.txt
  ffmpeg -loglevel error -y -f concat -safe 0 -i l.txt -c copy lofi_all.wav
fi
if [ "${INGEST:-primary}" = "backup" ]; then
  URL="rtmp://b.rtmp.youtube.com/live2?backup=1/${STREAM_KEY}"
else
  URL="rtmp://a.rtmp.youtube.com/live2/${STREAM_KEY}"
fi
END=$(( $(date +%s) + ${SECONDS_TO_RUN:-20400} ))
while [ "$(date +%s)" -lt "$END" ]; do
  LEFT=$(( END - $(date +%s) ))
  python3 stream.py --seconds "$LEFT" | ffmpeg -loglevel warning -re -f rawvideo -pix_fmt bgr24 -s 1280x720 -r 30 -i - \
    -re -stream_loop -1 -i lofi_all.wav -map 0:v -map 1:a -shortest \
    -c:v libx264 -preset veryfast -tune zerolatency -b:v 3000k -maxrate 3000k -bufsize 6000k -g 60 -keyint_min 60 -pix_fmt yuv420p \
    -c:a aac -b:a 128k -ar 44100 -f flv "$URL"
  echo "ffmpeg exited, restarting in 3s"; sleep 3
done
