"""Reads the channel's live chat in the background and queues rules viewers type,
e.g. "RRLL" or "lrrl". Never raises: if anything breaks, the queue just stays empty."""
import re, threading, time, urllib.request, collections, sys

CHANNEL = "UC8NmwAgiXLAmO23kSRLv-Ng"
RULE_RE = re.compile(r"(?<![A-Za-z])([LRlr]{2,12})(?![A-Za-z])")
queue = collections.deque(maxlen=30)
_seen = set()

def _log(m):
    print(f"[chat] {m}", file=sys.stderr, flush=True)

def _live_video_id():
    req = urllib.request.Request(f"https://www.youtube.com/channel/{CHANNEL}/live",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en",
                                          "Cookie": "CONSENT=YES+1"})
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    m = re.search(r'<link rel="canonical" href="https://www\.youtube\.com/watch\?v=([\w-]{11})"', html)
    if m and '"isLiveNow":true' in html:
        return m.group(1)
    return None

def _run():
    try:
        from chat_downloader import ChatDownloader
    except Exception as e:
        _log(f"chat_downloader unavailable: {e}"); return
    while True:
        try:
            vid = _live_video_id()
            if not vid:
                time.sleep(60); continue
            _log(f"listening to chat on {vid}")
            for msg in ChatDownloader().get_chat(f"https://www.youtube.com/watch?v={vid}", message_groups=["messages"]):
                text = msg.get("message") or ""
                name = (msg.get("author") or {}).get("name", "viewer")
                for r in RULE_RE.findall(text):
                    r = r.upper()
                    if "L" in r and "R" in r and r not in [q[0] for q in queue]:
                        queue.append((r, name[:24]))
                        _log(f"queued {r} from {name}")
                        break
        except Exception as e:
            _log(f"reconnecting after: {str(e)[:120]}")
        time.sleep(30)

def start():
    threading.Thread(target=_run, daemon=True).start()

def next_rule():
    """(rule, author) or None."""
    try:
        return queue.popleft()
    except IndexError:
        return None
