"""Optional alerting: POST a JSON message to a Slack-compatible incoming webhook (SPRINTML_WEBHOOK_URL)."""
import json
import os
import urllib.request


def notify(text: str) -> bool:
    url = os.environ.get("SPRINTML_WEBHOOK_URL")
    if not url:
        return False
    try:
        req = urllib.request.Request(url, data=json.dumps({"text": f"[sprintml] {text}"}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        return True
    except Exception as e:  # alerting must never crash the worker
        print("alert failed:", repr(e), flush=True)
        return False
