"""
PyroWatch - Alert Notification Module
=======================================
Dispatches alerts via Email (SMTP), SMS (Twilio), or HTTP webhook
when smoke or fire is detected above the configured threshold.

All credentials are read from environment variables — no hardcoded secrets.

Usage (standalone test):
    python src/alert.py

Environment variables:
    SMTP_USER, SMTP_PASS, EMAIL_TO, SMTP_HOST, SMTP_PORT
    TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO
    WEBHOOK_URL
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

log = logging.getLogger("PyroWatch.Alert")


def send_alert(
    result: Dict[str, Any],
    lat: Optional[float] = None,
    lng: Optional[float] = None,
) -> None:
    """
    Dispatch an alert through all configured channels.

    Parameters
    ----------
    result : detection result dict (output of SmokeFireDetector.detect())
    lat    : GPS latitude (optional)
    lng    : GPS longitude (optional)
    """
    status = result.get("status", "UNKNOWN")
    smoke  = result.get("smoke_pct", 0)
    fire   = result.get("fire_pct",  0)
    source = result.get("source",   "unknown")
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    loc    = f"{lat:.4f}N, {lng:.4f}E" if (lat and lng) else "GPS unavailable"

    subject = f"[PyroWatch ALERT] {status} — {ts}"
    body = (
        f"PyroWatch Wildfire Detection Alert\n"
        f"{'='*40}\n"
        f"Status   : {status}\n"
        f"Smoke    : {smoke:.1f}%\n"
        f"Fire     : {fire:.1f}%\n"
        f"Location : {loc}\n"
        f"Source   : {source}\n"
        f"Time     : {ts}\n"
        f"{'='*40}\n"
        f"Automated alert — verify before dispatching resources.\n"
    )

    log.warning(f"\n{'*'*52}\n{body}{'*'*52}")

    sent = []
    if _send_email(subject, body):        sent.append("email")
    if _send_sms(body[:160]):             sent.append("sms")
    if _send_webhook(result, lat, lng):   sent.append("webhook")

    if sent:
        log.info(f"Alert dispatched via: {', '.join(sent)}")
    else:
        log.info("No alert channels configured — logged to console only.")
        log.info("Set SMTP_USER/EMAIL_TO, TWILIO_*, or WEBHOOK_URL to enable.")


def _send_email(subject: str, body: str) -> bool:
    """Send email alert via SMTP (e.g. Gmail)."""
    user = os.getenv("SMTP_USER", "")
    pwd  = os.getenv("SMTP_PASS", "")
    to   = os.getenv("EMAIL_TO",  "")
    if not all([user, pwd, to]):
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg             = MIMEText(body)
        msg["Subject"]  = subject
        msg["From"]     = user
        msg["To"]       = to
        host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        port = int(os.getenv("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.starttls()
            s.login(user, pwd)
            s.sendmail(user, [to], msg.as_string())
        log.info(f"Email sent to {to}")
        return True
    except Exception as exc:
        log.error(f"Email failed: {exc}")
        return False


def _send_sms(message: str) -> bool:
    """Send SMS alert via Twilio. Requires: pip install twilio"""
    sid   = os.getenv("TWILIO_SID",   "")
    token = os.getenv("TWILIO_TOKEN", "")
    frm   = os.getenv("TWILIO_FROM",  "")
    to    = os.getenv("TWILIO_TO",    "")
    if not all([sid, token, frm, to]):
        return False
    try:
        from twilio.rest import Client       # type: ignore
        Client(sid, token).messages.create(body=message, from_=frm, to=to)
        log.info(f"SMS sent to {to}")
        return True
    except ImportError:
        log.warning("pip install twilio to enable SMS alerts.")
        return False
    except Exception as exc:
        log.error(f"SMS failed: {exc}")
        return False


def _send_webhook(
    result: Dict[str, Any],
    lat: Optional[float],
    lng: Optional[float],
) -> bool:
    """POST JSON payload to a webhook URL."""
    url = os.getenv("WEBHOOK_URL", "")
    if not url:
        return False
    try:
        import urllib.request
        payload = json.dumps({
            "event":     "pyrowatch_alert",
            "status":    result.get("status"),
            "smoke_pct": result.get("smoke_pct"),
            "fire_pct":  result.get("fire_pct"),
            "lat":       lat,
            "lng":       lng,
            "timestamp": datetime.now().isoformat(),
        }).encode()
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            log.info(f"Webhook dispatched → {url}  (HTTP {resp.status})")
        return True
    except Exception as exc:
        log.error(f"Webhook failed: {exc}")
        return False


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    print("Testing alert dispatch (no env vars configured = console only).\n")
    send_alert(
        {
            "status":    "SMOKE DETECTED",
            "smoke_pct": 71.2,
            "fire_pct":  6.8,
            "alert":     True,
            "source":    "camera_01",
        },
        lat=23.2599,
        lng=77.4126,
    )
