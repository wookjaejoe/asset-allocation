from __future__ import annotations

import argparse
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send an HTML email with optional attachments via SMTP.")
    p.add_argument("--subject", required=True)
    p.add_argument("--html", required=True, help="HTML file path")
    p.add_argument("--attach", action="append", default=[], help="Attachment path (repeatable)")
    p.add_argument("--to", default=os.getenv("MAIL_TO", ""), help="Comma-separated recipients (or MAIL_TO env)")
    p.add_argument("--from", dest="from_addr", default=os.getenv("MAIL_FROM", ""), help="From (or MAIL_FROM env)")
    return p.parse_args()


def _env_required(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise SystemExit(f"Missing env var: {name}")
    return val


def main() -> None:
    args = parse_args()

    host = _env_required("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASS", "").strip()
    use_starttls = os.getenv("SMTP_STARTTLS", "1").strip() not in {"0", "false", "False"}

    from_addr = args.from_addr.strip()
    to_addrs = [x.strip() for x in args.to.split(",") if x.strip()]
    if not from_addr:
        raise SystemExit("Missing sender: --from or MAIL_FROM")
    if not to_addrs:
        raise SystemExit("Missing recipients: --to or MAIL_TO")

    html_path = Path(args.html)
    html = html_path.read_text(encoding="utf-8")

    msg = EmailMessage()
    msg["Subject"] = args.subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.set_content("This email contains an HTML report. If you cannot view it, open the attached HTML.")
    msg.add_alternative(html, subtype="html")

    for ap in args.attach:
        path = Path(ap)
        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            maintype, subtype = "application", "octet-stream"
        else:
            maintype, subtype = mime.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)

    with smtplib.SMTP(host=host, port=port, timeout=30) as smtp:
        if use_starttls:
            smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)


if __name__ == "__main__":
    main()

