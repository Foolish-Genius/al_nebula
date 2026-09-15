"""Render docs/report/report.html (with figures inlined) and print it to PDF with Chrome.

Figure tokens in the template look like {{fig:reports/eval-ihp-v1/eye_diagram.png}}
and are replaced by base64 data URIs so the HTML is self-contained; text
tokens like {{include:reports/eval-ihp-v1/sized_ctle.sp}} inline a file.

Example:
    python scripts/build_report.py --output docs/report/AutoAnalog-RL_report.pdf
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
)


def inline(template: str) -> str:
    def figure(match: re.Match) -> str:
        path = ROOT / match.group(1)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}[suffix]
        return f"data:{mime};base64,{data}"

    def include(match: re.Match) -> str:
        return html.escape((ROOT / match.group(1)).read_text(encoding="utf-8"))

    rendered = re.sub(r"\{\{fig:([^}]+)\}\}", figure, template)
    return re.sub(r"\{\{include:([^}]+)\}\}", include, rendered)


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    raise SystemExit("Chrome/Edge not found; open the HTML and print to PDF manually")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", default="docs/report/report.html")
    parser.add_argument("--output", default="docs/report/AutoAnalog-RL_report.pdf")
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    template = (ROOT / args.template).read_text(encoding="utf-8")
    rendered = inline(template)
    rendered_path = (ROOT / args.output).with_suffix(".rendered.html")
    rendered_path.write_text(rendered, encoding="utf-8")
    print(f"wrote {rendered_path}")
    if args.html_only:
        return
    chrome = find_chrome()
    output = ROOT / args.output
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={output}",
            rendered_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    print(f"wrote {output} ({output.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
