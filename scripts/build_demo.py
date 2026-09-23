"""Inline the exported rollout into the demo page so it opens offline with a double-click.

Example:
    python scripts/export_demo_data.py reports/sac-ihp-pvt --seed 21 --pvt
    python scripts/build_demo.py --output docs/demo/autoanalog_demo.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", default="docs/demo/template.html")
    parser.add_argument("--data", default="docs/demo/rollout.json")
    parser.add_argument("--output", default="docs/demo/autoanalog_demo.html")
    args = parser.parse_args()

    template = (ROOT / args.template).read_text(encoding="utf-8")
    data = (ROOT / args.data).read_text(encoding="utf-8")
    marker = "const DATA = /*__DATA__*/ null;"
    if marker not in template:
        raise SystemExit("template is missing the data marker")
    # </script> inside the netlist string would close the tag early.
    page = template.replace(marker, "const DATA = " + data.replace("</", "<\\/") + ";")
    output = ROOT / args.output
    output.write_text(page, encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
