"""Inline the recorded replay library into the studio page.

    python -m studio.build_page            -> docs/demo/autoanalog_studio.html

Without a library the page still works in live mode against `python -m studio.server`.
"""

from __future__ import annotations

import argparse

from studio.engine import ROOT

MARKER = "const LIBRARY = /*__LIBRARY__*/ null;"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", default="studio/web/index.html")
    parser.add_argument("--library", default="docs/demo/studio_library.json")
    parser.add_argument("--output", default="docs/demo/autoanalog_studio.html")
    args = parser.parse_args()

    page = (ROOT / args.template).read_text(encoding="utf-8")
    if MARKER not in page:
        raise SystemExit("template is missing the library marker")
    library = ROOT / args.library
    # "</" inside the netlist strings would close the script tag early.
    data = library.read_text(encoding="utf-8").replace("</", "<" + chr(92) + "/") if library.exists() else "null"
    output = ROOT / args.output
    output.write_text(page.replace(MARKER, "const LIBRARY = " + data + ";"), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size // 1024} KB, library {'inlined' if library.exists() else 'absent'})")


if __name__ == "__main__":
    main()
