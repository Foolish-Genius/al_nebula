"""Check whether the IHP sg13g2 OSDI models are available for ngspice.

Reports the PDK checkout ($IHP_PDK_ROOT or --pdk-root), the OpenVAF compiler
on PATH, and the compiled .osdi files the IHP evaluator loads. Compiling the
models on Windows without Visual Studio's C++ tools is covered in
tools/openvaf-link-shim/README.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spice.spice_engine import SpiceEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdk-root", default=os.environ.get("IHP_PDK_ROOT"))
    args = parser.parse_args()

    root = Path(args.pdk_root) if args.pdk_root else None
    ngspice_dir = root / "ihp-sg13g2" / "libs.tech" / "ngspice" if root else None
    print(f"pdk_root: {root or 'unset ($IHP_PDK_ROOT or --pdk-root)'} -> {'found' if root and root.exists() else 'missing'}")

    compiler = shutil.which("openvaf-r") or shutil.which("openvaf")
    compiler_status = "missing"
    if compiler:
        try:
            subprocess.run([compiler, "--version"], check=True, capture_output=True)
            compiler_status = "usable"
        except (OSError, subprocess.SubprocessError):
            compiler_status = "installed but unusable"
    print(f"openvaf: {compiler or 'missing'} ({compiler_status})")

    files = (SpiceEvaluator.PDK_MODEL_LIB, SpiceEvaluator.PDK_CORNER_LIB, *SpiceEvaluator.PDK_OSDI_MODELS)
    present = {name: bool(ngspice_dir and (ngspice_dir / name).exists()) for name in files}
    for name, ok in present.items():
        print(f"  {'ok     ' if ok else 'missing'} {name}")
    ngspice = os.environ.get("NGSPICE", "ngspice")
    print(f"ngspice: {shutil.which(ngspice) or ngspice}")
    print("status: ready" if all(present.values()) else "status: models required (compile with openvaf, see README)")


if __name__ == "__main__":
    main()
