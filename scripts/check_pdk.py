"""Check whether the local IHP PDK can be compiled for ngspice OSDI."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def main() -> None:
    pdk = Path("/home/hp/ihp-open-pdk/ihp-sg13g2")
    compiler = shutil.which("openvaf-r") or shutil.which("openvaf")
    osdi_dir = pdk / "libs.tech/ngspice/osdi"
    print(f"pdk: {pdk.exists()}")
    compiler_status = "missing"
    if compiler:
        try:
            subprocess.run([compiler, "--version"], check=True, capture_output=True)
            compiler_status = "usable"
        except (OSError, subprocess.SubprocessError):
            compiler_status = "installed but unusable"
    print(f"openvaf: {compiler or 'missing'} ({compiler_status})")
    print(f"osdi_models: {len(tuple(osdi_dir.glob('*.osdi')))}")
    print("status: ready" if compiler_status == "usable" and any(osdi_dir.glob("*.osdi")) else "status: compiler/models required")


if __name__ == "__main__":
    main()
