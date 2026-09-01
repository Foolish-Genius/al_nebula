"""Persist validation data and plots for reproducible design decisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class ValidationReporter:
    """Write summary data and useful validation plots for one candidate."""

    def __init__(self, output_dir: str | Path = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        metrics: Mapping[str, Any],
        *,
        ac_frequency_hz: Sequence[float] | None = None,
        ac_gain_db: Sequence[float] | None = None,
        transient_time_s: Sequence[float] | None = None,
        transient_output_v: Sequence[float] | None = None,
        pvt_results: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Write JSON/CSV summaries and any plots for available validation data."""
        json_path = self.output_dir / "validation.json"
        serializable = {key: self._json_value(value) for key, value in metrics.items()}
        json_path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
        paths = {"json": str(json_path)}

        if ac_frequency_hz is not None and ac_gain_db is not None:
            paths["ac_csv"] = self._write_pairs("ac_response.csv", "frequency_hz", "gain_db", ac_frequency_hz, ac_gain_db)
            paths["ac_plot"] = self._plot_ac(ac_frequency_hz, ac_gain_db)
        else:
            paths["ac_plot"] = self._plot_placeholder("AC response unavailable", "ac_response.png")
        if transient_time_s is not None and transient_output_v is not None:
            paths["tran_csv"] = self._write_pairs("transient.csv", "time_s", "differential_output_v", transient_time_s, transient_output_v)
            paths["tran_plot"] = self._plot_transient(transient_time_s, transient_output_v)
           paths["eye_plot"] = self._plot_eye(
    transient_time_s,
    transient_output_v,
    eye_height_v=metrics.get("eye_height_v"),
    eye_width_ui=metrics.get("eye_width_ui"),
    eye_height_pass=metrics.get("eye_height_pass"),
    eye_width_pass=metrics.get("eye_width_pass"),
)
        if pvt_results is not None:
            pvt_path = self.output_dir / "pvt_results.csv"
            rows = list(pvt_results)
            fields = sorted({key for row in rows for key in row})
            with pvt_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows({key: self._json_value(value) for key, value in row.items()} for row in rows)
            paths["pvt_csv"] = str(pvt_path)
            paths["pvt_plot"] = self._plot_pvt(rows)
        else:
            paths["pvt_plot"] = self._plot_placeholder("PVT results unavailable", "pvt_peaking.png")
        return paths

    def write_search(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        """Write bounded-search candidates and a score-versus-evaluation plot."""
        path = self.output_dir / "search_candidates.csv"
        fields = ["evaluation", "score", "dc_valid", "peaking_boost", "power"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({field: self._json_value(row.get(field)) for field in fields} for row in rows)
        plot = self._pyplot()
        figure, axis = plot.subplots(figsize=(8, 4.5))
        axis.plot([row["evaluation"] for row in rows], [row["score"] for row in rows], marker=".")
        axis.set(xlabel="evaluation", ylabel="objective score", title="bounded design search")
        axis.grid(True, alpha=0.25)
        return {"search_csv": str(path), "search_plot": self._save_plot(figure, "search_convergence.png")}

    def _write_pairs(self, filename: str, x_name: str, y_name: str, x: Sequence[float], y: Sequence[float]) -> str:
        path = self.output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow((x_name, y_name))
            writer.writerows(zip(x, y))
        return str(path)

    def _plot_ac(self, frequency_hz: Sequence[float], gain_db: Sequence[float]) -> str:
        plot = self._pyplot()
        figure, axis = plot.subplots(figsize=(8, 4.5))
        axis.semilogx(frequency_hz, gain_db)
        axis.axvline(2.5e9, color="tab:red", linestyle="--", label="2.5 GHz Nyquist")
        axis.set(xlabel="frequency (Hz)", ylabel="gain (dB)", title="CTLE AC response")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        return self._save_plot(figure, "ac_response.png")

    def _plot_transient(self, time_s: Sequence[float], output_v: Sequence[float]) -> str:
        plot = self._pyplot()
        figure, axis = plot.subplots(figsize=(8, 4.5))
        axis.plot(np.asarray(time_s) * 1e9, output_v)
        axis.set(xlabel="time (ns)", ylabel="differential output (V)", title="5 Gbps transient response")
        axis.grid(True, alpha=0.25)
        return self._save_plot(figure, "transient.png")

    def _plot_eye(
    self,
    time_s: Sequence[float],
    output_v: Sequence[float],
    *,
    eye_height_v: float | None = None,
    eye_width_ui: float | None = None,
    eye_height_pass: bool | None = None,
    eye_width_pass: bool | None = None,
) -> str:
    plot = self._pyplot()

    figure, axis = plot.subplots(figsize=(7, 5))

    time = np.asarray(time_s)
    values = np.asarray(output_v)

    ui = 200e-12

    if time.size and values.size:
        for start in np.arange(
            time.min(),
            time.max() - ui,
            ui,
        ):
            mask = (
                (time >= start)
                & (time < start + 2 * ui)
            )

            if mask.any():
                axis.plot(
                    (time[mask] - start) / ui,
                    values[mask],
                    alpha=0.12,
                )
    else:
        axis.text(
            0.5,
            0.5,
            "no transient data\n(gate failed)",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set(
        xlabel="unit intervals",
        ylabel="differential output (V)",
        title="Eye Diagram",
    )

    axis.grid(True, alpha=0.25)

    # Display measured eye-opening values.
    if eye_height_v is not None and np.isfinite(eye_height_v):
        height_status = (
            "PASS" if eye_height_pass else "FAIL"
        )

        axis.text(
            0.02,
            0.96,
            f"Vertical eye opening: "
            f"{eye_height_v:.3f} V — {height_status}",
            transform=axis.transAxes,
            verticalalignment="top",
        )

    if eye_width_ui is not None and np.isfinite(eye_width_ui):
        width_status = (
            "PASS" if eye_width_pass else "FAIL"
        )

        axis.text(
            0.02,
            0.89,
            f"Horizontal eye opening: "
            f"{eye_width_ui:.3f} UI — {width_status}",
            transform=axis.transAxes,
            verticalalignment="top",
        )

    return self._save_plot(figure, "eye_diagram.png")

def _plot_pvt(self, rows: Sequence[Mapping[str, Any]]) -> str:
    plot = self._pyplot()
    figure, axis = plot.subplots(figsize=(11, 5.5))

    names = [
        str(row.get("name", index))
        for index, row in enumerate(rows)
    ]

    values = np.asarray(
        [
            float(row.get("peaking_boost", np.nan))
            for row in rows
        ],
        dtype=float,
    )

    x = np.arange(len(names))

    axis.bar(x, values)

    axis.axhline(
        3.0,
        linestyle="--",
        label="minimum = 3 dB",
    )

    axis.axhline(
        12.0,
        linestyle="--",
        label="maximum = 12 dB",
    )

    finite = np.isfinite(values)
    passed = finite & (values >= 3.0) & (values <= 12.0)

    total = len(rows)
    valid = int(np.count_nonzero(finite))
    passed_count = int(np.count_nonzero(passed))

    axis.set(
        xlabel="PVT corner",
        ylabel="peaking boost (dB)",
        title=(
            f"PVT peaking verification "
            f"({passed_count}/{total} passed, "
            f"{valid}/{total} simulated)"
        ),
    )

    axis.set_xticks(
        x,
        names,
        rotation=90,
        fontsize=7,
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    return self._save_plot(
        figure,
        "pvt_peaking.png",
    )

    def _plot_placeholder(self, message: str, filename: str) -> str:
        plot = self._pyplot()
        figure, axis = plot.subplots(figsize=(8, 4.5))
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return self._save_plot(figure, filename)

    @staticmethod
    def _pyplot():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot
        return matplotlib.pyplot

    def _save_plot(self, figure: Any, filename: str) -> str:
        path = self.output_dir / filename
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        self._pyplot().close(figure)
        return str(path)

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating):
            return value.item() if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
