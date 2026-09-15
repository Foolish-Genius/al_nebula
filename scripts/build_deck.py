"""Build the demo-video slide deck (16:9 PowerPoint) with the narration in the speaker notes.

The slide order follows the video script (docs/video_shotlist.md); notes carry
the script verbatim, with suggested additions prefixed [ADDED].

Example:
    python scripts/build_deck.py --output docs/report/AutoAnalog-RL_deck.pptx --team "Team Name"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "figures"

NAVY = RGBColor(0x16, 0x3A, 0x6B)
BLUE = RGBColor(0x2B, 0x57, 0x97)
GREEN = RGBColor(0x1A, 0x7F, 0x37)
RED = RGBColor(0xB4, 0x23, 0x18)
GOLD = RGBColor(0xB7, 0x79, 0x1F)
GREY = RGBColor(0x55, 0x5B, 0x66)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
TERM_BG = RGBColor(0x0C, 0x1B, 0x2E)
TERM_FG = RGBColor(0xE6, 0xEE, 0xF8)

W, H = Inches(13.333), Inches(7.5)


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.count = 0

    def slide(self, title: str, timing: str, notes: str, subtitle: str | None = None):
        s = self.prs.slides.add_slide(self.blank)
        self.count += 1
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.0))
        bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.12), Inches(10.8), Inches(0.8))
        p = tb.text_frame.paragraphs[0]; p.text = title
        p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = WHITE
        tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        tm = s.shapes.add_textbox(Inches(11.2), Inches(0.28), Inches(1.9), Inches(0.5))
        q = tm.text_frame.paragraphs[0]; q.text = timing; q.font.size = Pt(12); q.font.color.rgb = RGBColor(0xC8, 0xD4, 0xE8); q.alignment = PP_ALIGN.RIGHT
        if subtitle:
            st = s.shapes.add_textbox(Inches(0.5), Inches(1.08), Inches(12.3), Inches(0.5))
            r = st.text_frame.paragraphs[0]; r.text = subtitle; r.font.size = Pt(15); r.font.color.rgb = GREY; r.font.italic = True
        foot = s.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(9), Inches(0.35))
        f = foot.text_frame.paragraphs[0]; f.text = f"AutoAnalog-RL  ·  Nebula: AI/ML for Analog Circuit Design  ·  Round 0  ·  {self.count}"; f.font.size = Pt(10); f.font.color.rgb = GREY
        s.notes_slide.notes_text_frame.text = notes
        return s

    def bullets(self, s, items, left, top, width, height, size=16, color=BLACK, spacing=6):
        tb = s.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame; tf.word_wrap = True
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "•  " + item; p.font.size = Pt(size); p.font.color.rgb = color; p.space_after = Pt(spacing)
        return tb

    def text(self, s, txt, left, top, width, height, size=16, color=BLACK, bold=False, mono=False, align=None, fill=None, wrap=True):
        tb = s.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame; tf.word_wrap = wrap
        if fill is not None:
            tb.fill.solid(); tb.fill.fore_color.rgb = fill
        for i, line in enumerate(txt.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line; p.font.size = Pt(size); p.font.color.rgb = color; p.font.bold = bold
            if mono:
                p.font.name = "Consolas"
            if align:
                p.alignment = align
        return tb

    def image(self, s, path, left, top, width=None, height=None):
        return s.shapes.add_picture(str(path), left, top, width=width, height=height)

    def table(self, s, rows, left, top, width, col_widths=None, size=12):
        shape = s.shapes.add_table(len(rows), len(rows[0]), left, top, width, Inches(0.38) * len(rows))
        t = shape.table
        if col_widths:
            for i, w in enumerate(col_widths):
                t.columns[i].width = w
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                cell = t.cell(r, c); cell.text = str(val)
                for p in cell.text_frame.paragraphs:
                    p.font.size = Pt(size); p.font.bold = r == 0
                    p.font.color.rgb = WHITE if r == 0 else BLACK
                cell.fill.solid(); cell.fill.fore_color.rgb = BLUE if r == 0 else (LIGHT if r % 2 == 0 else WHITE)
                cell.margin_left = cell.margin_right = Inches(0.06); cell.margin_top = cell.margin_bottom = Inches(0.03)
        return shape

    def box(self, s, label, left, top, width, height, color=NAVY, title_size=14, body_size=11):
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        b.fill.solid(); b.fill.fore_color.rgb = color; b.line.fill.background()
        tf = b.text_frame; tf.word_wrap = True
        for i, line in enumerate(label.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line; p.font.size = Pt(title_size if i == 0 else body_size); p.font.bold = i == 0; p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER
        return b

    def arrow(self, s, left, top, width, height=Inches(0.28), color=BLUE, shape=MSO_SHAPE.RIGHT_ARROW):
        a = s.shapes.add_shape(shape, left, top, width, height)
        a.fill.solid(); a.fill.fore_color.rgb = color; a.line.fill.background()
        return a

    def stat(self, s, big, small, left, top, width=Inches(2.9), color=NAVY, height=Inches(1.35)):
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        b.fill.solid(); b.fill.fore_color.rgb = LIGHT; b.line.color.rgb = BLUE
        tf = b.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = big; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = color; p.alignment = PP_ALIGN.CENTER
        q = tf.add_paragraph(); q.text = small; q.font.size = Pt(12); q.font.color.rgb = GREY; q.alignment = PP_ALIGN.CENTER


def pipeline(d, s, top, scale=1.0):
    """Agent -> environment -> evaluator -> ngspice -> outputs, with the LLM frontend below."""
    boxes = [
        ("SAC agent\nstable-baselines3", NAVY, 1.8), ("Environment\nΔ actions, reward,\ncorner curriculum", NAVY, 2.1),
        ("SpiceEvaluator\n1 .op (fail fast)\n2 .ac peaking, power\n3 PRBS7 .tran + channel → eye, DFE\n4 HD3 once others pass", BLUE, 3.4),
        ("ngspice 47 ×8\nIHP sg13g2 PSP103\nvia OpenVAF / OSDI", NAVY, 2.1), ("Outputs\nsized netlist\nJSON / CSV / PNG", GREEN, 1.8),
    ]
    x = Inches(0.4); gap = Inches(0.3); h = Inches(1.9 * scale)
    for i, (label, colr, w) in enumerate(boxes):
        w = Inches(w * scale)
        if i:
            d.arrow(s, x - gap + Inches(0.03), top + h / 2 - Inches(0.14), gap - Inches(0.06))
        d.box(s, label, x, top, w, h, color=colr)
        x += w + gap
    llm_left = Inches(0.4 + 1.8 * scale + 0.3 + 0.05)
    d.arrow(s, llm_left + Inches(0.95), top + h + Inches(0.05), Inches(0.28), Inches(0.35), color=GOLD, shape=MSO_SHAPE.UP_ARROW)
    d.box(s, "LLM frontend\nfeedback → reward weights", llm_left, top + h + Inches(0.45), Inches(2.1 * scale), Inches(0.75), color=GOLD, title_size=12, body_size=10)


def build(team: str) -> Presentation:
    d = Deck()

    # 1 ------------------------------------------------------------------ 0:00-0:25
    s = d.prs.slides.add_slide(d.blank); d.count += 1
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    d.text(s, "AutoAnalog-RL", Inches(0.8), Inches(1.8), Inches(11.5), Inches(1.2), size=54, color=WHITE, bold=True)
    d.text(s, "Automated Sizing of High-Speed Interface Circuits via Reinforcement Learning", Inches(0.8), Inches(2.9), Inches(11.5), Inches(0.8), size=24, color=WHITE)
    d.text(s, "PCIe Gen 2 (5 Gbps) receiver equalizer: one-stage CTLE + one-tap DFE  ·  SAC agent driving ngspice  ·  IHP sg13g2 130 nm open PDK",
           Inches(0.8), Inches(3.8), Inches(11.5), Inches(0.9), size=16, color=RGBColor(0xC8, 0xD4, 0xE8))
    d.text(s, "github.com/Foolish-Genius/al_nebula", Inches(0.8), Inches(4.9), Inches(11.5), Inches(0.5), size=16, color=WHITE, mono=True)
    d.text(s, f"{team}\nNebula — AI/ML for Analog Circuit Design  ·  Round 0  ·  15 September 2026", Inches(0.8), Inches(5.6), Inches(11.5), Inches(1.0), size=16, color=WHITE)
    s.notes_slide.notes_text_frame.text = (
        "0:00–0:25 — Introduction. Show: project title / README / repository homepage.\n\n"
        "Hello everyone. This is our project, AutoAnalog-RL — Automated Sizing of High-Speed Interface Circuits via Reinforcement Learning.\n\n"
        "Our objective is to automatically size a PCIe Gen 2 receiver equalizer consisting of a one-stage CTLE and a one-tap DFE, using reinforcement learning and circuit simulation.\n\n"
        "Instead of manually sweeping circuit parameters, our SAC agent interacts directly with ngspice and searches for a design that satisfies the required specifications.")

    # 2 ------------------------------------------------------------------ 0:25-1:00
    s = d.slide("Repository and overall flow", "0:25 – 1:00", notes=(
        "0:25–1:00 — Repository and overall flow. Show: GitHub repository → folders rl/, spice/, netlists/, scripts/, tests/.\n\n"
        "This is our project repository.\n\n"
        "The rl folder contains the specifications, reward function, environment, DFE and PVT implementation. The spice folder handles the simulator interface and measurements, while netlists contains the circuit topology.\n\n"
        "The main training and evaluation scripts are inside scripts, and we also have a regression test suite.\n\n"
        "The overall flow is: the SAC agent generates a normalized design action, the evaluator converts it into circuit parameters, generates the SPICE netlist, and runs the simulations. The resulting circuit metrics are then converted into a reward and sent back to the agent.\n\n"
        "Show: architecture diagram.\n\n"
        "The simulator uses the IHP sg13g2 130-nanometer open PDK with PSP103 models, running through ngspice.\n\n"
        "[ADDED] The brief allows either IHP or sky130; our synopsis named sky130 — we built on IHP because its PSP103 Verilog-A models run through ngspice's OSDI interface at about 0.6 seconds per RL step."))
    d.image(s, FIG / "repo_page_crop.png", Inches(0.4), Inches(1.25), height=Inches(4.5))
    d.text(s, "rl/ specs, reward, environment, DFE, PVT, LLM feedback · spice/ ngspice adapter and every gate · netlists/ CTLE topology · scripts/ train, evaluate, baselines, demo · tests/ 59 tests · tools/ Windows OSDI link shim",
           Inches(0.4), Inches(5.8), Inches(5.9), Inches(0.75), size=10, color=GREY)
    d.text(s, "Simulator: IHP sg13g2 130 nm, PSP103 via OpenVAF/OSDI in ngspice 47, 8 processes, ~0.6 s per step (brief allows IHP or sky130)",
           Inches(0.4), Inches(6.5), Inches(5.9), Inches(0.5), size=10, color=GREY)
    # pipeline on the right at reduced scale
    boxes = [
        ("SAC agent", NAVY), ("Environment", NAVY), ("SpiceEvaluator\n.op → .ac → PRBS .tran\n→ eye/DFE → HD3", BLUE), ("ngspice ×8", NAVY), ("Outputs", GREEN),
    ]
    y = Inches(1.4); xl = Inches(6.5); bw = Inches(6.4); bh = Inches(0.62)
    for i, (label, colr) in enumerate(boxes):
        if i:
            d.arrow(s, xl + bw / 2 - Inches(0.14), y - Inches(0.3), Inches(0.28), Inches(0.26), shape=MSO_SHAPE.DOWN_ARROW)
        height = Inches(1.0) if "\n" in label else bh
        d.box(s, label, xl, y, bw, height, color=colr, title_size=13, body_size=11)
        y += height + Inches(0.34)
    d.box(s, "LLM frontend: feedback → reward weights", xl, y + Inches(0.05), bw, Inches(0.5), color=GOLD, title_size=12)

    # 3 ------------------------------------------------------------------ 1:00-1:35
    s = d.slide("Circuit and parameters", "1:00 – 1:35", notes=(
        "1:00–1:35 — Circuit and parameters. Show: CTLE circuit diagram.\n\n"
        "The circuit being sized is a differential NMOS CTLE with resistive loads and RC source degeneration.\n\n"
        "The agent controls five CTLE parameters: input transistor width, load resistance, bias current, source degeneration resistance, and source degeneration capacitance.\n\n"
        "For the complete equalizer, a sixth parameter is added for the one-tap DFE.\n\n"
        "At every step, the evaluator checks the design progressively: first the DC operating point, then AC response and peaking, followed by the PRBS transient through the lossy channel, eye measurement and DFE, and finally the HD3 linearity check."))
    d.image(s, FIG / "ctle_schematic.png", Inches(0.5), Inches(1.3), height=Inches(3.9))
    d.table(s, [
        ["Parameter", "Range"], ["W_in  input width", "0.5 – 50 µm"], ["R_load", "100 – 1000 Ω"], ["I_bias", "0.1 – 2 mA"],
        ["R_s  degeneration", "10 – 500 Ω"], ["C_s  degeneration", "1 fF – 1 pF"], ["DFE tap (6th, equalizer)", "−0.5 … +0.5"],
    ], Inches(7.2), Inches(1.4), Inches(5.6), col_widths=[Inches(3.2), Inches(2.4)], size=13)
    gates = [("1  .op", "DC operating point\nfail fast"), ("2  .ac", "peaking @ Nyquist,\npower"), ("3  PRBS7 .tran", "lossy channel → eye,\none-tap DFE"), ("4  HD3", "100 MHz tone,\nonce others pass")]
    x = Inches(0.5)
    for i, (t, b) in enumerate(gates):
        if i:
            d.arrow(s, x - Inches(0.34), Inches(5.95), Inches(0.26))
        d.box(s, f"{t}\n{b}", x, Inches(5.45), Inches(2.85), Inches(1.3), color=BLUE if i != 3 else NAVY, title_size=13, body_size=11)
        x += Inches(2.85) + Inches(0.4)
    d.text(s, "Gates run in this order; a design that fails early costs no further simulation.", Inches(0.5), Inches(6.78), Inches(12), Inches(0.3), size=11, color=GREY)

    # 4 ------------------------------------------------------------------ 1:35-2:25 (a)
    s = d.slide("RL training: the SAC loop, live", "1:35 – 2:25", subtitle="python scripts/demo_rollout.py reports/sac-ihp-pvt   — a real rollout: random design, random PVT corner", notes=(
        "1:35–2:25 — RL training demo. Show: the captured rollout on this slide (optionally alt-tab to a terminal and run demo_rollout.py live), then the learning curves on the next slide.\n\n"
        "Now we move to the main part of the project — the reinforcement learning loop.\n\n"
        "We use a Soft Actor-Critic, or SAC, agent. The agent starts from a random circuit design and learns which parameter changes improve the circuit performance.\n\n"
        "The reward considers the specification violations as well as power and the margin from the required specifications.\n\n"
        "We also use fail-fast simulation, so a design that already fails an early gate does not unnecessarily spend simulation time on the later analyses.\n\n"
        "[ADDED] What you see here is the trained policy running from a random design at a random corner. Each line is one ngspice evaluation of a new design; the checklist on the right shows which specs pass. From a random start at a random corner, every spec is met within a few simulations and the sized netlist is written."))
    d.image(s, FIG / "demo_terminal.png", Inches(0.4), Inches(1.55), width=Inches(11.2))
    d.stat(s, "3 sims", "to meet every spec", Inches(0.6), Inches(5.35))
    d.stat(s, "7 s", "wall time on a laptop", Inches(3.7), Inches(5.35))
    d.stat(s, "SS / 1.0 V / 62 °C", "random corner, random start", Inches(6.8), Inches(5.35), width=Inches(3.4))
    d.text(s, "Reward = −Σ wᵢ·violationᵢ − power charge + success bonus + margin bonus\nDC failure: −10, fail fast", Inches(10.4), Inches(5.35), Inches(2.7), Inches(1.4), size=11, color=GREY)

    # 5 ------------------------------------------------------------------ 1:35-2:25 (b)
    s = d.slide("Training converges in ~2k steps; 60/60 rollouts feasible over three seeds", "1:35 – 2:25", notes=(
        "Show: learning curve / reward curve / feasible fraction (and a live-training capture from reports/screenshots if you want to show it ran).\n\n"
        "The training converges in approximately two thousand environment steps. Across the three base seeds, all 60 out of 60 deterministic rollouts reached a feasible design, with an average of about 2.5 steps to reach feasibility.\n\n"
        "These plots show the reward and feasibility improving during training, demonstrating that the agent is learning rather than simply performing a blind sweep."))
    d.image(s, FIG / "learning_curves.png", Inches(0.4), Inches(1.3), width=Inches(12.5))
    d.stat(s, "~2k steps", "to converge, every run", Inches(0.6), Inches(5.5), height=Inches(1.2))
    d.stat(s, "60 / 60", "rollouts feasible, 3 seeds", Inches(3.7), Inches(5.5), height=Inches(1.2))
    d.stat(s, "2.53 ± 0.06", "steps to feasibility (mean over seeds)", Inches(6.8), Inches(5.5), width=Inches(3.4), height=Inches(1.2))
    d.stat(s, "95 %", "of training steps meet every spec", Inches(10.4), Inches(5.5), width=Inches(2.6), height=Inches(1.2))

    # 6 ------------------------------------------------------------------ 2:25-3:15 (eye)
    s = d.slide("Result: post-channel eye diagram", "2:25 – 3:15", notes=(
        "2:25–3:15 — Show actual circuit results. Show: eye diagram first.\n\n"
        "Here is one of the key outputs — the post-channel eye diagram.\n\n"
        "For the seed-1 design, the measured eye height is about 0.327 volts with an eye width of 0.88 UI, which satisfies our internal target of 0.25 volts and 0.7 UI."))
    d.image(s, FIG / "eye_diagram.png", Inches(0.5), Inches(1.3), height=Inches(5.5))
    d.stat(s, "0.327 V", "eye height (target ≥ 0.25 V)", Inches(8.6), Inches(1.6), width=Inches(4.2))
    d.stat(s, "0.88 UI", "eye width (target ≥ 0.7 UI)", Inches(8.6), Inches(3.2), width=Inches(4.2))
    d.text(s, "Two-UI eye after the −10 dB channel, PRBS7 at 5 Gbps, seed-1 policy design. Measured against the transmitted bits: opening = min(ones) − max(zeros) at the best sampling phase.",
           Inches(8.6), Inches(4.8), Inches(4.2), Inches(1.8), size=12, color=GREY)

    # 7 ------------------------------------------------------------------ 2:25-3:15 (AC)
    s = d.slide("Result: AC response and peaking", "2:25 – 3:15", notes=(
        "Show: AC response plot.\n\n"
        "Next is the AC response. The design shows approximately 4.2 dB of high-frequency peaking at the 2.5 GHz Nyquist frequency, which is within our required 3 to 12 dB range.\n\n"
        "[ADDED] The Nyquist frequency is a specification input: retargeting the same policy to 1.25 gigahertz, PCIe Gen 1, gives 5.23 dB of peaking where the 2.5 gigahertz design gave only 2.87, with 20 out of 20 rollouts feasible."))
    d.image(s, FIG / "ac_response.png", Inches(0.5), Inches(1.3), width=Inches(8.0))
    d.stat(s, "4.2 dB", "peaking at 2.5 GHz (target 3–12 dB)", Inches(8.9), Inches(1.6), width=Inches(4.0))
    d.stat(s, "5.23 dB", "at 1.25 GHz after retuning (--spec nyquist_frequency_hz=1.25e9)", Inches(8.9), Inches(3.2), width=Inches(4.0))
    d.text(s, "Peaking = gain at Nyquist − DC gain, CTLE alone (channel bypassed for the AC gate). Tunable 1.25–2.5 GHz: the 2.5 GHz design gives only 2.87 dB at 1.25 GHz; the retuned policy moves R_s 182 → 365 Ω.",
           Inches(8.9), Inches(4.8), Inches(4.0), Inches(1.9), size=12, color=GREY)

    # 8 ------------------------------------------------------------------ 2:25-3:15 (PVT + table)
    s = d.slide("Result: 45 PVT corners and the design summary", "2:25 – 3:15", notes=(
        "Show: 45-corner plot / validation table, then the results table.\n\n"
        "The design is then validated across the complete 45 PVT corners. The peaking remains within the required range across all of these corners.\n\n"
        "[ADDED] The policy was also fine-tuned across all 45 corners: in evaluation, 24 of 24 rollouts that started at random corners reached a feasible design in a median of 2 steps.\n\n"
        "For the curriculum-trained design, the best result has an eye height of 0.370 volts, power of 1.12 milliwatts, and HD3 of minus 61.9 dB, while passing all 45 PVT corners."))
    d.image(s, FIG / "pvt_peaking.png", Inches(0.4), Inches(1.25), width=Inches(7.6))
    d.table(s, [
        ["Run", "Rollouts", "Steps", "Best design"],
        ["seed 1 / 2 / 3", "60/60", "2 – 3", "0.32–0.35 V · 1.2–1.3 mW · 45/45"],
        ["HD3 enforced", "20/20", "2.5", "0.335 V · 1.24 mW · −58 dB · 45/45"],
        ["PVT curriculum", "24/24 @ corners", "2", "0.370 V · 1.12 mW · −61.9 dB · 45/45"],
        ["Equalizer (+DFE)", "20/20", "3", "0.355 V post-DFE · 1.36 mW · 45/45"],
        ["Retune 1.25 GHz", "20/20", "2", "5.23 dB · 0.350 V · 1.28 mW · 45/45"],
    ], Inches(8.2), Inches(1.4), Inches(4.8), col_widths=[Inches(1.35), Inches(1.05), Inches(0.6), Inches(1.8)], size=10)
    d.text(s, "5 process × 3 supply (±5 %) × 3 temperature (0 / 62.5 / 125 °C) = 45 corners; every corner inside the 3–12 dB window. Curriculum: stage 1 nominal, stage 2 a random corner per episode.",
           Inches(0.5), Inches(5.0), Inches(7.5), Inches(1.0), size=12, color=GREY)
    d.stat(s, "45 / 45", "corners pass", Inches(8.4), Inches(4.7), width=Inches(2.2), height=Inches(1.1))
    d.stat(s, "24 / 24", "rollouts @ random corners", Inches(10.8), Inches(4.7), width=Inches(2.2), height=Inches(1.1))

    # 9 ------------------------------------------------------------------ 3:15-3:45
    s = d.slide("DFE: the complete equalizer", "3:15 – 3:45", notes=(
        "3:15–3:45 — DFE demo. Show: CTLE → channel → DFE diagram and DFE eye plot.\n\n"
        "We also extended the system to size the complete equalizer by adding a one-tap DFE.\n\n"
        "Importantly, we corrected the eye measurement so that the DFE eye is evaluated against the transmitted PRBS bits rather than the DFE's own decisions.\n\n"
        "With the corrected measurement, the DFE still provides a real improvement. One example improves the eye height from 0.327 volts to 0.455 volts with a tap of approximately plus 0.075, with no bit errors.\n\n"
        "[ADDED] We found and fixed two measurement flaws this way, by letting the agent exploit them: an FFT leakage floor that had faked an HD3 failure at minus 25 dB — the real value is minus 58 — and the self-labelled DFE eye, where the agent pushed the tap to its bound while the receiver got 54 of 127 bits wrong."))
    chain = [("Transmitter", "PRBS7, NRZ\n5 Gbps"), ("Channel", "2-section RC\n−10 dB @ Nyquist"), ("CTLE", "5 sized values\npeaking 3–12 dB"), ("Sampler", "UI-centre samples\naligned to the bits"), ("1-tap DFE", "y[n] − tap · d[n−1]\ntap sized by the agent"), ("Eye vs. bits", "min(ones) − max(zeros)\nbit errors counted")]
    x = Inches(0.4)
    for i, (t, b) in enumerate(chain):
        if i:
            d.arrow(s, x - Inches(0.27), Inches(1.95), Inches(0.2))
        d.box(s, f"{t}\n{b}", x, Inches(1.4), Inches(1.9), Inches(1.15), color=BLUE if i in (2, 4) else NAVY, title_size=12, body_size=10)
        x += Inches(1.9) + Inches(0.27)
    d.image(s, FIG / "eye_diagram_equalizer.png", Inches(0.5), Inches(2.8), height=Inches(3.9))
    d.stat(s, "0.327 → 0.455 V", "eye with a swept tap of +0.075, zero bit errors", Inches(6.4), Inches(2.9), width=Inches(3.2), height=Inches(1.2))
    d.stat(s, "0.51 V", "post-DFE eye held during equalizer training (CTLE-only ≈ 0.35 V)", Inches(9.8), Inches(2.9), width=Inches(3.2), height=Inches(1.2))
    d.text(s, "Two measurement flaws found by the agent and fixed:\n"
              "•  HD3: 2.5-cycle rectangular FFT window → leakage floor at −25 dB on every design. Integer-cycle Hann window → −58 dB (28 dB margin).\n"
              "•  DFE eye: labelled by the DFE's own decisions → a large tap 'opens' the eye by itself. Agent pushed the tap to +0.5 with 54/127 bit errors. Now measured against the transmitted bits; zero eye on any error.",
           Inches(6.4), Inches(4.3), Inches(6.6), Inches(2.5), size=11.5, color=BLACK)

    # 10 ----------------------------------------------------------------- 3:45-4:20
    s = d.slide("Compared with conventional search", "3:45 – 4:20", notes=(
        "3:45–4:20 — Compare with conventional search. Show: SAC vs random / CMA-ES graph and table.\n\n"
        "Finally, we compare the RL policy against conventional search methods.\n\n"
        "Random search requires about 13 simulations per feasible design, while CMA-ES requires around 9 simulations from random starting points.\n\n"
        "In comparison, the trained SAC policy reaches a feasible design in a median of 2 to 3 simulations, without performing a new optimization search for every starting point.\n\n"
        "We also tested the pre-ML AC-only bounded search. Although it could optimize the AC response, its resulting post-channel eye was only 0.160 volts, showing why optimizing only frequency-domain characteristics is not sufficient."))
    d.image(s, FIG / "budget_curve.png", Inches(0.4), Inches(1.3), width=Inches(5.6))
    d.table(s, [
        ["Method (random starting designs)", "First feasible design", "Per new instance"],
        ["SAC policy (144 rollouts)", "median 2–3, worst 5", "2–3 simulations, no search"],
        ["CMA-ES, random start (6 trials)", "median 9, worst 24", "~10 to feasible; 400 to refine"],
        ["Random search (3000 designs)", "#27 (7.9 % feasible)", "~13 per feasible design"],
        ["AC-only bounded search (pre-ML)", "never reaches the eye", "0.160 V post-channel eye"],
    ], Inches(7.0), Inches(1.4), Inches(6.0), col_widths=[Inches(2.5), Inches(1.8), Inches(1.7)], size=11)
    d.bullets(s, [
        "3–4× fewer simulations than CMA-ES from the same starts; ~10× fewer than random",
        "CMA-ES matches the policy's training-best quality given 400 sims per design — the policy's advantage is amortisation: train once, then 2–3 sims per new instance",
        "AC-only optimisation cannot see the eye: 0.160 V vs 0.327 V for the policy",
    ], Inches(7.0), Inches(3.9), Inches(6.0), Inches(2.8), size=12)
    d.image(s, FIG / "eye_diagram_preml.png", Inches(0.9), Inches(4.85), height=Inches(2.1))
    d.text(s, "AC-only search: 0.160 V / 0.58 UI", Inches(3.7), Inches(5.6), Inches(3.0), Inches(0.6), size=12, color=RED, bold=True)

    # 11 ----------------------------------------------------------------- 4:20-4:45
    s = d.slide("Reproducibility and the final output", "4:20 – 4:45", notes=(
        "4:20–4:45 — Reproducibility / final output. Show: terminal command → generated sized_ctle.sp → JSON/CSV/PNG output.\n\n"
        "The entire flow is reproducible through the scripts in our repository.\n\n"
        "The training and evaluation scripts generate JSON, CSV and PNG evidence, and every validated design also produces a sized SPICE netlist containing the circuit parameters, model information and DFE tap.\n\n"
        "So the final output is not just an RL score — it is an actual sized circuit together with its simulation and validation results.\n\n"
        "[ADDED, optional 15 s] Show the LLM frontend: run reward_from_feedback.py with a sentence such as \"power matters much more than eye margin\" and show the weights JSON that feeds --reward-settings."))
    d.text(s, "python scripts/train_sac.py --model-source ihp --eye-height-min 0.25 --hd3 --timesteps 12000 --n-envs 8 ...\n"
              "python scripts/train_sac.py ... --corners all --resume reports/sac-ihp-hd3/sac_final.zip --timesteps 8000\n"
              "python scripts/evaluate_policy.py reports/sac-ihp-pvt --rollouts 24 --baseline reports/baseline-ihp-3000\n"
              "python scripts/reward_from_feedback.py \"power matters much more than eye margin\" --output reports/reward.json",
           Inches(0.4), Inches(1.3), Inches(12.5), Inches(1.45), size=10.5, mono=True, fill=TERM_BG, color=TERM_FG, wrap=False)
    d.text(s, "* AutoAnalog-RL sized CTLE\n* model source: ihp_ngspice\n* W_in = 2.01908e-05\n* R_load = 1000\n* I_bias = 0.000930072\n* R_s = 260.642\n* C_s = 3.2956e-13\n.param VDD=1.2\nX1 outP inP sourceP 0 sg13_lv_nmos W=2.01908e-05 L=0.13u\nX2 outN inN sourceN 0 sg13_lv_nmos W=2.01908e-05 L=0.13u\nRdegP sourceP tail 260.642\n...",
           Inches(0.4), Inches(3.0), Inches(5.6), Inches(3.6), size=10.5, mono=True, fill=LIGHT)
    d.text(s, "sized_ctle.sp — the deliverable schematic", Inches(0.4), Inches(6.6), Inches(5.6), Inches(0.35), size=11, color=GREY)
    d.bullets(s, [
        "validation.json — every gate result, 45-corner PVT, HD3, noise, area",
        "steps.csv / monitor files / checkpoints — the whole training trajectory",
        "budget_curve, eye, AC, PVT plots — PNG + CSV",
        "evaluation.json — rollouts, steps to feasibility, per-corner breakdown",
        "reward.json — weights from plain-language feedback (LLM frontend)",
        "59 tests run without ngspice: python -m pytest -q",
    ], Inches(6.4), Inches(3.0), Inches(6.6), Inches(3.8), size=13)

    # 12 ----------------------------------------------------------------- 4:45-5:00
    s = d.slide("Summary", "4:45 – 5:00", notes=(
        "4:45–5:00 — Closing. Show: final result summary / repository.\n\n"
        "To summarize, AutoAnalog-RL combines reinforcement learning, SPICE simulation and PVT-aware validation to automatically size a high-speed receiver equalizer.\n\n"
        "Our trained policy achieves feasible designs in only a few simulations, passes the required circuit specifications and all 45 PVT corners, and produces a corresponding sized netlist.\n\n"
        "Thank you."))
    d.stat(s, "2 – 3 sims", "from a random start to every spec met", Inches(0.6), Inches(1.5), width=Inches(3.9))
    d.stat(s, "45 / 45", "PVT corners, every design", Inches(4.7), Inches(1.5), width=Inches(3.9))
    d.stat(s, "sized_ctle.sp", "an actual circuit, not a score", Inches(8.8), Inches(1.5), width=Inches(3.9))
    d.bullets(s, [
        "SAC agent + ngspice gates in fail-fast order, on the real IHP sg13g2 PSP103 models",
        "Eye 0.32–0.37 V after a −10 dB channel, 3–12 dB peaking, ~1 mW, HD3 ≈ −60 dB, noise 0.24 mV rms",
        "Curriculum across 45 corners · tunable Nyquist 1.25–2.5 GHz · one-tap DFE in the action · LLM reward frontend",
        "Random search ~13 and CMA-ES ~9 simulations per feasible design; the policy 2–3, with no per-instance search",
        "github.com/Foolish-Genius/al_nebula — report, deck, 59 tests, every artifact",
    ], Inches(0.6), Inches(3.3), Inches(12.2), Inches(3.0), size=15)
    d.text(s, "Thank you", Inches(0.6), Inches(6.3), Inches(12.2), Inches(0.6), size=24, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

    return d.prs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/report/AutoAnalog-RL_deck.pptx")
    parser.add_argument("--team", default="Team: TBD")
    args = parser.parse_args()
    prs = build(args.team)
    out = ROOT / args.output
    prs.save(str(out))
    print(f"wrote {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
