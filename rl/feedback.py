"""Natural-language design feedback -> reward weights.

The LLM frontend of AutoAnalog-RL: an engineer says what they care about
("power matters more than eye margin", "we can spend current, open the eye")
and the reward's per-spec weights are adjusted before (or between) training
runs. Claude produces the weights as structured JSON; a keyword fallback
keeps the flow working offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

SPEC_NAMES = ("peaking_boost", "peaking_ceiling", "power", "eye_horizontal_ui", "eye_vertical_v", "hd3")
DEFAULT_WEIGHTS = {name: 1.0 for name in SPEC_NAMES}
WEIGHT_BOUNDS = (0.1, 10.0)

SYSTEM_PROMPT = """You tune the reward of a reinforcement-learning agent that sizes a PCIe Gen 2 (5 Gbps) CTLE equalizer with ngspice.

The reward is: -sum_i weight_i * violation_i - efficiency_weight * (power / power_budget) + success_bonus (all specs met) + margin_weight * tightest_margin (when met).
Violations are normalised (1.0 = a full spec's worth of miss). Specs and their weight keys:
- peaking_boost: high-frequency peaking at 2.5 GHz must be >= 3 dB
- peaking_ceiling: the same peaking must be <= 12 dB
- power: supply power <= 2 mW
- eye_horizontal_ui: eye width after the channel >= 0.7 UI
- eye_vertical_v: eye height after the channel >= the eye target
- hd3: third-harmonic distortion <= -30 dB (only when enforced)
Other knobs: efficiency_weight (continuous charge for power even inside the feasible set, default 1.0), margin_weight (bonus for spec margin once feasible, default 5.0).

Given an engineer's feedback and the current values, return new values. Keep every weight within 0.1..10, change only what the feedback justifies, and explain briefly."""


@dataclass
class RewardSettings:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    efficiency_weight: float = 1.0
    margin_weight: float = 5.0
    rationale: str = ""
    source: str = "default"

    def clipped(self) -> "RewardSettings":
        low, high = WEIGHT_BOUNDS
        weights = {name: float(min(high, max(low, self.weights.get(name, 1.0)))) for name in SPEC_NAMES}
        return RewardSettings(
            weights=weights,
            efficiency_weight=float(min(high, max(0.0, self.efficiency_weight))),
            margin_weight=float(min(high, max(0.0, self.margin_weight))),
            rationale=self.rationale,
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RewardSettings":
        return cls(
            weights={**DEFAULT_WEIGHTS, **{k: float(v) for k, v in dict(data.get("weights", {})).items()}},
            efficiency_weight=float(data.get("efficiency_weight", 1.0)),
            margin_weight=float(data.get("margin_weight", 5.0)),
            rationale=str(data.get("rationale", "")),
            source=str(data.get("source", "file")),
        ).clipped()


# Keyword fallback: (pattern, keys to scale). "More"/"less" language sets direction,
# intensifiers set magnitude. Good enough to demo the loop without an API key.
_TOPICS = (
    (r"\b(power|current|battery|efficien\w*|mw|milliwatt)\b", ("power",), "efficiency_weight"),
    (r"\b(eye|margin|open\w*|height|vertical)\b", ("eye_vertical_v",), None),
    (r"\b(width|horizontal|jitter|timing)\b", ("eye_horizontal_ui",), None),
    (r"\b(peak\w*|boost|gain|equali[sz]\w*)\b", ("peaking_boost", "peaking_ceiling"), None),
    (r"\b(linear\w*|distortion|hd3|harmonic)\b", ("hd3",), None),
)
_LESS = r"\b(less|lower|ignore|relax|don'?t care|de-?prioriti[sz]e|cheaper|not important|doesn'?t matter)\b"
_MORE = r"\b(more|higher|priorit\w*|important|critical|matters|focus|maximi[sz]e|tighten|must|strict\w*|open|wider|bigger|larger|as much as possible|as (?:low|high) as possible)\b"
_STRONG = r"\b(much|strongly|far|really|top|most|only|absolutely|way)\b"


def _direction_at(text: str, start: int, end: int, factor: float) -> float | None:
    """Scale factor implied by the words around one topic mention, or None if neutral."""
    before = text[max(0, start - 30):start]
    window = text[max(0, start - 60):end + 40]
    # "X matters more than <topic>" demotes the topic even though "more" is nearby.
    if re.search(r"\bthan\s+(the\s+|our\s+|its\s+)?$", before) or re.search(_LESS, window):
        return 1.0 / factor
    if re.search(_MORE, window):
        return factor
    return None


def keyword_settings(feedback: str, current: RewardSettings | None = None) -> RewardSettings:
    """Offline fallback: scale each mentioned spec by the sentiment around its mention."""
    base = (current or RewardSettings()).clipped()
    text = feedback.lower()
    factor = 3.0 if re.search(_STRONG, text) else 2.0
    notes = []
    for pattern, keys, extra in _TOPICS:
        directions = [d for m in re.finditer(pattern, text) if (d := _direction_at(text, m.start(), m.end(), factor)) is not None]
        if not directions:
            continue
        direction = directions[0]
        for key in keys:
            base.weights[key] = base.weights[key] * direction
        if extra == "efficiency_weight":
            base.efficiency_weight *= direction
        notes.append(f"{'raised' if direction > 1 else 'lowered'} {', '.join(keys)} x{direction:.2g}")
    margin = re.search(r"\bmargin\b", text)
    if margin and (direction := _direction_at(text, margin.start(), margin.end(), factor)) is not None:
        base.margin_weight *= direction
        notes.append(f"{'raised' if direction > 1 else 'lowered'} margin_weight x{direction:.2g}")
    base.rationale = "keyword fallback: " + ("; ".join(notes) if notes else "no recognised topic, weights unchanged")
    base.source = "keyword"
    return base.clipped()


def claude_settings(feedback: str, current: RewardSettings | None = None, model: str = "claude-opus-5") -> RewardSettings:
    """Ask Claude for new reward settings as validated JSON."""
    import anthropic
    from pydantic import BaseModel, Field

    class Settings(BaseModel):
        weights: dict[str, float] = Field(description="one entry per spec key")
        efficiency_weight: float
        margin_weight: float
        rationale: str = Field(description="one or two sentences on what changed and why")

    base = (current or RewardSettings()).clipped()
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Current settings:\n{json.dumps({'weights': base.weights, 'efficiency_weight': base.efficiency_weight, 'margin_weight': base.margin_weight}, indent=2)}\n\n"
                f"Engineer feedback:\n{feedback.strip()}"
            ),
        }],
        output_format=Settings,
    )
    parsed = response.parsed_output
    return RewardSettings(
        weights={**DEFAULT_WEIGHTS, **parsed.weights},
        efficiency_weight=parsed.efficiency_weight,
        margin_weight=parsed.margin_weight,
        rationale=parsed.rationale,
        source=f"claude:{model}",
    ).clipped()


def settings_from_feedback(feedback: str, current: RewardSettings | None = None, *, use_llm: bool | None = None, model: str = "claude-opus-5") -> RewardSettings:
    """Claude when credentials are available (or use_llm=True), keyword fallback otherwise."""
    if use_llm is False:
        return keyword_settings(feedback, current)
    try:
        return claude_settings(feedback, current, model=model)
    except Exception as error:  # noqa: BLE001 - any SDK/auth/network failure falls back
        if use_llm is True:
            raise
        result = keyword_settings(feedback, current)
        reason = "no Anthropic credentials" if "authentication" in str(error).lower() else f"{type(error).__name__}: {error}"
        result.rationale += f" (Claude unavailable - {reason})"
        return result
