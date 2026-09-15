"""Turn natural-language design feedback into a reward-settings JSON for train_sac.py.

Example:
    python scripts/reward_from_feedback.py "power matters much more than eye margin" \
        --output reports/reward_low_power.json
    python scripts/train_sac.py ... --reward-settings reports/reward_low_power.json

Uses Claude (Anthropic SDK; credentials from ANTHROPIC_API_KEY or `ant auth login`)
and falls back to a keyword parser when no credentials are available; --offline
forces the fallback, --llm forces Claude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.feedback import RewardSettings, settings_from_feedback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("feedback", help="what the engineer wants, in plain language")
    parser.add_argument("--current", default=None, help="existing reward-settings JSON to adjust (default: reward defaults)")
    parser.add_argument("--output", default=None, help="write the settings JSON here (default: print only)")
    parser.add_argument("--model", default="claude-opus-5")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="keyword fallback only")
    mode.add_argument("--llm", action="store_true", help="require Claude; fail instead of falling back")
    args = parser.parse_args()

    current = RewardSettings.from_dict(json.loads(Path(args.current).read_text(encoding="utf-8"))) if args.current else None
    use_llm = True if args.llm else (False if args.offline else None)
    settings = settings_from_feedback(args.feedback, current, use_llm=use_llm, model=args.model)
    payload = {"feedback": args.feedback, **settings.to_dict()}
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()
