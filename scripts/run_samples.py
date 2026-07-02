"""Replay the sample conversations against the /chat API and save actual responses.

Usage:
    uv run python scripts/run_samples.py [--url http://localhost:8000] [--out results_dir]

For each sample_conversations/GenAI_SampleConversations/C*.md:
  - extract the user turns
  - POST them turn-by-turn to /chat, feeding each response back as the assistant
    message (same convention as scripts/next_turn.py)
  - sleep TURN_DELAY seconds between turns (avoid LLM rate limits) and
    CONV_DELAY seconds between conversations
  - write <out>/C<n>_actual.json with the full transcript
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

TURN_DELAY = 15
CONV_DELAY = 30

CONV_DIR = Path(__file__).resolve().parent.parent / "sample_conversations" / "GenAI_SampleConversations"


def parse_turns(md_path: Path) -> list[dict]:
    """Return [{'user': str, 'expected_agent': str}, ...] from a sample transcript."""
    text = md_path.read_text()
    turns = []
    # Split on "### Turn N"
    for block in re.split(r"### Turn \d+", text)[1:]:
        user_match = re.search(r"\*\*User\*\*\n\n((?:>.*\n?)+)", block)
        if not user_match:
            continue
        user = "\n".join(
            line[1:].lstrip() for line in user_match.group(1).strip().splitlines()
        ).strip()
        agent_match = re.search(r"\*\*Agent\*\*\n\n(.*?)(?=\n_`end_of_conversation`|\Z)", block, re.S)
        expected = agent_match.group(1).strip() if agent_match else ""
        turns.append({"user": user, "expected_agent": expected})
    return turns


def post_chat(url: str, messages: list[dict]) -> dict:
    req = urllib.request.Request(
        f"{url}/chat",
        data=json.dumps({"messages": messages}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default="scratch_results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(CONV_DIR.glob("C*.md"), key=lambda p: int(p.stem[1:]))
    for ci, md in enumerate(files):
        turns = parse_turns(md)
        messages: list[dict] = []
        transcript = []
        print(f"=== {md.stem} ({len(turns)} turns) ===", flush=True)
        for ti, turn in enumerate(turns):
            messages.append({"role": "user", "content": turn["user"]})
            t0 = time.time()
            try:
                response = post_chat(args.url, messages)
            except Exception as e:  # keep going, record the failure
                response = {"error": str(e)}
            elapsed = round(time.time() - t0, 1)
            print(f"  turn {ti + 1}: {elapsed}s", flush=True)
            transcript.append(
                {
                    "turn": ti + 1,
                    "user": turn["user"],
                    "expected_agent": turn["expected_agent"],
                    "actual": response,
                    "latency_s": elapsed,
                }
            )
            messages.append({"role": "assistant", "content": json.dumps(response)})
            if ti < len(turns) - 1:
                time.sleep(TURN_DELAY)
        (out_dir / f"{md.stem}_actual.json").write_text(json.dumps(transcript, indent=2))
        if ci < len(files) - 1:
            print(f"  sleeping {CONV_DELAY}s before next conversation", flush=True)
            time.sleep(CONV_DELAY)
    print("done", flush=True)


if __name__ == "__main__":
    main()
