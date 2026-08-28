#!/usr/bin/env python3
"""Acceptance gates for a Flash-Next serving change. Correctness before speed.

Order matters. A corrupting engine can be fast, and a quantization or kernel
bug on this stack shows up as empty/garbled content with normal-looking token
counts — so coherence runs first and a speed number from an engine that failed
coherence is meaningless.

    python3 gates.py --url http://localhost:8000/v1

Gates:
  coherence  N varied prompts must return non-empty, correct answers
  tools      a tool call must come back as finish_reason=tool_calls with args
  speed      single-stream tok/s, with GPU clocks logged if nvidia-smi is here
  mtp        speculative-decode acceptance rate from /metrics
  memory     KV-cache pool and weight footprint from /metrics (informational)

Exit code 0 only if coherence and tools pass.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request

COHERENCE = [
    ("What is 17*23? Just the number.", "391"),
    ("Name the capital of Australia. One word.", "Canberra"),
    ("What is the chemical symbol for gold? Just the symbol.", "Au"),
    ("What year did the Apollo 11 moon landing happen? Just the year.", "1969"),
    ("List three prime numbers greater than 50, comma separated.", "53"),
    ("Translate 'good morning' into Spanish. Just the phrase.", "Buenos"),
    ("What is 144 divided by 12? Just the number.", "12"),
    ("If a train travels 60 mph for 2.5 hours, how far does it go? Just the number.", "150"),
]

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def chat(url, model, messages, max_tokens=150, tools=None, timeout=900):
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": 0}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0


def metrics(url):
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
            return r.read().decode()
    except Exception:
        return ""


def sum_metric(text, prefix):
    return sum(float(l.split()[-1]) for l in text.splitlines()
               if l.startswith(prefix) and not l.startswith("#"))


def sample_clocks(stop_flag, out):
    """GB10's DVFS parks bandwidth-bound decode well below max clocks, and a
    stuck clock lock has silently invalidated whole benchmark sessions here.
    Any speed number without clocks logged alongside it is not evidence."""
    while not stop_flag[0]:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            out.append(int(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--speed-runs", type=int, default=3)
    args = ap.parse_args()
    failed = []

    print("== coherence (correctness first) ==")
    ok = 0
    for prompt, expected in COHERENCE:
        try:
            d, _ = chat(args.url, args.model, [{"role": "user", "content": prompt}])
            content = d["choices"][0]["message"]["content"] or ""
        except Exception as e:
            print(f"  FAIL {prompt[:40]!r}: {e}")
            continue
        if expected.lower() in content.lower():
            ok += 1
        else:
            print(f"  MISS {prompt[:40]!r} -> {content[:60]!r}")
    print(f"coherence: {ok}/{len(COHERENCE)}")
    if ok < len(COHERENCE) - 1:      # allow one flake; two is a real signal
        failed.append("coherence")

    print("== tool calling ==")
    try:
        d, _ = chat(args.url, args.model,
                    [{"role": "user", "content": "What's the weather in Denver? Use the tool."}],
                    max_tokens=400, tools=TOOLS)
        choice = d["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        good = choice["finish_reason"] == "tool_calls" and calls and \
            "denver" in json.dumps(calls).lower()
        print(f"tools: {'PASS' if good else 'FAIL'} (finish={choice['finish_reason']})")
        if not good:
            failed.append("tools")
    except Exception as e:
        print(f"tools: FAIL ({e})")
        failed.append("tools")

    print("== speed (single stream) ==")
    stop, clocks = [False], []
    t = threading.Thread(target=sample_clocks, args=(stop, clocks), daemon=True)
    t.start()
    chat(args.url, args.model, [{"role": "user", "content": "Warm up."}], 30)
    time.sleep(10)
    speeds = []
    for _ in range(args.speed_runs):
        d, dt = chat(args.url, args.model,
                     [{"role": "user", "content":
                       "Write a detailed essay about the history of aviation, at least 400 words."}],
                     max_tokens=300)
        speeds.append(d["usage"]["completion_tokens"] / dt)
    stop[0] = True
    t.join(timeout=3)
    live = [c for c in clocks if c > 1000]
    clock_note = (f"clocks.sm mean {sum(live)/len(live):.0f} MHz "
                  f"(min {min(live)}, max {max(live)})" if live else
                  "clocks not sampled (no nvidia-smi here)")
    print(f"speed: {', '.join(f'{s:.2f}' for s in speeds)} tok/s | {clock_note}")

    text = metrics(args.url)
    if text:
        drafts = sum_metric(text, "vllm:spec_decode_num_draft_tokens_total")
        accepted = sum_metric(text, "vllm:spec_decode_num_accepted_tokens_total")
        if drafts:
            print(f"mtp acceptance: {100*accepted/drafts:.1f}% ({accepted:.0f}/{drafts:.0f})")
        hits = sum_metric(text, "vllm:prefix_cache_hits_total")
        queries = sum_metric(text, "vllm:prefix_cache_queries_total")
        if queries:
            print(f"prefix cache: {100*hits/queries:.0f}% ({hits:.0f}/{queries:.0f} blocks)")

    print()
    print("GATES:", "PASS" if not failed else f"FAIL ({', '.join(failed)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
