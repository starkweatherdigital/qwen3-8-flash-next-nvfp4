#!/usr/bin/env python3
"""Varied-geometry, seeded, accuracy-checked soak for prefix-cache changes.

Why this exists: mamba/GDN prefix-cache bugs are GEOMETRY-DEPENDENT. Repetitive
test prompts pass while a different mix of vocabulary, sentence lengths and
total length crashes the engine in one conversation — we shipped two
"fully passing" configurations that died on the first varied prompt. This
script is the test that catches them.

Three properties that matter:

1. VARIED  — every round builds a different prompt shape, so block boundaries
             land differently round to round.
2. SEEDED  — prompts derive from a seeded RNG, so a round that crashes the
             engine is exactly reproducible. Re-run it with --replay <seed>
             after a fix: a byte-identical replay of the killer input is the
             sharpest verification signal available.
3. CHECKED — every turn asks something verifiable (arithmetic, recall of a
             planted number). These bugs corrupt output SILENTLY before they
             crash; a liveness-only soak will call a corrupting engine healthy.

Usage:
    python3 soak.py --url http://localhost:8000/v1 --rounds 8
    python3 soak.py --url ... --replay 7919          # re-run one seed
    python3 soak.py --url ... --rounds 8 --base-seed 17

Exit codes: 0 = pass, 1 = accuracy failure, 2 = engine unreachable (crash).
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

WORDS = ["alpha", "bravo", "cadence", "delta", "ember", "falcon", "granite",
         "harbor", "indigo", "juniper", "kestrel", "lumen", "meridian", "nocturne"]


def chat(url, model, messages, max_tokens=100, timeout=600):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def build_round(seed):
    """Deterministic, geometry-varying conversation for one soak round."""
    rnd = random.Random(seed)
    paras = []
    for p in range(rnd.randint(20, 60)):
        sentence = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(120, 400)))
        paras.append(f"Section {p}: {sentence}.")
    filler = "\n\n".join(paras)
    a, b = rnd.randint(11, 29), rnd.randint(3, 9)
    planted = rnd.randint(1000, 9999)
    turns = [
        (f"{filler}\n\nIgnore all sections above. Remember the number {planted}. "
         f"What is {a}*{b}? Just the number.", str(a * b)),
        ("Add 100 to that. Just the number.", str(a * b + 100)),
        ("What number did I ask you to remember? Just the number.", str(planted)),
        ("Name the capital of Italy. One word.", "Rome"),
    ]
    return turns


def run_round(url, model, seed, verbose=True):
    turns = build_round(seed)
    messages, walls, failures = [], [], []
    prompt_tokens = 0
    for i, (question, expected) in enumerate(turns):
        messages.append({"role": "user", "content": question})
        try:
            d, dt = chat(url, model, messages)
        except (urllib.error.URLError, ConnectionError) as e:
            print(f"  seed {seed} turn {i+1}: ENGINE UNREACHABLE ({e}) "
                  f"— the engine most likely crashed", flush=True)
            return "unreachable", walls
        content = d["choices"][0]["message"]["content"] or ""
        prompt_tokens = d.get("usage", {}).get("prompt_tokens", 0)
        walls.append(dt)
        messages.append({"role": "assistant", "content": content})
        if expected.lower() not in content.lower():
            failures.append((i + 1, expected, content[:80]))
    if failures:
        for turn, expected, got in failures:
            print(f"  seed {seed} turn {turn}: WRONG ANSWER "
                  f"(expected {expected!r}, got {got!r})", flush=True)
        return "wrong", walls
    if verbose:
        print(f"  seed {seed}: OK  prompt={prompt_tokens} tok  "
              f"walls={[f'{w:.1f}s' for w in walls]}", flush=True)
    return "ok", walls


def cache_stats(url):
    """Hit rate from /metrics. 'Enabled' is not 'working' — this must rise."""
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
            text = r.read().decode()
    except Exception:
        return None
    hits = queries = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"):
            hits += float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_queries_total"):
            queries += float(line.split()[-1])
    return (hits, queries) if queries else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=1,
                    help="round N uses seed (base-seed + N) * 7919")
    ap.add_argument("--replay", type=int, default=None,
                    help="re-run exactly one seed (the killer-replay gate)")
    ap.add_argument("--sleep", type=int, default=240,
                    help="seconds between rounds (let the engine settle)")
    args = ap.parse_args()

    if args.replay is not None:
        print(f"KILLER REPLAY: seed {args.replay}")
        status, _ = run_round(args.url, args.model, args.replay)
        print("REPLAY:", "SURVIVED + ACCURATE" if status == "ok" else f"FAILED ({status})")
        return 0 if status == "ok" else (2 if status == "unreachable" else 1)

    print(f"SOAK: {args.rounds} varied-geometry rounds against {args.url}")
    for n in range(1, args.rounds + 1):
        seed = (args.base_seed + n) * 7919
        status, _ = run_round(args.url, args.model, seed)
        if status != "ok":
            print(f"SOAK FAIL at round {n} (seed {seed}, status={status}).")
            print(f"Reproduce it exactly:  python3 soak.py --url {args.url} "
                  f"--replay {seed}")
            return 2 if status == "unreachable" else 1
        if n < args.rounds:
            time.sleep(args.sleep)

    stats = cache_stats(args.url)
    if stats:
        hits, queries = stats
        pct = 100 * hits / queries if queries else 0
        print(f"prefix cache: {hits:.0f}/{queries:.0f} blocks ({pct:.0f}%)")
        if hits == 0:
            print("WARNING: caching is enabled but hit rate is exactly 0% — "
                  "see vllm#45238 (align keeps one checkpoint per request; "
                  "unlucky prompt geometry can silently defeat reuse).")
    print(f"SOAK PASS: {args.rounds} rounds clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
