#!/usr/bin/env python3
"""BCN-84: activation-amax calibration for the Flash-Next FP8-periphery input_scale.

Runs on a rented GPU (B200 + CPU offload). Loads the OFFICIAL BF16
Qwen/Qwen3.8-Flash-Next via transformers/accelerate, registers forward
pre-hooks on the exact 300 linear modules the BCN-79 FP8-periphery
checkpoint quantized, runs a mixed calibration set (prose + code + chat-
templated prompts), and records the running max |input| per module.
input_scale is derived at graft time on the Mac (amax / 448, with a
disclosed safety multiplier) — this job records raw amax only.

Contract with the gpu-broker (custom rental):
  GET /health  -> 200 immediately (readiness; keeps the broker from
                  killing the pod during the 335 GB download)
  GET /status  -> JSON stage/progress/errors — progress is OBSERVABLE
                  (campaign lesson: silence must differ from progress)
  GET /result  -> the amax JSON once stage == "done" (404 before)
  GET /logtail -> last 100 log lines

Fail-fast checks (lessons from the failed rung A):
  - discovered target modules must EXACTLY match the expected 300-name
    set (count and names) or the run aborts before spending GPU-hours;
  - every batch asserts finite amax values;
  - any exception -> stage=failed with full traceback in /status.
"""
import http.server
import json
import os
import random
import re
import socketserver
import threading
import time
import traceback

PORT = int(os.environ.get("PORT", "8000"))
MODEL_ID = os.environ.get("CALIB_MODEL", "Qwen/Qwen3.8-Flash-Next")
N_SAMPLES = int(os.environ.get("CALIB_SAMPLES", "192"))
SEQ_LEN = int(os.environ.get("CALIB_SEQ_LEN", "2048"))
BATCH = int(os.environ.get("CALIB_BATCH", "4"))
GPU_MEM = os.environ.get("CALIB_GPU_MEM", "165GiB")
EXPECTED_TARGETS = int(os.environ.get("CALIB_EXPECTED_TARGETS", "300"))
WORKDIR = os.environ.get("CALIB_WORKDIR", "/workspace")
RESULT_PATH = os.path.join(WORKDIR, "calib_result.json")

TARGET_RES = [
    r"^model\.language_model\.layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"^model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
    r"^model\.language_model\.layers\.\d+\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$",
]

LOCK = threading.Lock()
STATE = {
    "stage": "boot", "detail": "", "progress": 0.0,
    "samples_done": 0, "samples_total": N_SAMPLES,
    "targets_found": 0, "targets_expected": EXPECTED_TARGETS,
    "started_ts": time.time(), "error": None, "done": False,
}
LOG: list = []


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOCK:
        LOG.append(line)
        del LOG[:-500]


def set_state(**kw):
    with LOCK:
        STATE.update(kw)


def is_target(name):
    return any(re.match(p, name) for p in TARGET_RES)


def load_corpus(processor):
    docs = []
    cdir = "/app/corpus"
    for fn in ("prose.txt", "code.txt"):
        p = os.path.join(cdir, fn)
        if os.path.exists(p):
            docs += [d for d in open(p, encoding="utf-8", errors="replace")
                     .read().split("\n\n=====DOC=====\n\n") if len(d) > 200]
    chats = []
    cp = os.path.join(cdir, "chat_prompts.json")
    if os.path.exists(cp):
        for prompt in json.load(open(cp)):
            try:
                chats.append(processor.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True))
            except Exception as e:
                log(f"chat template failed ({e!r}); using raw prompt")
                chats.append(prompt)
    rng = random.Random(20260827)
    rng.shuffle(docs)
    # ~1/4 chat-shaped (serving traffic is chat-formatted), rest raw docs
    samples = (chats * ((N_SAMPLES // 4) // max(1, len(chats)) + 1))[: N_SAMPLES // 4]
    samples += docs[: N_SAMPLES - len(samples)]
    rng.shuffle(samples)
    if len(samples) < N_SAMPLES:
        log(f"WARNING corpus smaller than requested: {len(samples)} < {N_SAMPLES}")
    return samples


def run():
    try:
        import torch  # noqa: deferred heavy imports until thread start

        set_state(stage="download", detail=MODEL_ID)
        log(f"downloading {MODEL_ID} ...")
        from huggingface_hub import snapshot_download
        t0 = time.time()
        local = snapshot_download(MODEL_ID)
        log(f"download done in {time.time()-t0:.0f}s -> {local}")

        set_state(stage="load", detail="from_pretrained device_map=auto")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(local)
        tok = getattr(processor, "tokenizer", processor)

        # cap CPU memory to what the host actually has, minus margin
        avail_gb = 64
        try:
            for ln in open("/proc/meminfo"):
                if ln.startswith("MemAvailable"):
                    avail_gb = int(ln.split()[1]) // (1024 * 1024)
        except Exception:
            pass
        cpu_mem = f"{max(32, avail_gb - 40)}GiB"
        log(f"host MemAvailable ~{avail_gb} GiB -> cpu max_memory {cpu_mem}")

        try:
            from transformers import AutoModelForMultimodalLM as AutoCls
        except ImportError:
            from transformers import AutoModelForCausalLM as AutoCls
            log("AutoModelForMultimodalLM unavailable; using AutoModelForCausalLM")

        t0 = time.time()
        model = AutoCls.from_pretrained(
            local, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory={0: GPU_MEM, "cpu": cpu_mem},
            offload_folder=os.path.join(WORKDIR, "offload"),
        )
        model.eval()
        log(f"model loaded in {time.time()-t0:.0f}s")

        # ---- discover + verify targets (fail fast BEFORE burning GPU-hours)
        amax: dict = {}
        counts: dict = {}
        hooks = []
        matched = [n for n, m in model.named_modules() if is_target(n)]
        set_state(targets_found=len(matched))
        log(f"matched {len(matched)} target modules (expected {EXPECTED_TARGETS})")
        if len(matched) != EXPECTED_TARGETS:
            sample = matched[:5]
            near = [n for n, _ in model.named_modules()
                    if ("linear_attn" in n or "shared_expert" in n)][:10]
            raise RuntimeError(
                f"target-count mismatch: found {len(matched)}, expected "
                f"{EXPECTED_TARGETS}. matched[:5]={sample} nearby={near}")

        def mk_hook(name):
            def hook(module, args, kwargs=None):
                x = args[0] if args else None
                if x is None or not hasattr(x, "abs"):
                    return
                v = float(x.detach().abs().amax().item())
                if v != v or v == float("inf"):
                    raise RuntimeError(f"non-finite input amax at {name}")
                if v > amax.get(name, 0.0):
                    amax[name] = v
                counts[name] = counts.get(name, 0) + 1
            return hook

        for n, m in model.named_modules():
            if is_target(n):
                hooks.append(m.register_forward_pre_hook(mk_hook(n)))

        set_state(stage="corpus")
        samples = load_corpus(processor)
        total = len(samples)
        set_state(samples_total=total)

        set_state(stage="calibrate")
        done = 0
        with torch.no_grad():
            for i in range(0, total, BATCH):
                batch = samples[i:i + BATCH]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=SEQ_LEN)
                enc = {k: v.to("cuda:0") for k, v in enc.items()}
                model(**enc)
                done += len(batch)
                set_state(samples_done=done, progress=done / total)
                if (i // BATCH) % 4 == 0:
                    log(f"calibrated {done}/{total}")

        for h in hooks:
            h.remove()

        # every target must have fired every batch
        missing = [n for n in matched if counts.get(n, 0) == 0]
        if missing:
            raise RuntimeError(f"{len(missing)} targets never received input: {missing[:5]}")

        result = {
            "model": MODEL_ID, "n_samples": total, "seq_len": SEQ_LEN,
            "batch": BATCH, "ts": time.time(),
            "amax": amax, "hook_counts": counts,
        }
        with open(RESULT_PATH, "w") as f:
            json.dump(result, f)
        log(f"result written: {RESULT_PATH} ({len(amax)} scales)")
        set_state(stage="done", done=True, progress=1.0)
    except Exception:
        err = traceback.format_exc()
        log("FAILED:\n" + err)
        set_state(stage="failed", error=err)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        elif self.path == "/status":
            with LOCK:
                self._send(200, dict(STATE))
        elif self.path == "/logtail":
            with LOCK:
                self._send(200, {"log": LOG[-100:]})
        elif self.path == "/result":
            if os.path.exists(RESULT_PATH):
                self._send(200, open(RESULT_PATH, "rb").read())
            else:
                self._send(404, {"error": "not ready", "stage": STATE["stage"]})
        else:
            self._send(404, {"error": "unknown path"})


if __name__ == "__main__":
    threading.Thread(target=run, daemon=True).start()
    with socketserver.ThreadingTCPServer(("", PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        log(f"serving on :{PORT}")
        httpd.serve_forever()
