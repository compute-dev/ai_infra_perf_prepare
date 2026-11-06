#!/usr/bin/env python3
"""pdbench kit: single-node P-only / D-only tests against any OpenAI-compatible vLLM endpoint (protocol: ../PROTOCOL.md).

  pdbench.py gen-datasets --tokenizer <path> --out-dir <dir>
  pdbench.py validate-manifest <vendor_manifest.yaml|json>
  pdbench.py selftest-client --model-path <path>
  pdbench.py run --role p|d --base-url http://host:port --model-path <path> --served-model-name <name> \
                 --dataset-dir <dir> --out <dir> [--tiers "4 6"] [--repeat 2] [--manifest <manifest>] \
                 [--server-log <server log>] [--stop-after-fail 2] [--keep-texts]

Requires: the python3 standard library + `vllm bench serve` (the pinned client); reading a YAML manifest
needs pyyaml (a .json manifest works too). Every protocol constant lives in PROTO.
"""
import sys
if sys.version_info < (3, 8):   # must come before the other imports: on older Python, argparse fails with a confusing error
    sys.exit(f"pdbench needs Python >= 3.8, found {sys.version.split()[0]}. Use the pinned client image instead (client/Dockerfile).")
import argparse, csv, glob, hashlib, json, os, re, subprocess, sys, tempfile, threading, time, types, urllib.error, urllib.request

PROTO = dict(
    protocol_version="pdbench-1.0",
    ttft_slo_ms=7000.0, tpot_slo_ms=20.0, temperature=0.9, seed=0,
    p_prompts=80, p_warmups=10, d_warmups=2,
    fill_n_max=200, fill_conc=4, d_n_min=40, d_n_max=200, mult=10,   # fill_n_max: most prompts the replay set may use; the actual fill size is derived from the tiers by d_fill_n()
    expected_fingerprint="e9beb87e24e1", expected_rep_fingerprint="9adcdac59475",   # the latter: the _r2/_r3 repeat sets
    expected_tokenizer_fp="7f062efabe6a",   # combined fingerprint of the three tokenizer files of the official Hy3-FP8 release
    client_vllm_version="0.24.0",           # the pinned client vLLM version (public PyPI release; its request body was verified byte-identical to the earlier dev build)
    input_mean=84992, input_tol=0.02, prefix_tokens=78192, suffix_tokens=6800,
    p_hit_min=0.90, p_hit_max=0.95, d_hit_min=0.999,
    d_output_len=1200,   # the D replay pass emits a fixed 1200 tokens per request and ignores the dataset output_tokens column
                         # (variable-length output makes the per-tier mean output drift and drags the measured QPM down in the tail)
    d_touch_conc=4,      # concurrency of the per-tier re-touch pass (1 output token) run before each replay; see "per-tier re-touch" in PROTOCOL §6
    disp_warn=0.01, disp_fail=0.03, kv_margin=1.05,
)
DEFAULT_TIERS = dict(p="1 2 3 4 5 6 7 8", d="1 2 4 6 8 10 12 16 20")   # D ceiling c=20: a tier is capped at the 200-prompt replay set, so c>20 cannot meet C4 (>=10x concurrency)
P_FILE = "dataset_256k_prefix92_ponly100_c{c}.jsonl"
P_FILE_REP = "dataset_256k_prefix92_ponly100_c{c}_r{rep}.jsonl"   # repeats from the 2nd run on: same shared prefix, different unique suffix, so the previous pass cannot have cached the suffix
MAX_REP = 3
D_FILE = "dataset_256k_prefix92_replay200.jsonl"
KIT_PATH = os.path.abspath(__file__)
SELFTEST_DIR = os.path.join(os.path.dirname(KIT_PATH), "selftest")
GOLDEN = os.path.join(SELFTEST_DIR, "golden_requests.json")
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]   # chat_template determines the prompt the server actually sees

REQUIRED_MANIFEST = [
    "protocol_version", "vendor", "run_id",
    "hardware.gpu_model", "hardware.gpu_count", "hardware.gpu_mem_gib", "hardware.cpu_model", "hardware.cpu_cores",
    "hardware.ram_gib", "hardware.driver",
    "image.ref", "engine.vllm_version", "engine.fork", "engine.deviations",
    "precision.weights", "precision.kv_cache",
    "endpoints.p", "endpoints.d", "endpoints.hit_rate_source",
    "serve.p_cmd", "serve.d_cmd", "serve.tp", "serve.max_model_len", "serve.gpu_mem_util", "serve.prefix_caching", "serve.mtp_tokens",
    "attest.cold_compile_cache_cleared", "attest.exclusive_machine",
]


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(2)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- manifest
def load_manifest(path):
    txt = open(path, encoding="utf-8").read()
    if path.endswith(".json"): m = json.loads(txt)
    else:
        try: import yaml
        except ImportError: die("reading a YAML manifest needs pyyaml (pip install pyyaml); alternatively use a .json manifest")
        m = yaml.safe_load(txt)
    return m, hashlib.sha256(txt.encode()).hexdigest()


def dig(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d


def validate_manifest(m):
    errs = []
    for k in REQUIRED_MANIFEST:
        v = dig(m, k)
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, str) and v.strip().startswith("<")):
            errs.append(f"missing or not filled in: {k}")
    if dig(m, "protocol_version") not in (None, PROTO["protocol_version"]):
        errs.append(f"protocol_version={dig(m, 'protocol_version')} does not match the kit's {PROTO['protocol_version']}")
    if dig(m, "precision.weights") not in (None, "fp8") and not dig(m, "precision.rationale"):
        errs.append("precision.rationale is required when precision.weights is not fp8 (explain why the official Hy3-FP8 cannot run)")
    if dig(m, "serve.prefix_caching") is False: errs.append("serve.prefix_caching must be true")
    if dig(m, "endpoints.hit_rate_source") not in (None, "metrics", "usage"):
        errs.append("endpoints.hit_rate_source must be either metrics or usage")
    return errs


def cmd_validate_manifest(a):
    m, sha = load_manifest(a.manifest); errs = validate_manifest(m)
    if errs: print("manifest rejected:"); [print("  -", e) for e in errs]; sys.exit(1)
    print(f"manifest OK sha256={sha[:16]}…")


# ---------------------------------------------------------------- datasets
def fingerprint(d):
    """First 12 hex digits of the combined md5 of the 9 files; composed the same way as `md5sum f1 f2 ... | md5sum`."""
    names = [D_FILE] + [P_FILE.format(c=c) for c in range(1, 9)]
    lines = ""
    for n in names:
        p = os.path.join(d, n)
        if not os.path.exists(p): die(f"missing dataset file {p}")
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
        lines += f"{h.hexdigest()}  {n}\n"
    return hashlib.md5(lines.encode()).hexdigest()[:12]


def rep_fingerprint(d):
    """Extra fingerprint of the repeat-run datasets (_r2, _r3); returns None when they are absent. The base fingerprint from fingerprint() is unaffected."""
    lines = ""
    for r in range(2, MAX_REP + 1):
        for c in range(1, 9):
            p = os.path.join(d, P_FILE_REP.format(c=c, rep=r))
            if not os.path.exists(p): return None
            lines += f"{hashlib.md5(open(p, 'rb').read()).hexdigest()}  {os.path.basename(p)}\n"
    return hashlib.md5(lines.encode()).hexdigest()[:12]


def cmd_gen(a):
    here = os.path.dirname(KIT_PATH); gen = os.path.join(here, "dataset", "gen_prefix_dataset.py")
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(P_FILE.format(c=c), 100, 4000000 + c * 1000) for c in range(1, 9)] + [(D_FILE, 200, 5000000)]
    jobs += [(P_FILE_REP.format(c=c, rep=r), 100, 4000000 + c * 1000 + (r - 1) * 100000) for r in range(2, MAX_REP + 1) for c in range(1, 9)]
    for name, n, off in jobs:
        out = os.path.join(a.out_dir, name)
        if os.path.exists(out) and not a.force: print(f"skip {name} (already exists)"); continue
        r = subprocess.run([sys.executable, "-s", gen, "--tokenizer", a.tokenizer, "--scenario", "256k", "--prefix-hit-ratio", "0.92",
                            "--num-prompts", str(n), "--offset-base", str(off), "--out", out], capture_output=True, text=True,
                           cwd=os.path.dirname(gen))
        open(os.path.join(a.out_dir, name + ".gen.log"), "w").write(r.stdout + r.stderr)
        if r.returncode: die(f"failed to generate {name}; see {name}.gen.log")
        print(f"generated {name}")
    fp = fingerprint(a.out_dir); ok = fp == PROTO["expected_fingerprint"]
    rfp = rep_fingerprint(a.out_dir); rok = rfp == PROTO["expected_rep_fingerprint"]; ok = ok and rok
    print(f"repeat-set fingerprint: {rfp}  expected: {PROTO['expected_rep_fingerprint']}  {'match' if rok else '*** MISMATCH'}")
    print(f"fingerprint: {fp}  expected: {PROTO['expected_fingerprint']}  {'match' if ok else '*** MISMATCH: the requests differ, so the results are not comparable'}")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- metric collection
def scrape(url, names):
    """Sums vllm:<name> from /metrics (including labelled lines); returns None for anything unavailable."""
    try: text = urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception: return {n: None for n in names}
    out = {n: None for n in names}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip(): continue
        key = line.split()[0]
        for n in names:
            full = "vllm:" + n
            if key == full or key.startswith(full + "{"):
                out[n] = (out[n] or 0.0) + float(line.split()[-1])
    return out


HIT = ["prefix_cache_hits_total", "prefix_cache_queries_total"]
SPEC = ["spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"]
PRE = ["num_preemptions_total"]   # when KV is tight the server preempts running requests and recomputes them, inflating TPOT while the hit rate shows nothing (C15)


def delta(a, b, k): return None if a[k] is None or b[k] is None else b[k] - a[k]
def ratio(num, den): return (num / den) if num is not None and den else None


def env_snapshot():
    def sh(c):
        try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip() or None
        except Exception: return None
    return dict(client_vllm_version=sh("vllm --version 2>/dev/null | tail -1"), client_image=os.environ.get("PDBENCH_CLIENT_IMAGE"),
                gpu=sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sort | uniq -c"),
                kit_sha256=sha256_file(KIT_PATH), python=sys.version.split()[0])


# ---------------------------------------------------------------- client
def client_cmd(a, dataset, conc, out_len, n, warmups, outdir, name, pct):
    """The single place the client command is built: both the real load test and the client self-check (against the stub) go through it, so the self-check exercises the real command line."""
    return ["vllm", "bench", "serve", "--backend", "openai-chat", "--base-url", a.base_url, "--endpoint", "/v1/chat/completions",
            "--model", a.model_path, "--served-model-name", a.served_model_name, "--dataset-name", "custom", "--dataset-path", dataset,
            "--skip-chat-template", "--custom-output-len", str(out_len), "--num-prompts", str(n), "--seed", str(PROTO["seed"]),
            "--max-concurrency", str(conc), "--request-rate", "inf", "--num-warmups", str(warmups),
            "--temperature", str(PROTO["temperature"]), "--tokenizer", a.model_path, "--trust-remote-code", "--ignore-eos",
            "--percentile-metrics", pct, "--metric-percentiles", "50,99",
            "--save-result", "--save-detailed", "--result-dir", outdir, "--result-filename", name + ".json"]


def bench(a, dataset, conc, out_len, n, warmups, outdir, name, pct):
    cmd = client_cmd(a, dataset, conc, out_len, n, warmups, outdir, name, pct)
    with open(os.path.join(outdir, name + ".log"), "w") as lg:
        rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT).returncode
    p = os.path.join(outdir, name + ".json")
    if rc or not os.path.exists(p): die(f"vllm bench serve failed (rc={rc}); see {outdir}/{name}.log")
    r = json.load(open(p))
    if not a.keep_texts and "generated_texts" in r:   # the model output is bulky and not needed for evaluation; per-request timestamps and lengths are kept
        r.pop("generated_texts"); json.dump(r, open(p, "w"))
    return r


def n_lines(p): return sum(1 for l in open(p) if l.strip())


def d_tier_ceiling(tiers, dataset_dir):
    """D tier ceiling: a tier's request count is capped by the replay-set size (min(fill_n_max, replay-set lines)),
    while C4 requires at least 10x concurrency requests per tier. Both cannot hold once 10*c > the available
    lines, so such a tier is bound to fail C4. Returns (over-ceiling tiers, available lines, ceiling c)."""
    cap = min(PROTO["fill_n_max"], n_lines(os.path.join(dataset_dir, D_FILE)))
    return [c for c in tiers if PROTO["mult"] * c > cap], cap, cap // PROTO["mult"]


def d_fill_n(tiers, dataset_dir):
    """Actual fill size: fill only what this sweep needs (10x the highest tier); do not max out the replay set.
    Maxing it out only raises the resident footprint — the replay set resides as shared prefix + N*suffix, while
    the output blocks of past requests accumulate across tiers; once their sum crosses the KV capacity, every
    tier has to re-prefill during its re-touch pass (costing time). Prompts no tier uses are pure occupancy."""
    cap = min(PROTO["fill_n_max"], n_lines(os.path.join(dataset_dir, D_FILE)))
    return min(cap, max(PROTO["d_n_min"], PROTO["mult"] * max(tiers)))


# ---------------------------------------------------------------- pre-flight checks: tokenizer / client request body / server usage
def tokenizer_fingerprint(model_path):
    """Combined fingerprint of the three tokenizer files (first 12 hex digits of sha256); returns (None, missing list) if a file is absent."""
    lines, missing = "", []
    for n in TOKENIZER_FILES:
        p = os.path.join(model_path, n)
        if not os.path.exists(p): missing.append(n); continue
        lines += f"{sha256_file(p)}  {n}\n"
    return (None, missing) if missing else (hashlib.sha256(lines.encode()).hexdigest()[:12], [])


def canon_hash(body):
    """Canonical hash of a request body: independent of key order and whitespace."""
    return hashlib.sha256(json.dumps(json.loads(body), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def stub_server():
    """Local stub endpoint: records every POST body it receives and returns a valid streaming response (with usage) so the real client can run to completion."""
    import http.server
    rec = []

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b"{}")
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))); rec.append((self.path, body))
            try: req = json.loads(body)
            except Exception: req = {}
            mt = req.get("max_completion_tokens") or req.get("max_tokens") or 1
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            ev = lambda o: self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
            base = {"id": "stub", "object": "chat.completion.chunk", "model": req.get("model")}
            ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            ev({**base, "choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}]})
            ev({**base, "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": mt, "total_tokens": 100 + mt}})
            self.wfile.write(b"data: [DONE]\n\n")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, rec


def client_selftest(model_path, update_golden=False):
    """Drives the stub with the real client command line and compares the received request-body hashes against the golden set (hardware-independent; takes tens of seconds before a test)."""
    srv, rec = stub_server(); outdir = tempfile.mkdtemp(prefix="pdbench_selftest_")
    ns = types.SimpleNamespace(base_url=f"http://127.0.0.1:{srv.server_address[1]}", model_path=model_path, served_model_name="pdbench-selftest")
    cmd = client_cmd(ns, os.path.join(SELFTEST_DIR, "prompts.jsonl"), 1, -1, 3, 1, outdir, "selftest", "ttft,e2el")
    try: pr = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as e: srv.shutdown(); return dict(ok=False, detail=f"client failed to run: {e}")
    srv.shutdown()
    bodies = [b for p, b in rec if p.endswith("/v1/chat/completions")]
    if pr.returncode or len(bodies) < 3:
        return dict(ok=False, detail=f"client rc={pr.returncode}, stub received {len(bodies)} requests; {(pr.stderr or pr.stdout)[-300:]}")
    hashes = sorted({canon_hash(b) for b in bodies})
    def mask(o):   # keep the structure (whether content is a string or a segment list matters) and replace only long strings with a length placeholder
        if isinstance(o, str): return f"<{len(o)} chars>" if len(o) > 40 else o
        if isinstance(o, list): return [mask(x) for x in o]
        if isinstance(o, dict): return {k: mask(v) for k, v in o.items()}
        return o
    ex = mask(json.loads(bodies[0]))
    ver = subprocess.run("vllm --version 2>/dev/null | tail -1", shell=True, capture_output=True, text=True).stdout.strip()
    cur = dict(payload_hashes=hashes, example_payload=ex, recorded_with_client=ver)
    if update_golden:
        json.dump(cur, open(GOLDEN, "w"), indent=1, ensure_ascii=False); return dict(ok=True, detail=f"golden written ({len(hashes)} request-body hashes, client {ver})", hashes=hashes, example=ex)
    if not os.path.exists(GOLDEN): return dict(ok=False, detail=f"missing {GOLDEN}")
    g = json.load(open(GOLDEN)); ok = hashes == g["payload_hashes"]
    detail = f"request-body hashes {'match' if ok else '*** MISMATCH'} ({len(hashes)} bodies; golden recorded with client {g.get('recorded_with_client')}, current {ver})"
    if not ok:
        gk, ck = set(g["example_payload"]), set(ex)
        detail += f"; field diff extra={sorted(ck - gk)} missing={sorted(gk - ck)}; value diff " + str({k: (g["example_payload"][k], ex[k]) for k in gk & ck if g["example_payload"][k] != ex[k]})[:300]
    return dict(ok=ok, detail=detail, client_version=ver)


def probe_server(base_url, served_name):
    """Sends one small streaming request to the server under test: verifies that the stream carries usage (the client takes its input token count from it and silently falls back to its own count when it is absent, which voids C6) and that ignore_eos takes effect."""
    n = 8
    # content is a segment list, matching the real client (a server that does not accept that shape must surface here)
    body = {"model": served_name, "messages": [{"role": "user", "content": [{"type": "text", "text": "pdbench probe"}]}], "max_completion_tokens": n,
            "stream": True, "stream_options": {"include_usage": True}, "temperature": PROTO["temperature"], "ignore_eos": True}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode(errors="ignore").strip()
                if line.startswith("data:") and not line.endswith("[DONE]"):
                    d = json.loads(line[5:])
                    if d.get("usage"): usage = d["usage"]
    except urllib.error.HTTPError as e:
        return dict(ok=False, detail=f"probe request failed: HTTP {e.code} {e.read()[:300].decode(errors='ignore')}")
    except Exception as e:
        return dict(ok=False, detail=f"probe request failed: {e}")
    if not usage or not isinstance(usage.get("prompt_tokens"), int) or not isinstance(usage.get("completion_tokens"), int):
        return dict(ok=False, detail=f"the streaming response carries no complete usage (prompt_tokens/completion_tokens): {usage}")
    ok = usage["completion_tokens"] == n
    return dict(ok=ok, detail=f"usage complete (prompt_tokens={usage['prompt_tokens']}); asked for {n} output tokens, got {usage['completion_tokens']}" + ("" if ok else " (ignore_eos did NOT take effect)"))


def cmd_selftest(a):
    r = client_selftest(a.model_path, a.update_golden); print(("PASS " if r["ok"] else "FAIL ") + r["detail"]); sys.exit(0 if r["ok"] else 1)


# ---------------------------------------------------------------- single-role tests
def tier_row(role, c, r, hit, acc, rep, warmups_seen):
    n = r["completed"]
    row = dict(concurrency=c, rep=rep, num_prompts=r["num_prompts"], completed=n, mean_input_tokens=round(r["total_input_tokens"] / n),
               mean_output_tokens=round(r["total_output_tokens"] / n), mean_ttft_ms=r["mean_ttft_ms"], p50_ttft_ms=r["median_ttft_ms"],
               p99_ttft_ms=r["p99_ttft_ms"], qpm=r["request_throughput"] * 60, steady_qpm=c * 60000 / r["mean_e2el_ms"], prefix_hit=hit)
    if role == "d": row.update(mean_tpot_ms=r["mean_tpot_ms"], p99_tpot_ms=r["p99_tpot_ms"], mtp_accept=acc)
    row["_out_lens"] = r.get("output_lens"); row["_errors"] = [e for e in (r.get("errors") or []) if e]
    row["_warmups"] = warmups_seen
    return row


KEY = dict(p="mean_ttft_ms", d="mean_tpot_ms")
NUMS = ["mean_input_tokens", "mean_output_tokens", "mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms", "qpm", "steady_qpm", "prefix_hit",
        "mean_tpot_ms", "p99_tpot_ms", "mtp_accept"]


def aggregate(role, reps):
    """Several runs of one concurrency tier -> a mean row + the per-repeat detail + the verdict."""
    out = {k: v for k, v in reps[0].items() if not k.startswith("_") and k != "rep"}
    for k in NUMS:
        vs = [r[k] for r in reps if r.get(k) is not None]
        if vs: out[k] = sum(vs) / len(vs)
    out["runs"] = len(reps)
    pv = [r["preemptions"] for r in reps if r.get("preemptions") is not None]
    out["preemptions"] = max(pv) if pv else None   # the max across repeats; >0 means some run of this tier was preempted and recomputed
    p99k = "p99_ttft_ms" if role == "p" else "p99_tpot_ms"
    out["repeat"] = [dict(qpm=r["qpm"], **{KEY[role]: r[KEY[role]], p99k: r[p99k]}) for r in reps]   # per-run p99 is kept so an evaluation can apply a "every run must meet it" p99 rule
    slo = PROTO["ttft_slo_ms"] if role == "p" else PROTO["tpot_slo_ms"]
    out["verdict"] = "PASS" if out[KEY[role]] <= slo else "FAIL"
    return out


def spread(vs): return (max(vs) - min(vs)) / (sum(vs) / len(vs)) if len(vs) > 1 else None


def server_kv_tokens(path):
    if not path or not os.path.exists(path): return None
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", open(path, errors="ignore").read())
    return int(m.group(1).replace(",", "")) if m else None


def make_checks(role, raw, rows, a, m, fill, tiers, fp, ev=None):
    ck = []
    def add(i, ok, detail, status=None): ck.append(dict(id=i, status=status or ("PASS" if ok else "FAIL"), detail=detail))
    fmt = lambda f: ", ".join(f(r) for r in raw)
    add("C1", m is not None and dig(m, "protocol_version") == PROTO["protocol_version"],
        f"manifest protocol_version={dig(m, 'protocol_version') if m else None}; the kit's own sha256 is recorded in the result" if m else "no manifest given (--manifest), nothing to check against", None if m else "SKIP")
    add("C2", fp == PROTO["expected_fingerprint"], f"dataset fingerprint {fp}")
    ev = ev or {}
    tfp = ev.get("tokenizer_fp"); exp_t = PROTO["expected_tokenizer_fp"]
    add("C2t", exp_t is not None and tfp == exp_t, f"tokenizer three-file fingerprint {tfp}" + (f" (expected {exp_t})" if exp_t else " (the protocol has no expected value yet; recorded only)"),
        "SKIP" if (exp_t is None or "tokenizer_fp" not in ev) else None)
    st = ev.get("selftest"); cv = PROTO["client_vllm_version"]
    ver_ok = cv is None or (st or {}).get("client_version") == cv
    add("C13", bool(st and st["ok"] and ver_ok), (st["detail"] if st else "the client self-check did not run") + ("" if ver_ok else f"; the client version must be {cv}"), None if st else "SKIP")
    pb = ev.get("probe")
    add("C14", bool(pb and pb["ok"]), pb["detail"] if pb else "the server usage probe did not run", None if pb else "SKIP")
    add("C3", all(r["completed"] == r["num_prompts"] and not r["_errors"] for r in raw), "completed = planned, with no errors: " + fmt(lambda r: f"c{r['concurrency']}.{r['rep']}:{r['completed']}/{r['num_prompts']}"))
    add("C4", all(r["num_prompts"] >= 10 * r["concurrency"] for r in raw), "requests per tier >= 10x concurrency")
    lo = PROTO["p_hit_min"] if role == "p" else PROTO["d_hit_min"]
    hi = PROTO["p_hit_max"] if role == "p" else 1.01
    add("C5", all(r["prefix_hit"] is not None and lo <= r["prefix_hit"] <= hi for r in raw), f"hit rate {lo}~{hi if role == 'p' else 1.0}: " + fmt(lambda r: f"c{r['concurrency']}.{r['rep']}:{'NA' if r['prefix_hit'] is None else round(r['prefix_hit'], 3)}"))
    mean, tol = PROTO["input_mean"], PROTO["input_tol"]
    add("C6", all(abs(r["mean_input_tokens"] - mean) / mean <= tol for r in raw), f"mean input tokens {mean} +/-{tol:.0%}: " + fmt(lambda r: str(r["mean_input_tokens"])))
    if role == "p":
        add("C7", all(x["_out_lens"] and all(v == 1 for v in x["_out_lens"]) for x in raw), "P output length is 1 for every request (ignore_eos/max_tokens in effect)")
    else:
        fixed = PROTO["d_output_len"]   # every request must be exactly this length (ignore_eos in effect, nothing truncated)
        bad = sorted({r["concurrency"] for r in raw if not r["_out_lens"] or any(v != fixed for v in r["_out_lens"])})
        add("C7", not bad, f"D output length is {fixed} for every request (ignore_eos in effect)" + (f"; offending tiers {bad}" if bad else ""))
        add("C7f", fill["completed"] == fill["num_prompts"], f"cache fill completed {fill['completed']}/{fill['num_prompts']}")
    warm = PROTO["p_warmups"] if role == "p" else PROTO["d_warmups"]
    wok = all(r["_warmups"] == warm for r in raw)
    attest = bool(m and dig(m, "attest.cold_compile_cache_cleared") is True and dig(m, "attest.exclusive_machine") is True)
    add("C8", wok and attest, f"warmup={warm} requests (client argument check: {'match' if wok else 'mismatch'}); cold-compile and exclusivity are manifest declarations only ({'declared' if attest else 'not declared, or no manifest'})")
    groups = {}
    for r in raw: groups.setdefault(r["concurrency"], []).append(r)
    if max(len(g) for g in groups.values()) < 2:
        add("C9", False, "only 1 run per tier, so the spread cannot be assessed (the protocol requires >=2)", "FAIL" if m else "SKIP")
    else:
        worst, wc = 0.0, (None, None)   # when every repeat is identical worst=0, but wc must still be formattable
        for c, g in groups.items():
            if len(g) < 2: continue
            for k in (KEY[role], "qpm"):
                s = spread([x[k] for x in g])
                if s is not None and s > worst: worst, wc = s, (c, k)
        st = "FAIL" if worst > PROTO["disp_fail"] else "WARN" if worst > PROTO["disp_warn"] else "PASS"
        add("C9", st == "PASS", f"largest relative spread across repeats {worst:.2%} (c={wc[0]} {wc[1]}); warn>{PROTO['disp_warn']:.0%} block>{PROTO['disp_fail']:.0%}", st)
    src = dig(m, "endpoints.hit_rate_source") if m else None
    if src == "usage": add("C10", False, "the usage source is declared, but the kit does not implement usage.cached_tokens collection yet", "SKIP")
    else: add("C10", src == "metrics" and all(r["prefix_hit"] is not None for r in raw), f"declared source={src}; /metrics is {'collectable' if all(r['prefix_hit'] is not None for r in raw) else 'MISSING'} in practice", None if m else "SKIP")
    kv = server_kv_tokens(a.server_log)
    add("C11", kv is not None, f"the server log contains GPU KV cache size: {kv:,} tokens" if kv else "no --server-log given, or the log has no GPU KV cache size line", None if a.server_log else "SKIP")
    if role == "d":
        need = (PROTO["prefix_tokens"] + fill["num_prompts"] * PROTO["suffix_tokens"] + max(tiers) * PROTO["suffix_tokens"]) * PROTO["kv_margin"]
        add("C12", kv is not None and kv >= need, f"KV capacity {kv} >= (78192+{fill['num_prompts']}*6800+{max(tiers)}*6800)*{PROTO['kv_margin']} = {need:,.0f}" if kv else "no KV capacity data, so the fill capacity rule cannot be checked", None if kv else "SKIP")
    # C15: preemption counter. When KV is tight the server preempts running requests and recomputes them, which inflates that tier's latency while the hit rate may show nothing.
    pre = [r["preemptions"] for r in raw if r.get("preemptions") is not None]
    bad_pre = [f"c{r['concurrency']}.{r['rep']}:{r['preemptions']:.0f}" for r in raw if r.get("preemptions")]
    add("C15", not bad_pre, ("the preemption counter delta is 0 for every tier" if not bad_pre else "these runs were preempted and recomputed, so their tier latency is unusable: " + ", ".join(bad_pre))
        if pre else "the server /metrics has no num_preemptions_total, so no preemption counter was collected", None if pre else "SKIP")
    return ck


def capacity(role, rows):
    ok = [r for r in rows if r["verdict"] == "PASS"]
    first_fail = next((r["concurrency"] for r in rows if r["verdict"] == "FAIL"), None)
    f = lambda r: r and dict(concurrency=r["concurrency"], qpm=r["qpm"], steady_qpm=r["steady_qpm"])
    b = max(ok, key=lambda r: r["qpm"]) if ok else None
    # knee_found=False: every measured tier PASSed, so the knee lies at a higher concurrency and was not
    # reached. The capacity then reads as ">= the highest tier measured" and must not be presented as this
    # hardware's capacity (otherwise running out of tiers is indistinguishable from a real knee in result.json).
    knee = dict(knee_found=first_fail is not None, max_tier_measured=max((r["concurrency"] for r in rows), default=None))
    if role == "p": return dict(max_qpm=f(b), first_fail_concurrency=first_fail, **knee)
    contig = None
    for r in rows:
        if r["verdict"] == "PASS": contig = r
        else: break
    return dict(contiguous=f(contig), max_qpm=f(b), first_fail_concurrency=first_fail, **knee)


def cmd_run(a):
    role = a.role; tiers = [int(x) for x in (a.tiers or DEFAULT_TIERS[role]).split()]
    fp = fingerprint(a.dataset_dir)
    if fp != PROTO["expected_fingerprint"]: die(f"dataset fingerprint {fp} does not match the protocol's {PROTO['expected_fingerprint']} (run gen-datasets first)")
    if role == "p" and a.repeat > 1:
        rfp = rep_fingerprint(a.dataset_dir)
        if rfp != PROTO["expected_rep_fingerprint"]: die(f"repeat-set fingerprint {rfp} does not match the protocol's {PROTO['expected_rep_fingerprint']} (--repeat>1 needs the _r2.._r{a.repeat} datasets; run gen-datasets first)")
    if role == "d":
        over, cap, c_max = d_tier_ceiling(tiers, a.dataset_dir)
        if over: die(f"concurrency tiers {' '.join(map(str, over))} exceed the ceiling this replay set can support (c<={c_max}): "
                     f"a tier gets at most {cap} requests (the replay-set size), while the protocol requires >= 10x concurrency. "
                     f"Drop those tiers, or first raise fill_n_max and regenerate a larger replay set "
                     f"(which changes the dataset fingerprint and is therefore a protocol-level change).")
    m = None; msha = None
    if a.manifest:
        m, msha = load_manifest(a.manifest); errs = validate_manifest(m)
        if errs: die("manifest rejected:\n  - " + "\n  - ".join(errs))
    base = os.path.join(a.out, "p_only" if role == "p" else "d_only"); os.makedirs(base, exist_ok=True)
    # Pre-flight checks (tens of seconds): tokenizer matches, the client request body matches the golden set, the server stream carries usage.
    # Nothing is loaded unless they pass (--continue-on-preflight-fail is for debugging only).
    tfp, missing = tokenizer_fingerprint(a.model_path)
    ev = dict(tokenizer_fp=tfp, selftest=client_selftest(a.model_path), probe=probe_server(a.base_url, a.served_model_name))
    print("[preflight] tokenizer:", tfp or f"missing files {missing}", "| client self-check:", ev["selftest"]["detail"], "| server probe:", ev["probe"]["detail"], flush=True)
    pre_bad = [x for x in (not ev["selftest"]["ok"], not ev["probe"]["ok"],
                           PROTO["expected_tokenizer_fp"] is not None and tfp != PROTO["expected_tokenizer_fp"],
                           PROTO["client_vllm_version"] is not None and ev["selftest"].get("client_version") != PROTO["client_vllm_version"]) if x]
    if pre_bad and not a.continue_on_preflight_fail: die("the pre-flight checks did not pass (see above); no load was generated")
    murl = a.metrics_url or a.base_url.rstrip("/") + "/metrics"
    raw, rows, fill = [], [], None; t0 = time.time(); warm = PROTO["p_warmups"] if role == "p" else PROTO["d_warmups"]
    if role == "d":
        ds = os.path.join(a.dataset_dir, D_FILE); nfill = d_fill_n(tiers, a.dataset_dir)
        h0 = scrape(murl, HIT)
        fr = bench(a, ds, PROTO["fill_conc"], 1, nfill, PROTO["d_warmups"], base, "fill", "ttft,e2el")
        h1 = scrape(murl, HIT)
        fill = dict(num_prompts=nfill, completed=fr["completed"], hit_ratio=ratio(delta(h0, h1, HIT[0]), delta(h0, h1, HIT[1])))
        print(f"[fill] {fill['completed']}/{nfill} prompts, hit rate of this pass {fill['hit_ratio']}", flush=True)
    fails = 0
    for c in tiers:
        reps = []
        touch = None
        if role == "d":
            # Per-tier re-touch: when the cache is filled only once, the prompts no tier has used yet are never
            # touched again after the fill pass, which makes them the oldest entries in LRU order; meanwhile the
            # d_output_len output tokens of every request also occupy KV and accumulate across tiers. Once the
            # total approaches the KV capacity those prompts are evicted first, so the tier's first run has to
            # re-prefill while its second run hits fully, and the two repeats diverge (measured on the reference
            # platform: two configurations started at c=10 and c=14; hardware with less memory triggers earlier).
            # This pass re-touches the tier's prompts outside the measurement window (1 output token): prompts
            # still cached cost almost nothing, evicted ones are recomputed and become most-recently-used again.
            # touch_hit < 1 means this tier had suffered eviction.
            nt = min(nfill, max(PROTO["d_n_min"], PROTO["mult"] * c))
            t0h = scrape(murl, HIT)
            tr = bench(a, ds, PROTO["d_touch_conc"], 1, nt, PROTO["d_warmups"], base, f"c{c}_touch", "ttft,e2el")
            t1h = scrape(murl, HIT)
            touch = dict(num_prompts=nt, completed=tr["completed"], hit_ratio=ratio(delta(t0h, t1h, HIT[0]), delta(t0h, t1h, HIT[1])))
            print(f"D c={c} re-touch: {touch['completed']}/{nt} prompts, hit rate of this pass {touch['hit_ratio']}", flush=True)
        for rep in range(1, a.repeat + 1):
            name = f"c{c}_r{rep}"
            if role == "p":
                if c > 8: die(f"the P datasets only cover c=1..8; c={c} has to be generated yourself")
                ds = os.path.join(a.dataset_dir, P_FILE.format(c=c) if rep == 1 else P_FILE_REP.format(c=c, rep=rep)); n = max(PROTO["p_prompts"], 10 * c)
                if not os.path.exists(ds): die(f"missing {ds}; repeated runs need gen-datasets to produce the _r{rep} datasets")
                if n > n_lines(ds): die(f"{ds} has only {n_lines(ds)} lines, fewer than the {n} requests needed")
                s0 = scrape(murl, HIT + PRE); r = bench(a, ds, c, 1, n, warm, base, name, "ttft,e2el"); s1 = scrape(murl, HIT + PRE); acc = None
            else:
                n = min(nfill, max(PROTO["d_n_min"], PROTO["mult"] * c))
                s0 = scrape(murl, HIT + SPEC + PRE); r = bench(a, ds, c, PROTO["d_output_len"], n, warm, base, name, "ttft,tpot,itl,e2el"); s1 = scrape(murl, HIT + SPEC + PRE)
                acc = ratio(delta(s0, s1, SPEC[1]), delta(s0, s1, SPEC[0]))
            r["num_prompts"] = n
            hit = ratio(delta(s0, s1, HIT[0]), delta(s0, s1, HIT[1]))
            wm = re.search(r"num_warmups=(\d+)", open(os.path.join(base, name + ".log"), errors="ignore").read())
            row = tier_row(role, c, r, hit, acc, rep, int(wm.group(1)) if wm else None)
            row["preemptions"] = delta(s0, s1, PRE[0])   # requests the server preempted and recomputed during this run; >0 means KV pressure already affected this tier (C15)
            reps.append(row); raw.append(row)
            print(f"{role.upper()} c={c} #{rep}: {KEY[role][5:-3]} {row[KEY[role]]:.2f} QPM {row['qpm']:.1f} hit {hit}", flush=True)
        agg = aggregate(role, reps); rows.append(agg)
        if touch is not None: agg["touch_hit"] = touch["hit_ratio"]   # <1 means some prompts had been evicted before this tier started and were recomputed
        fails = fails + 1 if agg["verdict"] == "FAIL" else 0
        if a.stop_after_fail and fails >= a.stop_after_fail:
            print(f"{fails} consecutive tiers missed the SLO; stopping as requested by --stop-after-fail (unmeasured tiers do not enter the capacity verdict)"); break
    done = [r["concurrency"] for r in rows]
    ck = make_checks(role, raw, rows, a, m, fill, done, fp, ev)
    for r in raw:
        for k in ("_out_lens", "_errors", "_warmups"): r.pop(k, None)
    bad = [c for c in ck if c["status"] == "FAIL"]
    res = dict(protocol_version=PROTO["protocol_version"], role=role, created_at=time.strftime("%FT%T%z"), base_url=a.base_url,
               manifest_sha256=msha, vendor=dig(m, "vendor") if m else None, run_id=dig(m, "run_id") if m else None,
               declared=({k: dig(m, k) for k in ("hardware.gpu_model", "hardware.gpu_count", "precision.weights", "precision.kv_cache", "precision.rationale",
                                                 "engine.vllm_version", "engine.fork", "engine.deviations", "serve.tp", "serve.mtp_tokens",
                                                 "serve.chunked_prefill", "serve.cudagraph", "serve.other_optimizations")} if m else None),   # snapshot of the vendor's declarations, so an evaluation can list them side by side
               submission_grade=bool(m) and not bad and all(c["status"] in ("PASS", "WARN") for c in ck),
               env={**env_snapshot(), "tokenizer_fp": tfp}, dataset=dict(fingerprint=fp), slo=dict(ttft_ms=PROTO["ttft_slo_ms"], tpot_ms=PROTO["tpot_slo_ms"]),
               repeat=a.repeat, d_output_len=(PROTO["d_output_len"] if role == "d" else None), fill=fill, tiers=rows, capacity=capacity(role, rows), checks=ck,
               accuracy=dict(status="unverified", smoke=None), wall_seconds=round(time.time() - t0))
    json.dump(res, open(os.path.join(base, "result.json"), "w"), indent=1, ensure_ascii=False)
    flat = [{k: v for k, v in r.items() if k != "repeat"} for r in rows]
    with open(os.path.join(base, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys())); w.writeheader(); w.writerows(flat)
    print("\nself-checks:"); [print(f"  {c['id']:4} {c['status']:5} {c['detail']}") for c in ck]
    print("capacity:", json.dumps(res["capacity"], ensure_ascii=False)); print("submission_grade:", res["submission_grade"]); print("result:", os.path.join(base, "result.json"))
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen-datasets"); g.add_argument("--tokenizer", required=True); g.add_argument("--out-dir", required=True); g.add_argument("--force", action="store_true")
    v = sp.add_parser("validate-manifest"); v.add_argument("manifest")
    t = sp.add_parser("selftest-client", help="drive a local stub with the real client command line and check the request-body hashes"); t.add_argument("--model-path", required=True); t.add_argument("--update-golden", action="store_true")
    r = sp.add_parser("run"); r.add_argument("--role", choices=["p", "d"], required=True); r.add_argument("--base-url", required=True)
    r.add_argument("--metrics-url"); r.add_argument("--model-path", required=True); r.add_argument("--served-model-name", required=True)
    r.add_argument("--dataset-dir", required=True); r.add_argument("--out", required=True); r.add_argument("--tiers", help=f"defaults — P: '{DEFAULT_TIERS['p']}' D: '{DEFAULT_TIERS['d']}'")
    r.add_argument("--repeat", type=int, default=2, choices=range(1, MAX_REP + 1), help="independent runs per tier (the protocol requires >=2)"); r.add_argument("--manifest"); r.add_argument("--server-log")
    r.add_argument("--stop-after-fail", type=int, default=0, help="stop after N consecutive tiers miss the SLO (0 = never stop)")
    r.add_argument("--keep-texts", action="store_true", help="keep generated_texts (stripped by default)")
    r.add_argument("--continue-on-preflight-fail", action="store_true", help="continue even if the pre-flight checks fail (debugging only; submission_grade is then always false)")
    a = ap.parse_args(); {"gen-datasets": cmd_gen, "validate-manifest": cmd_validate_manifest, "selftest-client": cmd_selftest, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
