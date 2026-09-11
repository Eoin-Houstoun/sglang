"""End-to-end CPU serving benchmark for Artemis: launch the model through this checkout's
SGLang server, gate on GSM8K, measure the 1k-8k in / 1k out workload, write
artemis_results.json (numeric only) to the working directory. Diagnostics go to stderr.

Configuration is read from environment variables (see run_e2e.sh and config.env).
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def env(name, default):
    return os.environ.get(name, default)


MODEL = env("SGLANG_E2E_MODEL", "zai-org/GLM-5.2-FP8")
TP = int(env("SGLANG_E2E_TP", "6"))
PORT = int(env("SGLANG_E2E_PORT", "30100"))
SERVER_ARGS = shlex.split(env("SGLANG_E2E_SERVER_ARGS", "--trust-remote-code --disable-overlap-schedule"))
LOAD_TIMEOUT = int(env("SGLANG_E2E_LOAD_TIMEOUT", "7200"))
GSM8K_N = int(env("SGLANG_E2E_GSM8K_N", "200"))
GSM8K_MIN_ACC = float(env("SGLANG_E2E_GSM8K_MIN_ACC", "0.85"))
GSM8K_PARALLEL = int(env("SGLANG_E2E_GSM8K_PARALLEL", "16"))
GSM8K_MAX_NEW = int(env("SGLANG_E2E_GSM8K_MAX_NEW_TOKENS", "512"))
INPUT_LENS = [int(x) for x in env("SGLANG_E2E_INPUT_LENS", "1024 8192").split()]
OUTPUT_LEN = int(env("SGLANG_E2E_OUTPUT_LEN", "1024"))
NUM_PROMPTS = int(env("SGLANG_E2E_NUM_PROMPTS", "4"))
CONCURRENCY = int(env("SGLANG_E2E_CONCURRENCY", "4"))
BASE_URL = f"http://127.0.0.1:{PORT}"


def log(msg):
    print(f"[e2e] {msg}", file=sys.stderr, flush=True)


def write_results(results):
    with open(Path.cwd() / "artemis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(json.dumps(results))


def launch_server():
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL, "--device", "cpu", "--tp", str(TP), "--host", "127.0.0.1", "--port", str(PORT), *SERVER_ARGS]
    log("launching: " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.Popen(cmd, stdout=sys.stderr, stderr=subprocess.STDOUT, start_new_session=True)


def wait_healthy(proc):
    t0 = time.time()
    while time.time() - t0 < LOAD_TIMEOUT:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before becoming healthy")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"server not healthy after {LOAD_TIMEOUT} s")


def stop_server(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)


def gsm8k_accuracy():
    from sglang.test.few_shot_gsm8k import run_eval

    args = SimpleNamespace(num_shots=5, data_path=None, num_questions=GSM8K_N, max_new_tokens=GSM8K_MAX_NEW, parallel=GSM8K_PARALLEL, host="http://127.0.0.1", port=PORT, temperature=0.0)
    metrics = run_eval(args)
    log(f"gsm8k: accuracy {metrics['accuracy']:.3f} invalid {metrics['invalid']:.3f} over {GSM8K_N} questions")
    return float(metrics["accuracy"])


def serving_point(input_len):
    out = Path.cwd() / f"bench_serving_{input_len}.jsonl"
    out.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang", "--base-url", BASE_URL, "--dataset-name", "random", "--random-input-len", str(input_len), "--random-output-len", str(OUTPUT_LEN), "--random-range-ratio", "1.0", "--num-prompts", str(NUM_PROMPTS), "--max-concurrency", str(CONCURRENCY), "--request-rate", "inf", "--disable-tqdm", "--output-file", str(out)]
    log(f"bench_serving: {input_len} in, {OUTPUT_LEN} out, {NUM_PROMPTS} prompts, concurrency {CONCURRENCY}")
    subprocess.run(cmd, check=True, stdout=sys.stderr, stderr=subprocess.STDOUT)
    rec = json.loads(out.read_text().splitlines()[-1])
    out.unlink(missing_ok=True)
    if rec["completed"] != NUM_PROMPTS:
        raise RuntimeError(f"bench_serving at {input_len}: {rec['completed']} of {NUM_PROMPTS} requests completed")
    k = f"{input_len // 1024}k"
    return {
        f"ttft_ms_{k}_in": rec["median_ttft_ms"],
        f"tpot_ms_{k}_in": rec["median_tpot_ms"],
        f"e2e_s_{k}_in": rec["median_e2e_latency_ms"] / 1000.0,
        f"output_tps_{k}_in": rec["output_throughput"],
    }


def main():
    results_path = Path.cwd() / "artemis_results.json"
    results_path.unlink(missing_ok=True)
    proc = launch_server()
    try:
        load_s = wait_healthy(proc)
        log(f"server healthy after {load_s:.0f} s")
        results = {"server_load_s": round(load_s, 1)}
        acc = gsm8k_accuracy()
        results["gsm8k_accuracy"] = acc
        results["gsm8k_ok"] = 1 if acc >= GSM8K_MIN_ACC else 0
        if not results["gsm8k_ok"]:
            write_results(results)
            log(f"gsm8k accuracy {acc:.3f} below the gate {GSM8K_MIN_ACC}: failing this version")
            sys.exit(1)
        for n in INPUT_LENS:
            results.update(serving_point(n))
        write_results(results)
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
