#!/usr/bin/env python3
"""端到端 HTTP 测试：启动 onetrans_server → ingest → score → 与 golden 对拍。

覆盖（对应设计文档 §4 接入层 + §3 两阶段编排）：
1. GET  /healthz                 存活探测
2. POST /ingest                  Stage I：cases/ingest_case.json → accepted + checksum
3. POST /score                   Stage II：cases/score_case.json → logits 对拍 golden（<1e-5）
4. POST /score（未 ingest 用户） KV miss 降级：全零 logits + kv_hit=false（G8 语义）
5. GET  /metrics                 指标暴露
6. 404/405 路由行为

运行：python cpp/tools/e2e_test.py [--server PATH] [--weights DIR] [--golden DIR]
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method: str, url: str, body: bytes | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def wait_ready(port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            code, _ = http("GET", f"http://127.0.0.1:{port}/healthz")
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def load_golden_logits(golden_dir: Path) -> list[list[float]]:
    """读 golden.bin 中 logits_two_stage（manifest 声明 shape/offset）。"""
    manifest = json.loads((golden_dir / "manifest.json").read_text(encoding="utf-8"))
    meta = manifest["tensors"]["logits_two_stage"]
    m, t = meta["shape"]
    blob = (golden_dir / "golden.bin").read_bytes()
    vals = struct.unpack_from(f"<{m * t}f", blob, meta["offset"] * 4)
    return [list(vals[i * t : (i + 1) * t]) for i in range(m)]


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--server", type=Path, default=root / "cpp/build/onetrans_server")
    ap.add_argument("--weights", type=Path, default=root / "cpp/artifacts/weights")
    ap.add_argument("--golden", type=Path, default=root / "cpp/artifacts/golden")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    if not args.server.exists():
        print(f"[e2e] server 不存在: {args.server}（先 cmake --build cpp/build）")
        return 2
    port = args.port or free_port()

    proc = subprocess.Popen(
        [
            str(args.server),
            "--weights", str(args.weights),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--model-version", "v42",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures += 1

    try:
        if not wait_ready(port):
            out = proc.stdout.read().decode() if proc.stdout else ""
            print(f"[e2e] server 启动失败:\n{out}")
            return 1
        base = f"http://127.0.0.1:{port}"

        # 1. healthz
        code, body = http("GET", f"{base}/healthz")
        check("healthz", code == 200 and json.loads(body)["status"] == "ok")

        # 2. ingest（Stage I）
        ingest_body = (args.golden / "cases/ingest_case.json").read_bytes()
        code, body = http("POST", f"{base}/ingest", ingest_body)
        j = json.loads(body)
        check(
            "ingest/accepted",
            code == 200 and j["accepted"] and len(j["checksum"]) == 64,
            f"s_len=37 事件入 KV, checksum={j['checksum'][:16]}…",
        )

        # 3. score（Stage II）与 golden 对拍
        score_body = (args.golden / "cases/score_case.json").read_bytes()
        code, body = http("POST", f"{base}/score", score_body)
        j = json.loads(body)
        want = load_golden_logits(args.golden)
        ok = code == 200 and j["kv_hit"] and len(j["logits"]) == len(want)
        max_diff = 0.0
        if ok:
            for got_row, want_row in zip(j["logits"], want):
                for g, w in zip(got_row, want_row):
                    max_diff = max(max_diff, abs(g - w))
            ok = max_diff < 1e-5
        check("score/logits_vs_golden", ok, f"max|diff|={max_diff:.2e}")

        # 4. KV miss 降级（G8）：未 ingest 用户 → 全零 + kv_hit=false
        miss = json.loads(score_body)
        miss["user_id"] = "user-never-ingested"
        code, body = http("POST", f"{base}/score", json.dumps(miss).encode())
        j = json.loads(body)
        zeros = all(abs(v) == 0.0 for row in j.get("logits", [[1]]) for v in row)
        check("score/kv_miss_degrade", code == 200 and not j["kv_hit"] and zeros)

        # 5. metrics
        code, body = http("GET", f"{base}/metrics")
        check("metrics", code == 200 and "kv.hit" in body and "online.qps" in body)

        # 6. 路由行为
        code, _ = http("GET", f"{base}/nope")
        check("route/404", code == 404)
        code, _ = http("PUT", f"{base}/score")
        check("route/405", code == 405)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n[e2e] {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
