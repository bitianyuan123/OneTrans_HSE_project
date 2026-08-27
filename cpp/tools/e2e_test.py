#!/usr/bin/env python3
"""端到端 HTTP 测试：启动 onetrans_server → ingest → score → 与 golden 对拍。

覆盖（对应设计文档 §4 接入层 + §7.4 SEDA 编排 + 混合计算后端）：
1. GET  /healthz                 存活探测（断言生效计算后端）
2. POST /ingest                  Stage I：cases/ingest_case.json → accepted + checksum
3. POST /score                   Stage II：cases/score_case.json → logits 对拍 golden（<1e-5）
4. POST /score（未 ingest 用户） KV miss 降级：全零 logits + kv_hit=false（G8 语义）
5. 并发 32 路异步 score           SEDA 流水线 + 异步接入（全部 200 + hit 行对拍）
6. GET  /metrics                 指标暴露（flow.backend.* 计数）
7. 404/405 路由行为

运行：python cpp/tools/e2e_test.py [--server PATH] [--weights DIR] [--golden DIR]
                                     [--backend auto|python|cpp]
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
from concurrent.futures import ThreadPoolExecutor
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


def wait_ready(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            code, _ = http("GET", f"http://127.0.0.1:{port}/healthz")
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
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
    ap.add_argument("--backend", choices=["auto", "python", "cpp"], default="auto")
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
            "--compute-backend", args.backend,
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

        # 1. healthz（含生效计算后端断言）
        code, body = http("GET", f"{base}/healthz")
        backend = json.loads(body).get("compute_backend", "?")
        expect_backend = "cpp" if args.backend == "cpp" else "python"
        check(
            "healthz",
            code == 200
            and json.loads(body)["status"] == "ok"
            and backend == expect_backend,
            f"compute_backend={backend} (want {expect_backend})",
        )

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

        # 5. 并发 32 路：异步接入 + SEDA 流水线（hit/miss 混合；hit 行与 golden 对拍）
        def one_score(i: int) -> tuple[int, dict]:
            b = json.loads(score_body)
            if i % 2:
                b["user_id"] = f"user-rand-{i}"  # miss 路径
            c, body2 = http("POST", f"{base}/score", json.dumps(b).encode())
            return c, json.loads(body2)

        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(ex.map(one_score, range(32)))
        all_200 = all(c == 200 for c, _ in results)
        hit_ok = all(
            abs(g - w) <= 1e-5
            for _, j2 in results[::2]  # 偶数下标 = hit 行
            for gr, wr in zip(j2["logits"], want)
            for g, w in zip(gr, wr)
        )
        check(
            "score/concurrent_32",
            all_200 and hit_ok,
            f"32 reqs, all_200={all_200}, hit_rows_match_golden={hit_ok}",
        )

        # 6. metrics（含计算后端路径计数：flow.backend.python / flow.backend.cpp）
        code, body = http("GET", f"{base}/metrics")
        backend_metric = f"flow.backend.{backend}"
        check(
            "metrics",
            code == 200
            and "kv.hit" in body
            and "online.qps" in body
            and backend_metric in body,
            f"backend counter={backend_metric}",
        )

        # 7. 路由行为
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
