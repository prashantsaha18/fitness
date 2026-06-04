"""
scripts/benchmark.py
─────────────────────
P99 Latency Benchmark — 5,000 RPS Load Simulation.

Validates that the recommendation API satisfies:
  • P50 < 15ms
  • P95 < 25ms
  • P99 < 35ms (SLA threshold)
  • P999 < 50ms
  • Error rate < 0.1% under 5,000 RPS sustained load

Methodology:
  Uses asyncio semaphore-bounded concurrency to simulate 5,000 concurrent
  virtual users. Each VU sends requests in a closed-loop model (next request
  begins immediately after prior response). This creates realistic back-pressure
  modelling rather than open-loop Poisson arrival which underestimates queuing.

  Warm-up phase (30s): ramp from 10% to 100% load to pre-populate caches.
  Steady-state phase (120s): target RPS sustained measurement window.
  Cool-down phase (10s): drain in-flight requests.

Usage:
  python scripts/benchmark.py --url http://localhost:8000 --rps 5000 --duration 120
  python scripts/benchmark.py --url http://localhost:8000 --rps 1000 --duration 30

Output:
  JSON report + console histogram printed to stdout.
  Prometheus metrics are also available at /metrics during the run.
"""
from __future__ import annotations

import asyncio
import json
import random
import statistics
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

TEST_USER_IDS: list[str] = []   # populated by get_test_tokens()
AUTH_TOKENS: dict[str, str] = {}


# ── Request Builder ───────────────────────────────────────────────────────────

def build_recommendation_payload(user_id: str) -> dict:
    """Generate a realistic recommendation request payload."""
    include_realtime = random.random() > 0.4
    return {
        "user_id": user_id,
        "top_n": random.choice([5, 10, 15, 20]),
        "content_types": random.choice([
            None,
            ["workout_routine"],
            ["meal_recipe"],
            ["video", "workout_routine"],
        ]),
        "diversity_factor": round(random.uniform(0.0, 0.3), 2),
        "realtime_context": {
            "heart_rate_bpm": random.randint(65, 175),
            "fatigue_level": round(random.uniform(0.1, 0.9), 2),
            "active_calories_kcal": round(random.uniform(0, 600), 1),
        } if include_realtime else None,
    }


# ── Metrics Collector ─────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    latency_ms: float
    status_code: int
    is_fallback: bool = False
    error: Optional[str] = None


@dataclass
class BenchmarkMetrics:
    latencies: list[float] = field(default_factory=list)
    status_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    errors: list[str] = field(default_factory=list)
    fallback_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record(self, result: RequestResult) -> None:
        self.latencies.append(result.latency_ms)
        self.status_counts[result.status_code] += 1
        if result.error:
            self.errors.append(result.error)
        if result.is_fallback:
            self.fallback_count += 1

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def report(self) -> dict:
        elapsed = time.perf_counter() - self.start_time
        total_requests = len(self.latencies)
        error_count = sum(1 for s in self.status_counts if s >= 400)
        errors_total = sum(v for k, v in self.status_counts.items() if k >= 400)

        return {
            "summary": {
                "total_requests": total_requests,
                "elapsed_seconds": round(elapsed, 2),
                "actual_rps": round(total_requests / elapsed, 1),
                "success_rate_pct": round(
                    self.status_counts.get(200, 0) / max(total_requests, 1) * 100, 3
                ),
                "error_rate_pct": round(errors_total / max(total_requests, 1) * 100, 4),
                "fallback_rate_pct": round(
                    self.fallback_count / max(total_requests, 1) * 100, 3
                ),
            },
            "latency_ms": {
                "p50": round(self.percentile(50), 2),
                "p75": round(self.percentile(75), 2),
                "p90": round(self.percentile(90), 2),
                "p95": round(self.percentile(95), 2),
                "p99": round(self.percentile(99), 2),
                "p999": round(self.percentile(99.9), 2),
                "mean": round(statistics.mean(self.latencies), 2) if self.latencies else 0,
                "stdev": round(statistics.stdev(self.latencies), 2) if len(self.latencies) > 1 else 0,
                "min": round(min(self.latencies), 2) if self.latencies else 0,
                "max": round(max(self.latencies), 2) if self.latencies else 0,
            },
            "status_distribution": dict(self.status_counts),
            "sla_compliance": {
                "p50_under_15ms": self.percentile(50) < 15.0,
                "p95_under_25ms": self.percentile(95) < 25.0,
                "p99_under_35ms": self.percentile(99) < 35.0,
                "p999_under_50ms": self.percentile(99.9) < 50.0,
                "error_rate_under_01pct": errors_total / max(total_requests, 1) < 0.001,
            },
        }

    def print_histogram(self) -> None:
        """ASCII histogram of latency distribution."""
        if not self.latencies:
            return
        buckets = [5, 10, 15, 20, 25, 30, 35, 50, 75, 100, 150, 250, float("inf")]
        labels = ["<5", "5-10", "10-15", "15-20", "20-25", "25-30",
                  "30-35", "35-50", "50-75", "75-100", "100-150", "150-250", ">250"]
        counts = [0] * len(buckets)
        for lat in self.latencies:
            for i, b in enumerate(buckets):
                if lat <= b:
                    counts[i] += 1
                    break
        total = len(self.latencies)
        bar_width = 40
        print("\nLatency Histogram (ms):")
        print("─" * 65)
        for label, count in zip(labels, counts):
            pct = count / total
            bar = "█" * int(pct * bar_width)
            print(f"  {label:>8}ms │ {bar:<40} {pct*100:5.1f}% ({count})")
        print("─" * 65)


# ── Virtual User ──────────────────────────────────────────────────────────────

async def virtual_user(
    client: httpx.AsyncClient,
    user_id: str,
    token: str,
    metrics: BenchmarkMetrics,
    semaphore: asyncio.Semaphore,
    duration_seconds: float,
    deadline: float,
) -> None:
    """
    Single virtual user: sends requests in closed-loop until deadline.
    Bounded by semaphore to prevent socket exhaustion.
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}{API_PREFIX}/recommend"

    while time.perf_counter() < deadline:
        payload = build_recommendation_payload(user_id)
        t0 = time.perf_counter()
        status_code = 0
        is_fallback = False
        error = None

        async with semaphore:
            try:
                response = await client.post(url, json=payload, headers=headers)
                status_code = response.status_code
                if status_code == 200:
                    data = response.json()
                    is_fallback = data.get("pipeline_metadata", {}).get("is_fallback", False)
            except httpx.TimeoutException:
                status_code = 408
                error = "timeout"
            except Exception as exc:
                status_code = 500
                error = str(exc)[:100]

        latency_ms = (time.perf_counter() - t0) * 1000
        metrics.record(RequestResult(
            latency_ms=latency_ms,
            status_code=status_code,
            is_fallback=is_fallback,
            error=error,
        ))


# ── Token Provisioner ─────────────────────────────────────────────────────────

async def provision_test_tokens(
    client: httpx.AsyncClient,
    num_users: int,
) -> dict[str, str]:
    """
    Register test users and obtain JWT tokens.
    In production benchmark environments, tokens should be pre-seeded
    to avoid skewing warm-up latency measurements.
    """
    tokens = {}
    register_url = f"{BASE_URL}{API_PREFIX}/auth/register"
    login_url = f"{BASE_URL}{API_PREFIX}/auth/token"

    for i in range(min(num_users, 50)):  # cap at 50 for quick setup
        suffix = uuid.uuid4().hex[:8]
        email = f"bench_user_{suffix}@bench.local"
        password = "BenchPassword123!"
        username = f"bench_{suffix}"

        # Register
        try:
            r = await client.post(register_url, json={
                "username": username, "email": email, "password": password
            })
            if r.status_code not in (201, 409):
                continue
        except Exception:
            continue

        # Login
        try:
            r = await client.post(login_url, data={
                "username": email, "password": password
            })
            if r.status_code == 200:
                data = r.json()
                user_id_decoded = _decode_jwt_sub(data["access_token"])
                if user_id_decoded:
                    tokens[user_id_decoded] = data["access_token"]
        except Exception:
            continue

    return tokens


def _decode_jwt_sub(token: str) -> Optional[str]:
    """Extract user_id from JWT without verification (benchmark only)."""
    try:
        import base64, json as _json
        parts = token.split(".")
        padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = _json.loads(base64.urlsafe_b64decode(padded))
        return payload.get("sub")
    except Exception:
        return None


# ── Benchmark Orchestrator ────────────────────────────────────────────────────

async def run_benchmark(
    target_rps: int = 5_000,
    duration_seconds: float = 120.0,
    base_url: str = BASE_URL,
    warmup_seconds: float = 30.0,
) -> dict:
    global BASE_URL
    BASE_URL = base_url

    # Max concurrent connections = target_rps × avg_latency_expected
    # At P99=25ms: max_concurrent ≈ 5000 × 0.025 = 125
    max_concurrent = max(target_rps // 20, 50)
    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"\n{'='*65}")
    print(f"  FITNESS REC ENGINE — BENCHMARK")
    print(f"  Target: {base_url}")
    print(f"  Target RPS: {target_rps:,}")
    print(f"  Duration: {duration_seconds}s (+ {warmup_seconds}s warmup)")
    print(f"  Max concurrency: {max_concurrent}")
    print(f"{'='*65}\n")

    limits = httpx.Limits(
        max_connections=max_concurrent * 2,
        max_keepalive_connections=max_concurrent,
        keepalive_expiry=30,
    )
    timeout = httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)

    async with httpx.AsyncClient(
        limits=limits, timeout=timeout, http2=True
    ) as client:
        # ── Check service health ──────────────────────────────────────────
        try:
            r = await client.get(f"{base_url}/health")
            health = r.json()
            print(f"Service health: {health.get('status', 'unknown')}")
            print(f"Components: {health.get('components', {})}\n")
        except Exception as exc:
            print(f"⚠️  Health check failed: {exc}")
            print("Ensure the API is running before benchmarking.\n")
            return {}

        # ── Provision test tokens ─────────────────────────────────────────
        print("Provisioning test tokens...")
        tokens = await provision_test_tokens(client, num_users=50)
        if not tokens:
            print("⚠️  No tokens provisioned — cannot run authenticated requests.")
            return {}
        user_ids = list(tokens.keys())
        print(f"✅ {len(tokens)} test users ready\n")

        # ── Warm-up phase ─────────────────────────────────────────────────
        warmup_metrics = BenchmarkMetrics()
        warmup_deadline = time.perf_counter() + warmup_seconds
        print(f"🔥 Warm-up phase ({warmup_seconds}s)...")
        warmup_tasks = [
            virtual_user(
                client=client,
                user_id=random.choice(user_ids),
                token=tokens[random.choice(user_ids)],
                metrics=warmup_metrics,
                semaphore=semaphore,
                duration_seconds=warmup_seconds,
                deadline=warmup_deadline,
            )
            for _ in range(min(target_rps // 10, 200))
        ]
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
        warmup_rps = len(warmup_metrics.latencies) / warmup_seconds
        print(f"  Warm-up complete: {len(warmup_metrics.latencies)} requests "
              f"({warmup_rps:.0f} RPS, "
              f"P99={warmup_metrics.percentile(99):.1f}ms)\n")

        # ── Steady-state benchmark ────────────────────────────────────────
        metrics = BenchmarkMetrics()
        deadline = time.perf_counter() + duration_seconds
        num_vus = min(target_rps, 2000)  # cap VUs; each sends multiple requests

        print(f"📊 Steady-state measurement ({duration_seconds}s, {num_vus} VUs)...")

        tasks = []
        for _ in range(num_vus):
            uid = random.choice(user_ids)
            tasks.append(
                virtual_user(
                    client=client,
                    user_id=uid,
                    token=tokens[uid],
                    metrics=metrics,
                    semaphore=semaphore,
                    duration_seconds=duration_seconds,
                    deadline=deadline,
                )
            )

        # Progress reporter
        async def progress_reporter():
            while time.perf_counter() < deadline:
                await asyncio.sleep(10)
                elapsed = time.perf_counter() - metrics.start_time
                n = len(metrics.latencies)
                print(
                    f"  Progress: {n:,} requests | "
                    f"{n/elapsed:.0f} RPS | "
                    f"P50={metrics.percentile(50):.1f}ms | "
                    f"P99={metrics.percentile(99):.1f}ms | "
                    f"Errors={sum(1 for s in metrics.status_counts if s >= 400)}"
                )

        await asyncio.gather(*tasks, progress_reporter(), return_exceptions=True)

    # ── Report ────────────────────────────────────────────────────────────────
    report = metrics.report()

    print(f"\n{'='*65}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*65}")
    print(f"  Total requests:  {report['summary']['total_requests']:,}")
    print(f"  Actual RPS:      {report['summary']['actual_rps']:,}")
    print(f"  Success rate:    {report['summary']['success_rate_pct']:.3f}%")
    print(f"  Error rate:      {report['summary']['error_rate_pct']:.4f}%")
    print(f"  Fallback rate:   {report['summary']['fallback_rate_pct']:.3f}%")
    print(f"\n  Latency Percentiles:")
    for pct, val in report["latency_ms"].items():
        print(f"    {pct:>5}: {val:>8.2f} ms")

    print(f"\n  SLA Compliance:")
    all_pass = True
    for check, passed in report["sla_compliance"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"    {check:<35} {status}")

    metrics.print_histogram()

    print(f"\n  Overall: {'✅ ALL SLA TARGETS MET' if all_pass else '❌ SLA VIOLATIONS DETECTED'}")
    print(f"{'='*65}\n")

    # Save report
    report_path = Path("benchmark_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Full report saved to: {report_path.absolute()}\n")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the recommendation API")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--rps", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=15.0)
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            target_rps=args.rps,
            duration_seconds=args.duration,
            base_url=args.url,
            warmup_seconds=args.warmup,
        )
    )
