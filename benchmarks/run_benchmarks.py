#!/usr/bin/env python
"""
SEAMS Benchmark Suite

Standalone benchmark runner that does NOT require GUI or display.
Generates synthetic 2048x2048 float32 test images and measures
performance of all core processing operations.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --compare benchmarks/results_old.json benchmarks/results_new.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 42
N_RUNS = 5
SIZE = 2048
TARGETS = {
    "edge_blend_numba": 12.0,
    "edge_blend_rust": 5.0,
    "splat_numba": 180.0,
    "splat_rust": 100.0,
    "gradients_numba": 80.0,
    "gradients_rust": 10.0,
    "inpaint_cpu": 150.0,
    "pbr_sequential": 300.0,
    "pbr_parallel": 120.0,
    "cache_hit": 1.0,
    "cache_miss": 50.0,
    "full_pipeline": 500.0,
}


def _make_test_image(size: int = SIZE) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    return rng.uniform(0, 255, (size, size, 3)).astype(np.float32)


def _run_benchmark(name: str, func, *args, **kwargs) -> dict:
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        try:
            func(*args, **kwargs)
        except Exception as exc:
            print(f"  {name}: FAILED ({exc})")
            return {"name": name, "min_ms": -1, "mean_ms": -1, "target": TARGETS.get(name, 999), "pass": False}
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)

    min_ms = min(times)
    mean_ms = sum(times) / len(times)
    target = TARGETS.get(name, 999)
    passed = min_ms <= target
    return {"name": name, "min_ms": round(min_ms, 1), "mean_ms": round(mean_ms, 1), "target": target, "pass": passed}


def main():
    print("=" * 72)
    print(f" SEAMS Benchmark Suite  |  Image: {SIZE}x{SIZE}  |  Runs: {N_RUNS}")
    print("=" * 72)
    print()

    img = _make_test_image()
    results = []

    # Edge blend (NumPy vectorized)
    def bench_edge_blend_numba():
        from app.core.edge_blending import blend_seams
        offset = np.roll(np.roll(img, SIZE // 2, axis=0), SIZE // 2, axis=1)
        blend_seams(offset, blend_strength=0.5, smoothness=0.5, symmetric=True)

    results.append(_run_benchmark("edge_blend_numba", bench_edge_blend_numba))

    # Edge blend (JIT)
    def bench_edge_blend_jit():
        from app.core.edge_blending_jit import blend_seams_fast
        offset = np.roll(np.roll(img, SIZE // 2, axis=0), SIZE // 2, axis=1)
        blend_seams_fast(offset, blend_strength=0.5, smoothness=0.5)

    results.append(_run_benchmark("edge_blend_numba_jit", bench_edge_blend_jit))

    # Edge blend (Rust)
    try:
        from seams_core import edge_blend_symmetric
        def bench_edge_blend_rust():
            offset = np.roll(np.roll(img, SIZE // 2, axis=0), SIZE // 2, axis=1)
            edge_blend_symmetric(offset, 128, True)
        results.append(_run_benchmark("edge_blend_rust", bench_edge_blend_rust))
    except ImportError:
        print("  [SKIP] Rust edge_blend not available")

    # Gradients (Numba)
    def bench_gradients_numba():
        from app.core.normal_generator import compute_gradients_jit
        gray = img[:, :, 0] / 255.0
        compute_gradients_jit(gray.astype(np.float32), 2.5)

    results.append(_run_benchmark("gradients_numba", bench_gradients_numba))

    # Gradients (Rust)
    try:
        from seams_core import compute_gradients as rs_compute_gradients
        def bench_gradients_rust():
            gray = img[:, :, 0] / 255.0
            rs_compute_gradients(gray.astype(np.float32), 2.5)
        results.append(_run_benchmark("gradients_rust", bench_gradients_rust))
    except ImportError:
        print("  [SKIP] Rust gradients not available")

    # PBR parallel
    def bench_pbr_parallel():
        from app.core.normal_generator import NormalGenerator
        NormalGenerator.process(img, use_cache=False, normal_intensity=0.5, rough_intensity=0.5, ao_intensity=0.5, height_depth=0.5)

    results.append(_run_benchmark("pbr_parallel", bench_pbr_parallel))

    # Cache hit
    def bench_cache_hit():
        from app.core.cache import ResultCache
        cache = ResultCache(max_size=10)
        arr = np.zeros((64, 64, 3), dtype=np.float32)
        cache.set({"k": 1}, arr, "h1")
        for _ in range(1000):
            cache.get({"k": 1}, "h1")

    results.append(_run_benchmark("cache_hit", bench_cache_hit))

    # Full pipeline (smaller image for reasonable time)
    def bench_full_pipeline():
        from app.core.seamless import SeamlessProcessor
        small_img = _make_test_image(512)
        proc = SeamlessProcessor()
        proc.load_image(small_img)
        proc.process(preview=False, chunked=False)

    results.append(_run_benchmark("full_pipeline", bench_full_pipeline))

    # Print results table
    print(f"{'Benchmark':<30} | {'Min ms':>8} | {'Mean ms':>8} | {'Target ms':>9} | {'PASS':>4}")
    print("-" * 30 + "-+-" + "-" * 8 + "-+-" + "-" * 8 + "-+-" + "-" * 9 + "-+-" + "-" * 4)

    all_pass = True
    for r in results:
        status = "YES" if r["pass"] else "NO"
        if not r["pass"]:
            all_pass = False
        min_str = f"{r['min_ms']:.1f}" if r["min_ms"] >= 0 else "FAIL"
        mean_str = f"{r['mean_ms']:.1f}" if r["mean_ms"] >= 0 else "FAIL"
        print(f"{r['name']:<30} | {min_str:>8} | {mean_str:>8} | {r['target']:>9.1f} | {status:>4}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).parent / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({"timestamp": timestamp, "size": SIZE, "runs": N_RUNS, "results": results}, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Compare mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"), help="Compare two result JSON files")
    args, _ = parser.parse_known_args()

    if args.compare:
        _compare_results(args.compare[0], args.compare[1])

    sys.exit(0 if all_pass else 1)


def _compare_results(old_path: str, new_path: str):
    with open(old_path) as f:
        old = json.load(f)
    with open(new_path) as f:
        new = json.load(f)

    old_map = {r["name"]: r for r in old.get("results", [])}
    new_map = {r["name"]: r for r in new.get("results", [])}

    print("\n" + "=" * 72)
    print(" COMPARISON")
    print("=" * 72)
    print(f"{'Benchmark':<30} | {'Old ms':>8} | {'New ms':>8} | {'Delta':>8} | {'Verdict':>8}")
    print("-" * 30 + "-+-" + "-" * 8 + "-+-" + "-" * 8 + "-+-" + "-" * 8 + "-+-" + "-" * 8)

    for name in sorted(set(old_map) | set(new_map)):
        o = old_map.get(name, {}).get("min_ms", 0)
        n = new_map.get(name, {}).get("min_ms", 0)
        delta = n - o if o > 0 and n > 0 else 0
        pct = (delta / o * 100) if o > 0 else 0
        verdict = "FASTER" if delta < -1 else "SLOWER" if delta > 1 else "SAME"
        print(f"{name:<30} | {o:>8.1f} | {n:>8.1f} | {delta:>+8.1f} | {verdict:>8}")


if __name__ == "__main__":
    main()
