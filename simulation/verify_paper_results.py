#!/usr/bin/env python3
"""Compare reproduced Habitat metrics with the values reported in the paper."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


LOOKSTEP_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r2r-metrics",
        default=str(LOOKSTEP_ROOT / "simulation/outputs/r2r_val_unseen/metrics.json"),
    )
    parser.add_argument(
        "--rxr-metrics",
        default=str(LOOKSTEP_ROOT / "simulation/outputs/rxr_val_unseen/metrics.json"),
    )
    parser.add_argument(
        "--percent-tolerance",
        type=float,
        default=1.0,
        help="Allowed absolute difference in percentage points for SR/OS/SPL.",
    )
    parser.add_argument(
        "--ne-tolerance",
        type=float,
        default=0.15,
        help="Allowed absolute navigation-error difference in meters.",
    )
    return parser.parse_args()


def load(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalized_metrics(metrics: Dict) -> Dict[str, float]:
    aliases = {
        "episodes": ("episodes", "episode_count"),
        "navigation_error": ("navigation_error", "ones_all", "ne"),
        "oracle_success": ("oracle_success", "oss_all", "os"),
        "success": ("success", "sucs_all", "sr"),
        "spl": ("spl", "spls_all"),
    }
    result = {}
    for canonical, keys in aliases.items():
        value = next((metrics[key] for key in keys if key in metrics), None)
        if value is None:
            continue
        value = float(value)
        if canonical not in {"episodes", "navigation_error"} and value <= 1.0:
            value *= 100.0
        result[canonical] = value
    return result


def compare(
    benchmark: str,
    actual: Dict[str, float],
    expected: Dict[str, float],
    percent_tolerance: float,
    ne_tolerance: float,
) -> Tuple[int, list]:
    failures = 0
    rows = []
    for name, target in expected.items():
        if name not in actual:
            failures += 1
            rows.append((benchmark, name, "missing", target, "FAIL"))
            continue
        value = actual[name]
        if name == "episodes":
            tolerance = 0.0
        elif name == "navigation_error":
            tolerance = ne_tolerance
        else:
            tolerance = percent_tolerance
        status = "PASS" if abs(value - target) <= tolerance else "FAIL"
        failures += status == "FAIL"
        rows.append((benchmark, name, value, target, status))
    return failures, rows


def compare_recipe(benchmark: str, metrics: Dict) -> Tuple[int, list]:
    expected = {
        "eval_mode": "lookstep_event_fifo",
        "split": "val_unseen",
        "max_long_memory": 6,
        "recent_size": 2,
        "max_steps": 400,
        "limit_episodes": -1,
        "seed": 42,
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
        "torch_dtype": "bfloat16",
    }
    failures = 0
    rows = []
    for name, target in expected.items():
        if name not in metrics:
            status = "FAIL"
            actual = "missing"
        else:
            actual = metrics[name]
            status = "PASS" if actual == target else "FAIL"
        failures += status == "FAIL"
        rows.append((benchmark, name, actual, target, status))
    return failures, rows


def main() -> int:
    args = parse_args()
    targets = {
        "R2R": {
            "episodes": 1839,
            "navigation_error": 5.34,
            "oracle_success": 55.9,
            "success": 49.7,
            "spl": 45.3,
        },
        "RxR": {
            "episodes": 3669,
            "navigation_error": 6.89,
            "success": 46.9,
            "spl": 39.9,
        },
    }
    paths = {
        "R2R": Path(args.r2r_metrics),
        "RxR": Path(args.rxr_metrics),
    }
    rows = []
    recipe_rows = []
    failures = 0
    for benchmark in ("R2R", "RxR"):
        raw_metrics = load(paths[benchmark])
        actual = normalized_metrics(raw_metrics)
        count, benchmark_rows = compare(
            benchmark,
            actual,
            targets[benchmark],
            args.percent_tolerance,
            args.ne_tolerance,
        )
        failures += count
        rows.extend(benchmark_rows)
        count, benchmark_recipe_rows = compare_recipe(benchmark, raw_metrics)
        failures += count
        recipe_rows.extend(benchmark_recipe_rows)

    print(
        f"{'Benchmark':<10} {'Metric':<20} {'Actual':>10} {'Paper':>10} {'Status':>8}"
    )
    for benchmark, metric, actual, expected, status in rows:
        actual_text = actual if isinstance(actual, str) else f"{actual:.3f}"
        print(
            f"{benchmark:<10} {metric:<20} {actual_text:>10} {expected:>10.3f} {status:>8}"
        )
    print("\nRecipe checks:")
    for benchmark, name, actual, expected, status in recipe_rows:
        print(
            f"{benchmark:<10} {name:<20} actual={actual!r} expected={expected!r} {status}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
