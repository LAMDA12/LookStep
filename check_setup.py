#!/usr/bin/env python3
"""Preflight checks for LookStep data construction, training, and simulation."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


LOOKSTEP_ROOT = Path(__file__).resolve().parent
PROJECT_PARENT = LOOKSTEP_ROOT.parent
DEFAULT_DATA_ROOT = Path(os.environ.get("LOOKSTEP_DATA_ROOT", PROJECT_PARENT / "data"))

TRAINING_VERSIONS = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "5.6.2",
    "ms-swift": "4.1.3",
    "accelerate": "1.13.0",
    "deepspeed": "0.18.9",
    "peft": "0.19.1",
    "qwen-vl-utils": "0.0.14",
}

SIMULATION_VERSIONS = {
    "torch": "2.5.1+cu124",
    "torchvision": "0.20.1+cu124",
    "transformers": "4.57.6",
    "accelerate": "1.10.1",
    "habitat-lab": "0.2.4",
    "habitat-sim": "0.2.4",
    "numpy": "1.26.4",
    "Pillow": "9.5.0",
    "tqdm": "4.67.1",
    "hydra-core": "1.3.2",
    "omegaconf": "2.3.0",
    "qwen-vl-utils": "0.0.14",
}


@dataclass
class Check:
    level: str
    name: str
    detail: str


def existing_file(path: Path, name: str) -> Check:
    if path.is_file():
        return Check("PASS", name, str(path))
    return Check("FAIL", name, f"missing: {path}")


def existing_dir(path: Path, name: str) -> Check:
    if path.is_dir():
        return Check("PASS", name, str(path))
    return Check("FAIL", name, f"missing: {path}")


def package_checks(expected: dict, strict_versions: bool) -> Iterable[Check]:
    for distribution, expected_version in expected.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            yield Check("FAIL", distribution, "not installed")
            continue
        if actual == expected_version:
            yield Check("PASS", distribution, actual)
        else:
            level = "FAIL" if strict_versions else "WARN"
            yield Check(
                level,
                distribution,
                f"installed={actual}, reproduced environment={expected_version}",
            )


def python_version_check(stage: str, strict_versions: bool) -> Check:
    actual = sys.version.split()[0]
    expected = {
        "data": "3.11.15",
        "training": "3.11.15",
        "simulation": "3.9.23",
    }[stage]
    if actual == expected:
        return Check("PASS", "Python", actual)
    if not strict_versions and actual.startswith(expected.rsplit(".", 1)[0] + "."):
        return Check(
            "WARN", "Python", f"installed={actual}, reproduced environment={expected}"
        )
    level = "FAIL" if strict_versions else "WARN"
    return Check(
        level, "Python", f"installed={actual}, reproduced environment={expected}"
    )


def model_checks(
    model_path: Optional[str], processor_path: Optional[str]
) -> Iterable[Check]:
    if not model_path:
        yield Check(
            "WARN",
            "model",
            "not supplied; pass --model-path /path/to/downloaded/checkpoint",
        )
        return

    model = Path(model_path).expanduser()
    if not model.exists():
        yield Check("FAIL", "model", f"path does not exist: {model}")
        return

    config_path = model / "config.json"
    yield existing_file(config_path, "model config")
    if config_path.is_file():
        try:
            model_type = json.loads(config_path.read_text(encoding="utf-8")).get(
                "model_type"
            )
            level = "PASS" if model_type == "qwen3_vl" else "WARN"
            yield Check(level, "model architecture", str(model_type or "unspecified"))
        except (OSError, json.JSONDecodeError) as exc:
            yield Check("FAIL", "model architecture", f"invalid config.json: {exc}")

    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))[
                "weight_map"
            ]
            shard_names = sorted(set(weight_map.values()))
            missing_shards = [
                name
                for name in shard_names
                if not (model / name).is_file() or (model / name).stat().st_size == 0
            ]
            if missing_shards:
                yield Check(
                    "FAIL",
                    "model weights",
                    f"missing {len(missing_shards)} shard(s): {missing_shards[:3]}",
                )
            else:
                yield Check(
                    "PASS",
                    "model weights",
                    f"all {len(shard_names)} indexed shard(s) present",
                )
        except (KeyError, OSError, json.JSONDecodeError) as exc:
            yield Check("FAIL", "model weights", f"invalid weight index: {exc}")
    else:
        weight_files = [
            path
            for path in list(model.glob("*.safetensors")) + list(model.glob("*.bin"))
            if path.stat().st_size > 0
        ]
        if weight_files:
            yield Check(
                "PASS", "model weights", f"{len(weight_files)} file(s) in {model}"
            )
        else:
            yield Check(
                "FAIL", "model weights", f"no .safetensors or .bin files in {model}"
            )

    processor = Path(processor_path or model_path).expanduser()
    processor_files = [
        processor / "preprocessor_config.json",
        processor / "tokenizer_config.json",
    ]
    if all(path.is_file() for path in processor_files):
        try:
            from transformers import AutoProcessor

            AutoProcessor.from_pretrained(
                str(processor), trust_remote_code=True, local_files_only=True
            )
            yield Check("PASS", "processor", f"loads successfully from {processor}")
        except Exception as exc:
            yield Check(
                "FAIL",
                "processor",
                f"cannot load from {processor}: {type(exc).__name__}: {exc}. "
                "Use the pinned Qwen3-VL base-model processor or publish compatible processor files.",
            )
    else:
        yield Check(
            "WARN",
            "processor",
            f"processor files incomplete in {processor}; set --processor-path to the base model",
        )


def gpu_check() -> Check:
    if importlib.util.find_spec("torch") is None:
        return Check("FAIL", "CUDA", "torch is not installed")
    try:
        import torch

        if torch.cuda.is_available():
            names = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            return Check("PASS", "CUDA", f"{len(names)} GPU(s): {', '.join(names)}")
        return Check("WARN", "CUDA", "no CUDA GPU visible")
    except Exception as exc:
        return Check("FAIL", "CUDA", str(exc))


def data_checks(data_root: Path) -> Iterable[Check]:
    yield existing_file(
        data_root / "datasets/r2r/train/train.json.gz", "R2R train annotations"
    )
    yield existing_file(
        data_root / "datasets/rxr/train/train_guide.json.gz",
        "RxR train annotations",
    )
    yield existing_dir(
        data_root / "trajectory_data/R2R-CE-640x480/train",
        "R2R expert RGB trajectories",
    )
    yield existing_dir(
        data_root / "trajectory_data/RxR-CE-640x480/train",
        "RxR expert RGB trajectories",
    )


def training_data_checks(training_data_root: Path) -> Iterable[Check]:
    required = (
        "r2r_event_fifo_ms_swift/train.jsonl",
        "r2r_event_fifo_ms_swift/val.jsonl",
        "rxr_event_fifo_ms_swift/train.jsonl",
    )
    for relative in required:
        yield existing_file(training_data_root / relative, f"training data {relative}")

    expected_stats = {
        "r2r_event_fifo_ms_swift": {
            "num_episodes": 10819,
            "train_samples": 619266,
            "val_samples": 11978,
        },
        "rxr_event_fifo_ms_swift": {
            "num_episodes": 19705,
            "train_samples": 1798376,
            "val_samples": 0,
        },
    }
    for dataset_dir, expected in expected_stats.items():
        stats_path = training_data_root / dataset_dir / "stats.json"
        if not stats_path.is_file():
            yield Check(
                "FAIL", f"training stats {dataset_dir}", f"missing: {stats_path}"
            )
            continue
        try:
            actual = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            yield Check("FAIL", f"training stats {dataset_dir}", f"invalid JSON: {exc}")
            continue
        mismatches = [
            f"{key}={actual.get(key)!r} (expected {value})"
            for key, value in expected.items()
            if actual.get(key) != value
        ]
        if mismatches:
            yield Check("FAIL", f"training stats {dataset_dir}", "; ".join(mismatches))
        else:
            summary = ", ".join(f"{key}={value}" for key, value in expected.items())
            yield Check("PASS", f"training stats {dataset_dir}", summary)


def simulation_data_checks(data_root: Path, benchmark: str) -> Iterable[Check]:
    if benchmark in {"r2r", "both"}:
        yield existing_file(
            data_root / "datasets/r2r/val_unseen/val_unseen.json.gz",
            "R2R val-unseen episodes",
        )
    if benchmark in {"rxr", "both"}:
        yield existing_file(
            data_root / "datasets/rxr/val_unseen/val_unseen_guide_en.json.gz",
            "RxR val-unseen English episodes",
        )
    scene_root = data_root / "scene_datasets/mp3d"
    if not scene_root.is_dir():
        yield Check("FAIL", "Matterport3D scenes", f"missing: {scene_root}")
    else:
        scene_count = sum(1 for _ in scene_root.rglob("*.glb"))
        level = "PASS" if scene_count >= 90 else "WARN"
        yield Check(
            level, "Matterport3D scenes", f"{scene_count} .glb file(s) in {scene_root}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["data", "training", "simulation"],
        required=True,
        help="Run this command once inside the environment for the selected stage.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--training-data-root",
        default=str(LOOKSTEP_ROOT / "model_training/data"),
    )
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--processor-path", default=os.environ.get("PROCESSOR_PATH"))
    parser.add_argument("--benchmark", choices=["r2r", "rxr", "both"], default="both")
    parser.add_argument(
        "--strict-versions",
        action="store_true",
        help="Treat package-version drift as an error instead of a warning.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    checks: List[Check] = [
        python_version_check(args.stage, args.strict_versions),
        existing_dir(data_root, "data root"),
    ]

    if args.stage == "data":
        checks.extend(data_checks(data_root))
        checks.extend(package_checks({"tqdm": "4.67.1"}, args.strict_versions))
    elif args.stage == "training":
        checks.extend(package_checks(TRAINING_VERSIONS, args.strict_versions))
        checks.extend(
            training_data_checks(Path(args.training_data_root).expanduser().resolve())
        )
        checks.append(
            Check("PASS", "swift command", shutil.which("swift"))
            if shutil.which("swift")
            else Check("FAIL", "swift command", "not found on PATH")
        )
        checks.append(gpu_check())
    else:
        checks.extend(package_checks(SIMULATION_VERSIONS, args.strict_versions))
        for module in ("habitat", "habitat_sim"):
            checks.append(
                Check("PASS", f"import {module}", "available")
                if importlib.util.find_spec(module)
                else Check("FAIL", f"import {module}", "module not found")
            )
        checks.extend(simulation_data_checks(data_root, args.benchmark))
        checks.extend(model_checks(args.model_path, args.processor_path))
        checks.append(gpu_check())

    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"[{check.level:4}] {check.name:<{width}}  {check.detail}")
    failures = sum(check.level == "FAIL" for check in checks)
    warnings = sum(check.level == "WARN" for check in checks)
    print(
        f"\nSummary: {len(checks) - failures - warnings} pass, {warnings} warning, {failures} failure"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
