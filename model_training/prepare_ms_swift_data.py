import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set

from tqdm import tqdm


LOOKSTEP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LOOKSTEP_ROOT.parent
DEFAULT_INPUT = LOOKSTEP_ROOT / "data_construction/outputs/r2r_event_fifo_full.jsonl"
DEFAULT_OUTPUT_DIR = LOOKSTEP_ROOT / "model_training/data/r2r_event_fifo_ms_swift"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert EventFIFO VLN JSONL into ms-swift multimodal SFT JSONL."
    )
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--image-root",
        default=os.environ.get("LOOKSTEP_IMAGE_ROOT", str(WORKSPACE_ROOT)),
        help="Base directory used to resolve relative image paths in source JSONL.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Episode-level validation ratio. Set 0 to disable validation split unless --val-episodes > 0.",
    )
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=-1,
        help="Use exactly this many validation episodes if positive.",
    )
    parser.add_argument(
        "--no-val-split",
        action="store_true",
        help="Put all samples into train.jsonl and write an empty val.jsonl.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional cap for quick debugging. Applied before writing.",
    )
    parser.add_argument(
        "--path-mode",
        choices=["absolute", "relative"],
        default="absolute",
        help="ms-swift recommends absolute image paths for multimodal data.",
    )
    parser.add_argument(
        "--keep-extra",
        action="store_true",
        help="Keep id/dataset/episode_id/metadata fields in the output JSONL.",
    )
    parser.add_argument(
        "--drop-missing-images",
        action="store_true",
        help="Skip samples with missing image files instead of raising an error.",
    )
    parser.add_argument(
        "--allow-image-tag-mismatch",
        action="store_true",
        help="Do not fail when the prompt <image> count differs from len(images).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.val_ratio <= 1:
        raise ValueError("--val-ratio must be in [0, 1]")
    if args.val_episodes < -1:
        raise ValueError("--val-episodes must be -1 or >= 0")
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max-samples must be -1 or a positive integer")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_image_path(path: str, image_root: Path) -> Path:
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = image_root / image_path
    return image_path.resolve()


def normalize_image_path(path: str, path_mode: str, image_root: Path) -> str:
    image_path = resolve_image_path(path, image_root)
    if path_mode == "absolute":
        return str(image_path)
    try:
        return str(image_path.relative_to(image_root))
    except ValueError:
        return str(image_path)


def collect_episode_ids(input_path: Path, max_samples: int) -> List[str]:
    episode_ids: List[str] = []
    seen: Set[str] = set()
    sample_count = 0
    for sample in tqdm(
        iter_jsonl(input_path),
        desc="Scanning episodes",
        unit="sample",
        dynamic_ncols=True,
    ):
        if max_samples > 0 and sample_count >= max_samples:
            break
        sample_count += 1
        episode_id = str(sample.get("episode_id", ""))
        if episode_id not in seen:
            seen.add(episode_id)
            episode_ids.append(episode_id)
    return episode_ids


def choose_val_episodes(
    episode_ids: List[str],
    val_ratio: float,
    val_episodes: int,
    seed: int,
    no_val_split: bool,
) -> Set[str]:
    if no_val_split:
        return set()

    rng = random.Random(seed)
    shuffled = list(episode_ids)
    rng.shuffle(shuffled)

    if val_episodes >= 0:
        val_count = min(val_episodes, len(shuffled))
    elif val_ratio <= 0:
        val_count = 0
    else:
        val_count = max(1, int(round(len(shuffled) * val_ratio))) if shuffled else 0
    return set(shuffled[:val_count])


def convert_sample(
    sample: Dict,
    path_mode: str,
    image_root: Path,
    keep_extra: bool,
    allow_image_tag_mismatch: bool,
) -> Dict:
    conversations = sample.get("conversations", [])
    if len(conversations) < 2:
        raise ValueError(f"Sample {sample.get('id')} has no two-turn conversations.")

    user_content = conversations[0]["value"]
    assistant_content = conversations[1]["value"]
    absolute_images = [
        resolve_image_path(path, image_root) for path in sample.get("images", [])
    ]

    missing = [str(path) for path in absolute_images if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Sample {sample.get('id')} has missing images: {missing[:3]}"
        )

    images = [
        normalize_image_path(str(path), path_mode, image_root)
        for path in absolute_images
    ]

    image_tag_count = user_content.count("<image>")
    if image_tag_count != len(images) and not allow_image_tag_mismatch:
        raise ValueError(
            f"Sample {sample.get('id')} has {image_tag_count} <image> tags but {len(images)} images."
        )

    converted = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": images,
    }
    if keep_extra:
        converted.update(
            {
                "id": sample.get("id"),
                "dataset": sample.get("dataset"),
                "episode_id": sample.get("episode_id"),
                "step_index": sample.get("step_index"),
                "target_action": sample.get("target_action"),
                "short_labels": sample.get("short_labels"),
                "metadata": sample.get("metadata"),
            }
        )
    return converted


def write_splits(args) -> Dict:
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_ids = collect_episode_ids(input_path, args.max_samples)
    val_episode_ids = choose_val_episodes(
        episode_ids=episode_ids,
        val_ratio=args.val_ratio,
        val_episodes=args.val_episodes,
        seed=args.seed,
        no_val_split=args.no_val_split,
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    sample_path = output_dir / "sample.json"
    skipped_path = output_dir / "skipped.jsonl"

    stats = {
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "image_root": str(image_root),
        "path_mode": args.path_mode,
        "val_ratio": args.val_ratio,
        "val_episodes_arg": args.val_episodes,
        "no_val_split": args.no_val_split,
        "num_episodes": len(episode_ids),
        "num_val_episodes": len(val_episode_ids),
        "num_train_episodes": len(episode_ids) - len(val_episode_ids),
        "train_samples": 0,
        "val_samples": 0,
        "skipped_samples": 0,
        "image_count": Counter(),
        "target_actions": Counter(),
        "memory_write": Counter(),
        "memory_roles": Counter(),
    }

    sample_count = 0
    first_record = None
    with (
        train_path.open("w", encoding="utf-8") as train_f,
        val_path.open("w", encoding="utf-8") as val_f,
        skipped_path.open("w", encoding="utf-8") as skipped_f,
    ):
        for sample in tqdm(
            iter_jsonl(input_path),
            desc="Writing ms-swift JSONL",
            unit="sample",
            dynamic_ncols=True,
        ):
            if args.max_samples > 0 and sample_count >= args.max_samples:
                break
            sample_count += 1
            try:
                converted = convert_sample(
                    sample=sample,
                    path_mode=args.path_mode,
                    image_root=image_root,
                    keep_extra=args.keep_extra,
                    allow_image_tag_mismatch=args.allow_image_tag_mismatch,
                )
            except FileNotFoundError as exc:
                if not args.drop_missing_images:
                    raise
                stats["skipped_samples"] += 1
                skipped_f.write(
                    json.dumps(
                        {"id": sample.get("id"), "error": str(exc)}, ensure_ascii=False
                    )
                    + "\n"
                )
                continue

            episode_id = str(sample.get("episode_id", ""))
            target_f = val_f if episode_id in val_episode_ids else train_f
            split = "val" if episode_id in val_episode_ids else "train"
            target_f.write(json.dumps(converted, ensure_ascii=False) + "\n")

            stats[f"{split}_samples"] += 1
            stats["image_count"][len(converted["images"])] += 1
            stats["target_actions"][sample.get("target_action", "")] += 1
            labels = sample.get("short_labels", {})
            stats["memory_write"][labels.get("memory_write", "")] += 1
            stats["memory_roles"][labels.get("memory_role", "")] += 1
            if first_record is None:
                first_record = converted

    if first_record is not None:
        sample_path.write_text(
            json.dumps(first_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    finalized = dict(stats)
    for key in ["image_count", "target_actions", "memory_write", "memory_roles"]:
        finalized[key] = dict(finalized[key])
    finalized["train_jsonl"] = str(train_path)
    finalized["val_jsonl"] = str(val_path)
    finalized["sample_json"] = str(sample_path)
    finalized["skipped_jsonl"] = str(skipped_path)
    return finalized


def main():
    args = parse_args()
    validate_args(args)
    stats = write_splits(args)
    stats_path = Path(args.output_dir) / "stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote ms-swift data to {args.output_dir}")


if __name__ == "__main__":
    main()
