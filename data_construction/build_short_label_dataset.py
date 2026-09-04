import argparse
import gzip
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from tqdm import tqdm


LOOKSTEP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LOOKSTEP_ROOT.parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("LOOKSTEP_DATA_ROOT", str(WORKSPACE_ROOT / "data"))
)
ACTION_NAMES = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
TURN_ACTIONS = {"TURN_LEFT", "TURN_RIGHT"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build short-label SFT data with event-driven rolling memory supervision."
    )
    parser.add_argument(
        "--dataset",
        choices=["r2r", "rxr", "all"],
        default="r2r",
        help="Main-experiment dataset source. 'all' selects exactly R2R and RxR.",
    )
    parser.add_argument(
        "--trajectory-root",
        default=str(DEFAULT_DATA_ROOT / "trajectory_data"),
        help="Root directory containing trajectory images.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATA_ROOT / "datasets"),
        help="Root containing r2r/ and rxr/ Habitat annotation directories.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(LOOKSTEP_ROOT / "data_construction/outputs/short_labels.jsonl"),
        help="Output JSONL path. One training sample per line.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path. Defaults to output-jsonl with .summary.json suffix.",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=-1,
        help="Only process the first N episodes per selected dataset. Use for debugging.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Stop after writing this many samples globally.",
    )
    parser.add_argument(
        "--future-window",
        type=int,
        default=5,
        help="How many future GT actions are used to create approaching_* labels.",
    )
    parser.add_argument(
        "--history-policy",
        choices=["event_fifo", "full"],
        default="event_fifo",
        help="event_fifo simulates online rolling memory; full keeps all previous images as a baseline.",
    )
    parser.add_argument(
        "--max-long-memory",
        type=int,
        default=6,
        help="Maximum event-memory frames for event_fifo. Oldest event memory is removed first.",
    )
    parser.add_argument(
        "--recent-size",
        type=int,
        default=2,
        help="Number of recent observations kept outside long event memory.",
    )
    parser.add_argument(
        "--path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Write image paths relative to the directory containing LookStep, or as absolute paths.",
    )
    parser.add_argument(
        "--include-current-in-history",
        action="store_true",
        help="If set, history_images also includes current_frame. Default keeps history before current step only.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit_episodes == 0 or args.limit_episodes < -1:
        raise ValueError("--limit-episodes must be -1 or a positive integer")
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max-samples must be -1 or a positive integer")
    if args.future_window <= 0:
        raise ValueError("--future-window must be > 0")
    if args.max_long_memory < 0 or args.recent_size < 0:
        raise ValueError("memory sizes must be >= 0")


def read_json_or_gzip(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_output_path(path: Path, path_mode: str) -> str:
    path = path.resolve()
    if path_mode == "absolute":
        return str(path)
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def get_instruction_text(ep: dict) -> str:
    instruction = ep.get("instruction", "")
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text", "")).strip()
    if isinstance(instruction, str):
        return instruction.strip()
    instructions = ep.get("instructions", [])
    if instructions:
        return str(instructions[0]).strip()
    return ""


def parse_vlnce_action_from_name(path: Path) -> str:
    stem = path.stem
    match = re.match(r"step_\d+_(.+)", stem)
    if match:
        return match.group(1).upper()
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-2].upper() == "MOVE":
        return "MOVE_FORWARD"
    if len(parts) >= 2:
        return parts[-1].upper()
    return "STOP"


def is_turn_action(action: str) -> bool:
    return action in TURN_ACTIONS


def turn_direction(action: str) -> str:
    if action == "TURN_LEFT":
        return "left"
    if action == "TURN_RIGHT":
        return "right"
    return "none"


def opposite_turn(action: str) -> str:
    if action == "TURN_LEFT":
        return "TURN_RIGHT"
    if action == "TURN_RIGHT":
        return "TURN_LEFT"
    return ""


def segment_bounds(actions: Sequence[str], step_idx: int) -> Dict[str, int]:
    action = actions[step_idx]
    start = step_idx
    while start > 0 and actions[start - 1] == action:
        start -= 1
    end = step_idx
    while end + 1 < len(actions) and actions[end + 1] == action:
        end += 1
    return {"start": start, "end": end, "length": end - start + 1}


def turn_segment_position(actions: Sequence[str], step_idx: int) -> str:
    if not is_turn_action(actions[step_idx]):
        return "none"
    bounds = segment_bounds(actions, step_idx)
    if bounds["length"] == 1:
        return "single_step"
    if step_idx == bounds["start"]:
        return "start"
    if step_idx == bounds["end"]:
        return "finishing"
    return "in_progress"


def first_future_turn(future_actions: Sequence[str]) -> str:
    for action in future_actions:
        if is_turn_action(action):
            return action
    return ""


def is_post_turn_alignment(actions: Sequence[str], step_idx: int) -> bool:
    if step_idx == 0:
        return False
    return actions[step_idx] == "MOVE_FORWARD" and is_turn_action(actions[step_idx - 1])


def is_goal_approach_start(
    actions: Sequence[str], step_idx: int, future_window: int
) -> bool:
    if actions[step_idx] == "STOP":
        return False
    future_actions = actions[step_idx + 1 : step_idx + 1 + future_window]
    if "STOP" not in future_actions:
        return False
    if step_idx == 0:
        return True
    previous_future_actions = actions[step_idx : step_idx + future_window]
    return "STOP" not in previous_future_actions


def progress_label(step_idx: int, total_steps: int, target_action: str) -> str:
    if step_idx == 0:
        return "start"
    if target_action == "STOP" or step_idx >= max(0, total_steps - 2):
        return "near_goal"
    ratio = step_idx / max(1, total_steps - 1)
    if ratio < 0.33:
        return "early"
    if ratio < 0.66:
        return "middle"
    return "late"


def event_label(actions: Sequence[str], step_idx: int, future_window: int) -> str:
    target_action = actions[step_idx]
    future_actions = actions[step_idx + 1 : step_idx + 1 + future_window]
    if target_action == "STOP":
        return "stop_now"
    if is_turn_action(target_action):
        direction = turn_direction(target_action)
        position = turn_segment_position(actions, step_idx)
        return f"turn_{direction}_{position}"
    if is_post_turn_alignment(actions, step_idx):
        return "post_turn_alignment"
    if is_goal_approach_start(actions, step_idx, future_window):
        return "goal_approach"
    if "STOP" in future_actions:
        return "approaching_goal"
    next_turn = first_future_turn(future_actions)
    if next_turn == "TURN_LEFT":
        return "approaching_left_turn"
    if next_turn == "TURN_RIGHT":
        return "approaching_right_turn"
    return "follow_route"


def memory_write_label(
    actions: Sequence[str], step_idx: int, future_window: int
) -> Dict[str, str]:
    target_action = actions[step_idx]
    if target_action == "STOP":
        return {"memory_write": "keep", "memory_role": "stop_evidence"}
    if step_idx == 0:
        return {"memory_write": "keep", "memory_role": "start_view"}
    if is_turn_action(target_action):
        position = turn_segment_position(actions, step_idx)
        if position in {"start", "single_step"}:
            return {"memory_write": "keep", "memory_role": "turn_start"}
        if position == "finishing":
            return {"memory_write": "keep", "memory_role": "turn_end"}
        return {"memory_write": "drop", "memory_role": "recent_only"}
    if is_post_turn_alignment(actions, step_idx):
        return {"memory_write": "keep", "memory_role": "post_turn_alignment"}
    if is_goal_approach_start(actions, step_idx, future_window):
        return {"memory_write": "keep", "memory_role": "goal_approach"}
    return {"memory_write": "drop", "memory_role": "recent_only"}


def action_outcomes(
    actions: Sequence[str], step_idx: int, future_window: int
) -> Dict[str, str]:
    target_action = actions[step_idx]
    future_actions = actions[step_idx + 1 : step_idx + 1 + future_window]
    if target_action == "STOP":
        return {
            "MOVE_FORWARD": "overshoot_goal",
            "TURN_LEFT": "wrong_at_goal",
            "TURN_RIGHT": "wrong_at_goal",
            "STOP": "success_stop",
        }

    outcomes: Dict[str, str] = {}
    next_turn = first_future_turn(future_actions)
    for candidate in ACTION_NAMES:
        if candidate == target_action:
            if candidate == "MOVE_FORWARD":
                if is_post_turn_alignment(actions, step_idx):
                    outcomes[candidate] = "advance_after_turn"
                elif "STOP" in future_actions:
                    outcomes[candidate] = "advance_to_goal"
                else:
                    outcomes[candidate] = "advance"
            elif candidate == "TURN_LEFT":
                position = turn_segment_position(actions, step_idx)
                if position == "finishing":
                    outcomes[candidate] = "finish_turn"
                elif position == "in_progress":
                    outcomes[candidate] = "continue_turn"
                else:
                    outcomes[candidate] = "start_turn"
            elif candidate == "TURN_RIGHT":
                position = turn_segment_position(actions, step_idx)
                if position == "finishing":
                    outcomes[candidate] = "finish_turn"
                elif position == "in_progress":
                    outcomes[candidate] = "continue_turn"
                else:
                    outcomes[candidate] = "start_turn"
            else:
                outcomes[candidate] = "success_stop"
            continue

        if candidate == "STOP":
            outcomes[candidate] = "too_early"
        elif is_turn_action(target_action):
            outcomes[candidate] = (
                "premature_forward" if candidate == "MOVE_FORWARD" else "wrong_turn"
            )
        elif is_post_turn_alignment(actions, step_idx) and is_turn_action(candidate):
            if candidate == actions[step_idx - 1]:
                outcomes[candidate] = "over_turn"
            elif candidate == opposite_turn(actions[step_idx - 1]):
                outcomes[candidate] = "reverse_turn"
            else:
                outcomes[candidate] = "wrong_turn"
        elif candidate == "TURN_LEFT" and next_turn == "TURN_LEFT":
            outcomes[candidate] = "early_left_turn"
        elif candidate == "TURN_RIGHT" and next_turn == "TURN_RIGHT":
            outcomes[candidate] = "early_right_turn"
        else:
            outcomes[candidate] = "wrong_turn"
    return outcomes


def step_short_labels(
    actions: Sequence[str], step_idx: int, future_window: int
) -> Dict:
    target_action = actions[step_idx]
    memory_label = memory_write_label(actions, step_idx, future_window)
    return {
        "progress": progress_label(step_idx, len(actions), target_action),
        "event": event_label(actions, step_idx, future_window),
        "memory_write": memory_label["memory_write"],
        "memory_role": memory_label["memory_role"],
        "outcomes": action_outcomes(actions, step_idx, future_window),
    }


def build_assistant_output(labels: Dict, target_action: str) -> str:
    outcomes = labels["outcomes"]
    return (
        f"<progress>{labels['progress']}</progress>\n"
        f"<event>{labels['event']}</event>\n"
        f"<memory_write>{labels['memory_write']}</memory_write>\n"
        f"<memory_role>{labels['memory_role']}</memory_role>\n"
        "<outcomes>\n"
        f"<move_forward>{outcomes['MOVE_FORWARD']}</move_forward>\n"
        f"<turn_left>{outcomes['TURN_LEFT']}</turn_left>\n"
        f"<turn_right>{outcomes['TURN_RIGHT']}</turn_right>\n"
        f"<stop>{outcomes['STOP']}</stop>\n"
        "</outcomes>\n"
        f"<action>{target_action}</action>"
    )


def build_full_prompt(instruction: str, history_count: int) -> str:
    history_tags = "<image>" * history_count
    return (
        "You are an end-to-end vision-language navigation model. "
        "Given the instruction, all historical observations, and the current observation, "
        "predict compact navigation labels and the next action.\n"
        f"<instruction>{instruction}</instruction>\n"
        f"<history>{history_tags}</history>\n"
        "<current_observation><image></current_observation>\n"
        "Candidate actions: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP.\n"
        "Output exactly these tags: <progress>, <event>, <memory_write>, <memory_role>, <outcomes>, and <action>."
    )


def build_event_fifo_prompt(
    instruction: str, long_memory: Sequence[Dict], recent_count: int
) -> str:
    long_memory_tags = "".join(
        f"<image><memory_role>{entry['memory_role']}</memory_role>"
        for entry in long_memory
    )
    recent_tags = "<image>" * recent_count
    return (
        "You are an end-to-end vision-language navigation model with an online event memory. "
        "Use long-term event memory, recent observations, and the current observation to predict compact labels and the next action.\n"
        f"<instruction>{instruction}</instruction>\n"
        f"<long_memory>{long_memory_tags}</long_memory>\n"
        f"<recent_observations>{recent_tags}</recent_observations>\n"
        "<current_observation><image></current_observation>\n"
        "Candidate actions: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP.\n"
        "Output exactly these tags: <progress>, <event>, <memory_write>, <memory_role>, <outcomes>, and <action>."
    )


def normalize_memory_entry(entry: Dict, path_mode: str) -> Dict:
    return {
        "image": normalize_output_path(entry["path"], path_mode),
        "step_index": entry["step_index"],
        "action": entry["action"],
        "memory_role": entry["memory_role"],
    }


def visible_recent_entries(
    long_memory: Sequence[Dict], recent_buffer: Sequence[Dict]
) -> List[Dict]:
    long_paths = {entry["path"] for entry in long_memory}
    return [entry for entry in recent_buffer if entry["path"] not in long_paths]


def make_full_history_sample(
    dataset: str,
    episode_id: str,
    instruction: str,
    image_files: Sequence[Path],
    target_actions: Sequence[str],
    step_idx: int,
    future_window: int,
    path_mode: str,
    include_current_in_history: bool,
) -> Dict:
    total_steps = len(image_files)
    target_action = target_actions[step_idx]
    future_actions = list(target_actions[step_idx + 1 : step_idx + 1 + future_window])
    labels = step_short_labels(target_actions, step_idx, future_window)

    history_end = step_idx + 1 if include_current_in_history else step_idx
    history_images = [
        normalize_output_path(path, path_mode) for path in image_files[:history_end]
    ]
    current_frame = normalize_output_path(image_files[step_idx], path_mode)
    # The prompt always contains one current-observation image tag after history.
    # If include_current_in_history is enabled, the current image is intentionally
    # duplicated so the image list still matches the prompt tags.
    images = history_images + [current_frame]
    assistant_output = build_assistant_output(labels, target_action)

    return {
        "id": f"{dataset}_{episode_id}_step_{step_idx:03d}",
        "dataset": dataset,
        "episode_id": str(episode_id),
        "step_index": step_idx,
        "num_steps": total_steps,
        "instruction": instruction,
        "history_images": history_images,
        "current_frame": current_frame,
        "images": images,
        "target_action": target_action,
        "short_labels": labels,
        "metadata": {
            "future_window": future_window,
            "future_actions": future_actions,
            "history_policy": (
                "full_history_including_current"
                if include_current_in_history
                else "full_history_before_current"
            ),
            "long_memory_length": 0,
            "recent_length": len(history_images),
        },
        "conversations": [
            {
                "from": "human",
                "value": build_full_prompt(instruction, len(history_images)),
            },
            {
                "from": "gpt",
                "value": assistant_output,
            },
        ],
    }


def make_event_fifo_sample(
    dataset: str,
    episode_id: str,
    instruction: str,
    image_files: Sequence[Path],
    target_actions: Sequence[str],
    step_idx: int,
    future_window: int,
    path_mode: str,
    long_memory: Sequence[Dict],
    recent_buffer: Sequence[Dict],
    max_long_memory: int,
    recent_size: int,
) -> Dict:
    total_steps = len(image_files)
    target_action = target_actions[step_idx]
    future_actions = list(target_actions[step_idx + 1 : step_idx + 1 + future_window])
    labels = step_short_labels(target_actions, step_idx, future_window)

    visible_recent = visible_recent_entries(long_memory, recent_buffer)
    normalized_long_memory = [
        normalize_memory_entry(entry, path_mode) for entry in long_memory
    ]
    normalized_recent = [
        normalize_memory_entry(entry, path_mode) for entry in visible_recent
    ]
    long_memory_images = [entry["image"] for entry in normalized_long_memory]
    recent_images = [entry["image"] for entry in normalized_recent]
    history_images = long_memory_images + recent_images
    current_frame = normalize_output_path(image_files[step_idx], path_mode)
    images = history_images + [current_frame]
    assistant_output = build_assistant_output(labels, target_action)

    return {
        "id": f"{dataset}_{episode_id}_step_{step_idx:03d}",
        "dataset": dataset,
        "episode_id": str(episode_id),
        "step_index": step_idx,
        "num_steps": total_steps,
        "instruction": instruction,
        "long_memory": normalized_long_memory,
        "recent_observations": normalized_recent,
        "history_images": history_images,
        "current_frame": current_frame,
        "images": images,
        "target_action": target_action,
        "short_labels": labels,
        "metadata": {
            "future_window": future_window,
            "future_actions": future_actions,
            "history_policy": "event_fifo",
            "max_long_memory": max_long_memory,
            "recent_size": recent_size,
            "long_memory_length": len(normalized_long_memory),
            "recent_length": len(normalized_recent),
        },
        "conversations": [
            {
                "from": "human",
                "value": build_event_fifo_prompt(
                    instruction, normalized_long_memory, len(normalized_recent)
                ),
            },
            {
                "from": "gpt",
                "value": assistant_output,
            },
        ],
    }


def vlnce_episode_source(
    dataset: str,
    trajectory_root: Path,
    dataset_root: Path,
    limit_episodes: int,
) -> Dict:
    if dataset == "r2r":
        episode_json = dataset_root / "r2r/train/train.json.gz"
        image_root = trajectory_root / "R2R-CE-640x480/train"
    elif dataset == "rxr":
        episode_json = dataset_root / "rxr/train/train_guide.json.gz"
        image_root = trajectory_root / "RxR-CE-640x480/train"
    else:
        raise ValueError(f"Unsupported VLN-CE dataset: {dataset}")

    data = read_json_or_gzip(episode_json)
    episodes = data["episodes"]
    if limit_episodes > 0:
        episodes = episodes[:limit_episodes]

    def iterator():
        for ep in episodes:
            episode_id = str(ep["episode_id"])
            instruction = get_instruction_text(ep)
            image_dir = image_root / episode_id
            image_files = sorted(image_dir.glob("*.png"))
            if not image_files:
                image_files = sorted(image_dir.glob("*.jpg"))
            if not image_files:
                continue
            target_actions = [
                parse_vlnce_action_from_name(path) for path in image_files
            ]
            yield {
                "dataset": dataset,
                "episode_id": episode_id,
                "instruction": instruction,
                "image_files": image_files,
                "target_actions": target_actions,
            }

    return {"name": dataset, "iterator": iterator(), "total": len(episodes)}


def selected_episode_sources(
    dataset: str,
    trajectory_root: Path,
    dataset_root: Path,
    limit_episodes: int,
) -> List[Dict]:
    if dataset in {"r2r", "rxr"}:
        return [
            vlnce_episode_source(dataset, trajectory_root, dataset_root, limit_episodes)
        ]
    if dataset == "all":
        return [
            vlnce_episode_source("r2r", trajectory_root, dataset_root, limit_episodes),
            vlnce_episode_source("rxr", trajectory_root, dataset_root, limit_episodes),
        ]
    raise ValueError(f"Unsupported main-experiment dataset: {dataset}")


def make_memory_entry(
    image_files: Sequence[Path],
    target_actions: Sequence[str],
    step_idx: int,
    memory_role: str,
) -> Dict:
    return {
        "path": image_files[step_idx],
        "step_index": step_idx,
        "action": target_actions[step_idx],
        "memory_role": memory_role,
    }


def update_stats(stats: Dict, sample: Dict) -> None:
    labels = sample["short_labels"]
    metadata = sample["metadata"]
    stats["samples"] += 1
    stats["actions"][sample["target_action"]] += 1
    stats["progress"][labels["progress"]] += 1
    stats["events"][labels["event"]] += 1
    stats["memory_write"][labels["memory_write"]] += 1
    stats["memory_roles"][labels["memory_role"]] += 1
    stats["input_history_lengths"][len(sample["history_images"])] += 1
    stats["input_long_memory_lengths"][metadata.get("long_memory_length", 0)] += 1
    stats["input_recent_lengths"][metadata.get("recent_length", 0)] += 1
    stats["total_history_images"] += len(sample["history_images"])
    stats["total_long_memory_images"] += metadata.get("long_memory_length", 0)
    stats["total_recent_images"] += metadata.get("recent_length", 0)


def fifo_append(memory: List[Dict], entry: Dict, max_size: int) -> None:
    memory.append(entry)
    while len(memory) > max_size:
        memory.pop(0)


def write_samples(args) -> Dict:
    trajectory_root = Path(args.trajectory_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset_arg": args.dataset,
        "trajectory_root": str(trajectory_root),
        "dataset_root": str(dataset_root),
        "output_jsonl": str(output_path),
        "path_mode": args.path_mode,
        "future_window": args.future_window,
        "history_policy": args.history_policy,
        "max_long_memory": args.max_long_memory,
        "recent_size": args.recent_size,
        "include_current_in_history": args.include_current_in_history,
        "episodes": 0,
        "samples": 0,
        "skipped_episodes": 0,
        "datasets": Counter(),
        "actions": Counter(),
        "progress": Counter(),
        "events": Counter(),
        "memory_write": Counter(),
        "memory_roles": Counter(),
        "input_history_lengths": Counter(),
        "input_long_memory_lengths": Counter(),
        "input_recent_lengths": Counter(),
        "total_history_images": 0,
        "total_long_memory_images": 0,
        "total_recent_images": 0,
    }

    episode_sources = selected_episode_sources(
        args.dataset,
        trajectory_root,
        dataset_root,
        args.limit_episodes,
    )
    with output_path.open("w", encoding="utf-8") as f:
        for source in episode_sources:
            progress = tqdm(
                source["iterator"],
                total=source["total"],
                desc=f"Building {source['name']}",
                unit="episode",
                dynamic_ncols=True,
            )
            for ep in progress:
                image_files = ep["image_files"]
                target_actions = ep["target_actions"]
                if len(image_files) != len(target_actions):
                    stats["skipped_episodes"] += 1
                    progress.set_postfix(
                        samples=stats["samples"], skipped=stats["skipped_episodes"]
                    )
                    continue
                invalid_actions = sorted(set(target_actions) - set(ACTION_NAMES))
                if invalid_actions:
                    raise ValueError(
                        f"Episode {ep['dataset']}/{ep['episode_id']} contains "
                        f"unsupported actions: {invalid_actions}"
                    )
                stats["episodes"] += 1
                stats["datasets"][ep["dataset"]] += 1
                long_memory: List[Dict] = []
                recent_buffer: List[Dict] = []

                for step_idx in range(len(image_files)):
                    if args.max_samples > 0 and stats["samples"] >= args.max_samples:
                        return finalize_stats(stats)
                    if args.history_policy == "event_fifo":
                        sample = make_event_fifo_sample(
                            dataset=ep["dataset"],
                            episode_id=ep["episode_id"],
                            instruction=ep["instruction"],
                            image_files=image_files,
                            target_actions=target_actions,
                            step_idx=step_idx,
                            future_window=args.future_window,
                            path_mode=args.path_mode,
                            long_memory=long_memory,
                            recent_buffer=recent_buffer,
                            max_long_memory=args.max_long_memory,
                            recent_size=args.recent_size,
                        )
                    else:
                        sample = make_full_history_sample(
                            dataset=ep["dataset"],
                            episode_id=ep["episode_id"],
                            instruction=ep["instruction"],
                            image_files=image_files,
                            target_actions=target_actions,
                            step_idx=step_idx,
                            future_window=args.future_window,
                            path_mode=args.path_mode,
                            include_current_in_history=args.include_current_in_history,
                        )
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    update_stats(stats, sample)

                    if args.history_policy == "event_fifo":
                        labels = sample["short_labels"]
                        if labels["memory_write"] == "keep":
                            fifo_append(
                                long_memory,
                                make_memory_entry(
                                    image_files,
                                    target_actions,
                                    step_idx,
                                    labels["memory_role"],
                                ),
                                args.max_long_memory,
                            )
                        fifo_append(
                            recent_buffer,
                            make_memory_entry(
                                image_files, target_actions, step_idx, "recent_only"
                            ),
                            args.recent_size,
                        )
                progress.set_postfix(
                    samples=stats["samples"], skipped=stats["skipped_episodes"]
                )
    return finalize_stats(stats)


def finalize_stats(stats: Dict) -> Dict:
    finalized = dict(stats)
    for key in [
        "datasets",
        "actions",
        "progress",
        "events",
        "memory_write",
        "memory_roles",
        "input_history_lengths",
        "input_long_memory_lengths",
        "input_recent_lengths",
    ]:
        finalized[key] = dict(finalized[key])
    if finalized["samples"] > 0:
        finalized["avg_history_images"] = (
            finalized["total_history_images"] / finalized["samples"]
        )
        finalized["avg_long_memory_images"] = (
            finalized["total_long_memory_images"] / finalized["samples"]
        )
        finalized["avg_recent_images"] = (
            finalized["total_recent_images"] / finalized["samples"]
        )
    return finalized


def main():
    args = parse_args()
    validate_args(args)
    summary = write_samples(args)
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else Path(args.output_jsonl).with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
