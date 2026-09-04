import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


LOOKSTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LOOKSTEP_ROOT.parent
for import_root in (LOOKSTEP_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

if __package__:  # package import
    from .distributed import (
        get_rank,
        get_world_size,
        init_distributed_mode,
        is_dist_avail_and_initialized,
    )
    from .habitat_evaluator import VLNEvaluator, set_seed
else:  # direct script execution
    from simulation.distributed import (  # type: ignore
        get_rank,
        get_world_size,
        init_distributed_mode,
        is_dist_avail_and_initialized,
    )
    from simulation.habitat_evaluator import VLNEvaluator, set_seed  # type: ignore


VALID_ACTIONS = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
OUTCOME_TAG_TO_ACTION = {
    "move_forward": "MOVE_FORWARD",
    "turn_left": "TURN_LEFT",
    "turn_right": "TURN_RIGHT",
    "stop": "STOP",
}
KEEP_MEMORY_ROLES = {
    "start_view",
    "turn_start",
    "turn_end",
    "post_turn_alignment",
    "goal_approach",
    "stop_evidence",
}
ACTION_PATTERNS = [
    (re.compile(r"\bMOVE[_\s-]*FORWARD\b", re.I), "MOVE_FORWARD"),
    (re.compile(r"\bGO[_\s-]*FORWARD\b", re.I), "MOVE_FORWARD"),
    (re.compile(r"\bFORWARD\b", re.I), "MOVE_FORWARD"),
    (re.compile(r"\bSTRAIGHT\b", re.I), "MOVE_FORWARD"),
    (re.compile(r"\bAHEAD\b", re.I), "MOVE_FORWARD"),
    (re.compile(r"\bTURN[_\s-]*LEFT\b", re.I), "TURN_LEFT"),
    (re.compile(r"\bGO[_\s-]*LEFT\b", re.I), "TURN_LEFT"),
    (re.compile(r"\bLEFT\b", re.I), "TURN_LEFT"),
    (re.compile(r"\bTURN[_\s-]*RIGHT\b", re.I), "TURN_RIGHT"),
    (re.compile(r"\bGO[_\s-]*RIGHT\b", re.I), "TURN_RIGHT"),
    (re.compile(r"\bRIGHT\b", re.I), "TURN_RIGHT"),
    (re.compile(r"\bSTOP\b", re.I), "STOP"),
    (re.compile(r"\bHALT\b", re.I), "STOP"),
    (re.compile(r"\bDONE\b", re.I), "STOP"),
    (re.compile(r"\bEND\b", re.I), "STOP"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Habitat simulator evaluation for Short-label EventFIFO VLN models."
    )
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Optional processor/tokenizer path. Use the base model path when the checkpoint does not include processor files.",
    )
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default=str(LOOKSTEP_ROOT / "simulation/configs/r2r.yaml"),
    )
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("LOOKSTEP_DATA_ROOT", str(REPO_ROOT / "data")),
        help="Directory containing datasets/ and scene_datasets/.",
    )
    parser.add_argument(
        "--scenes_dir",
        type=str,
        default=None,
        help="Optional override for the Matterport3D scene directory.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional Habitat episode JSON path/template; may contain {split}.",
    )
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--save_video_ratio", type=float, default=0.05, help="0~1")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--rank", default=0, type=int, help="rank")
    parser.add_argument("--gpu", default=0, type=int, help="gpu")
    parser.add_argument(
        "--sim_gpu_id",
        default=None,
        type=int,
        help="Habitat-Sim GPU index; defaults to the torchrun local rank.",
    )
    parser.add_argument("--port", default="1111")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--max_steps", default=400, type=int, help="max episode steps")
    parser.add_argument(
        "--limit_episodes",
        default=-1,
        type=int,
        help="Evaluate only the first N episodes globally; -1 evaluates the full split.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--max_long_memory", type=int, default=6)
    parser.add_argument("--recent_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--save_step_traces",
        action="store_true",
        help="Write per-step model outputs for debugging.",
    )
    parser.add_argument(
        "--log_step_time", action="store_true", help="Print per-step inference timing."
    )
    parser.add_argument(
        "--log_step_time_every",
        type=int,
        default=1,
        help="Print timing every N steps when enabled.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be > 0")
    if args.limit_episodes == 0 or args.limit_episodes < -1:
        raise ValueError("--limit_episodes must be -1 or a positive integer")
    if args.max_long_memory < 0 or args.recent_size < 0:
        raise ValueError("memory sizes must be >= 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be > 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if args.num_beams < 1:
        raise ValueError("--num_beams must be >= 1")
    if not 0 <= args.save_video_ratio <= 1:
        raise ValueError("--save_video_ratio must be in [0, 1]")


DEFAULT_QWEN3VL_ROPE_SCALING = {
    "mrope_interleaved": True,
    "mrope_section": [24, 20, 20],
    "rope_type": "default",
}


def _ensure_qwen3vl_rope_scaling(config) -> None:
    """Work around transformers bug: Qwen3VLTextRotaryEmbedding uses rope_scaling.get()
    even when rope_scaling is None (see modeling_qwen3_vl.py ~297).
    """
    model_type = getattr(config, "model_type", None)
    text_cfg = None
    if model_type == "qwen3_vl":
        text_cfg = getattr(config, "text_config", None)
    elif model_type == "qwen3_vl_text":
        text_cfg = config
    if text_cfg is not None and getattr(text_cfg, "rope_scaling", None) is None:
        text_cfg.rope_scaling = dict(DEFAULT_QWEN3VL_ROPE_SCALING)


def normalize_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def _portable_source(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.exists() else value


def ensure_run_manifest(args, eval_mode: str = "lookstep_event_fifo") -> None:
    """Prevent silently resuming an output directory with a different recipe."""
    manifest = {
        "eval_mode": eval_mode,
        "model_path": _portable_source(args.model_path),
        "processor_path": _portable_source(args.processor_path or args.model_path),
        "habitat_config_path": str(
            Path(args.habitat_config_path).expanduser().resolve()
        ),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "scenes_dir": _portable_source(args.scenes_dir),
        "dataset_path": args.dataset_path,
        "eval_split": args.eval_split,
        "world_size": get_world_size(),
        "max_steps": args.max_steps,
        "limit_episodes": args.limit_episodes,
        "seed": args.seed,
        "max_long_memory": args.max_long_memory,
        "recent_size": args.recent_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "torch_dtype": args.torch_dtype,
        "save_video": args.save_video,
        "save_video_ratio": args.save_video_ratio,
    }
    manifest_path = Path(args.output_path) / "run_config.json"
    error = None
    if get_rank() == 0:
        try:
            if manifest_path.is_file():
                with manifest_path.open("r", encoding="utf-8") as file:
                    existing = json.load(file)
                if existing != manifest:
                    error = (
                        f"Output directory has a different run_config.json: {manifest_path}. "
                        "Use a new OUTPUT_PATH for a different model or recipe."
                    )
            else:
                old_outputs = [
                    path
                    for pattern in ("episodes_rank*.jsonl", "metrics.json")
                    for path in manifest_path.parent.glob(pattern)
                    if path.is_file() and path.stat().st_size > 0
                ]
                if old_outputs:
                    error = (
                        f"Output directory contains results but no run_config.json: "
                        f"{manifest_path.parent}. Use a new OUTPUT_PATH so their recipe "
                        "cannot be mixed with this run."
                    )
                else:
                    with manifest_path.open("w", encoding="utf-8") as file:
                        json.dump(
                            manifest,
                            file,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
        except Exception as exc:
            error = f"Could not validate run manifest {manifest_path}: {exc}"
    if is_dist_avail_and_initialized():
        errors = [error]
        torch.distributed.broadcast_object_list(errors, src=0)
        error = errors[0]
    if error:
        raise RuntimeError(error)


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    return text.strip()


def normalize_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value.strip())
    return value if value else None


def normalize_action(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().upper()
    if value in VALID_ACTIONS:
        return value
    for pattern, action in ACTION_PATTERNS:
        if pattern.search(value):
            return action
    return None


def extract_tag(text: str, tag_name: str) -> Optional[str]:
    cleaned = strip_thinking(text)
    match = re.search(
        rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>", cleaned, flags=re.S | re.I
    )
    if not match:
        return None
    return normalize_label(match.group(1))


def text_without_auxiliary_tags(text: str) -> str:
    """Remove structured state fields before attempting a legacy action fallback."""
    cleaned = strip_thinking(text)
    auxiliary_tags = [
        "progress",
        "event",
        "memory_write",
        "memory_role",
        "outcomes",
        *OUTCOME_TAG_TO_ACTION,
    ]
    for tag_name in auxiliary_tags:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{tag_name}>", " ", cleaned, flags=re.S | re.I
        )
    return re.sub(r"<[^>]+>", " ", cleaned)


def parse_action_from_text(text: str) -> Optional[str]:
    action = normalize_action(extract_tag(text, "action"))
    if action:
        return action
    cleaned = text_without_auxiliary_tags(text)
    if not cleaned:
        return None
    for candidate in VALID_ACTIONS:
        if candidate in cleaned.upper():
            return candidate
    for pattern, action in ACTION_PATTERNS:
        if pattern.search(cleaned):
            return action
    return None


class ShortLabelEventFIFOInference:
    def __init__(
        self,
        pretrained: str,
        processor_path: Optional[str],
        device: str,
        args: argparse.Namespace,
    ):
        processor_source = processor_path or pretrained
        model_dtype = normalize_dtype(args.torch_dtype)
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": {"": device},
        }
        if model_dtype != "auto":
            model_kwargs["torch_dtype"] = model_dtype
        model_config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        _ensure_qwen3vl_rope_scaling(model_config)
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained, config=model_config, **model_kwargs
        ).eval()
        if not hasattr(self.model, "past_key_values_vggt"):
            self.model.past_key_values_vggt = None

        self.processor = AutoProcessor.from_pretrained(
            processor_source, trust_remote_code=True, padding_side="left"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            processor_source, trust_remote_code=True, padding_side="left"
        )
        self.device = device

        self.max_long_memory = args.max_long_memory
        self.recent_size = args.recent_size
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams

        self.long_memory: List[Dict] = []
        self.recent_buffer: List[Dict] = []
        self.last_prediction: Optional[Dict] = None

    def reset_episode(self):
        for entry in self.long_memory:
            entry["image"].close()
        for entry in self.recent_buffer:
            entry["image"].close()
        self.long_memory = []
        self.recent_buffer = []
        self.last_prediction = None
        self.model.past_key_values_vggt = None

    def _append_fifo(self, buffer: List[Dict], entry: Dict, max_size: int) -> None:
        buffer.append(entry)
        while len(buffer) > max_size:
            removed = buffer.pop(0)
            removed["image"].close()

    def _visible_recent_entries(self) -> List[Dict]:
        long_steps = {entry["step_id"] for entry in self.long_memory}
        return [
            entry for entry in self.recent_buffer if entry["step_id"] not in long_steps
        ]

    def _build_user_content(
        self, instruction: str, current_image: Image.Image
    ) -> List[Dict]:
        visible_recent = self._visible_recent_entries()
        content: List[Dict] = [
            {
                "type": "text",
                "text": (
                    "You are an end-to-end vision-language navigation model with an online event memory. "
                    "Use long-term event memory, recent observations, and the current observation to predict compact "
                    "labels and the next action.\n"
                    f"<instruction>{instruction}</instruction>\n"
                    "<long_memory>"
                ),
            }
        ]

        for entry in self.long_memory:
            content.append({"type": "image", "image": entry["image"]})
            content.append(
                {
                    "type": "text",
                    "text": f"<memory_role>{entry['memory_role']}</memory_role>",
                }
            )

        content.append(
            {"type": "text", "text": "</long_memory>\n<recent_observations>"}
        )

        for entry in visible_recent:
            content.append({"type": "image", "image": entry["image"]})

        content.append(
            {"type": "text", "text": "</recent_observations>\n<current_observation>"}
        )
        content.append({"type": "image", "image": current_image})
        content.append(
            {
                "type": "text",
                "text": (
                    "</current_observation>\n"
                    "Candidate actions: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP.\n"
                    "Output exactly these tags: <progress>, <event>, <memory_write>, <memory_role>, <outcomes>, and <action>."
                ),
            }
        )
        return content

    def _move_inputs(self, inputs):
        if hasattr(inputs, "to"):
            return inputs.to(self.model.device)
        moved = {}
        for key, value in inputs.items():
            moved[key] = value.to(self.model.device) if hasattr(value, "to") else value
        return moved

    def _sync_cuda(self) -> None:
        if torch.cuda.is_available() and "cuda" in str(self.model.device):
            torch.cuda.synchronize(self.model.device)

    def _parse_output(self, text: str) -> Dict:
        return {
            "progress": extract_tag(text, "progress"),
            "event": extract_tag(text, "event"),
            "memory_write": extract_tag(text, "memory_write"),
            "memory_role": extract_tag(text, "memory_role"),
            "outcomes": {
                action: extract_tag(text, tag)
                for tag, action in OUTCOME_TAG_TO_ACTION.items()
            },
            "action": parse_action_from_text(text),
        }

    def _update_memory(
        self, current_image: Image.Image, step_id: int, parsed: Dict
    ) -> None:
        memory_write = (parsed.get("memory_write") or "").lower()
        memory_role = parsed.get("memory_role") or "recent_only"
        if memory_write == "keep":
            if memory_role not in KEEP_MEMORY_ROLES:
                if parsed.get("action") == "STOP":
                    memory_role = "stop_evidence"
                else:
                    memory_role = "post_turn_alignment"
            self._append_fifo(
                self.long_memory,
                {
                    "image": current_image.copy(),
                    "step_id": step_id,
                    "memory_role": memory_role,
                },
                self.max_long_memory,
            )

        self._append_fifo(
            self.recent_buffer,
            {
                "image": current_image.copy(),
                "step_id": step_id,
                "memory_role": "recent_only",
            },
            self.recent_size,
        )

    def call_model(
        self,
        observations,
        task,
        step_id,
        add_frame_index: bool = False,
        gen_kwargs: Optional[dict] = None,
    ):
        del add_frame_index
        total_start = time.perf_counter()
        gen_kwargs = gen_kwargs or {}
        if isinstance(observations, Image.Image):
            current_image = observations
        elif (
            isinstance(observations, (list, tuple))
            and observations
            and isinstance(observations[-1], Image.Image)
        ):
            current_image = observations[-1]
        else:
            raise TypeError(f"Unsupported observation type: {type(observations)}")

        content = self._build_user_content(task, current_image)
        messages = [{"role": "user", "content": content}]
        try:
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        all_images = (
            [entry["image"] for entry in self.long_memory]
            + [entry["image"] for entry in self._visible_recent_entries()]
            + [current_image]
        )
        inputs = self.processor(
            text=[prompt_text],
            images=all_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)
        # self._sync_cuda()
        generate_start = time.perf_counter()

        generation_args = {
            "max_new_tokens": gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": gen_kwargs.get("temperature", self.temperature),
            "top_p": gen_kwargs.get("top_p", self.top_p),
            "num_beams": gen_kwargs.get("num_beams", self.num_beams),
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=generation_args["temperature"] > 0,
                temperature=(
                    generation_args["temperature"]
                    if generation_args["temperature"] > 0
                    else None
                ),
                top_p=(
                    generation_args["top_p"]
                    if generation_args["temperature"] > 0
                    else None
                ),
                num_beams=generation_args["num_beams"],
                max_new_tokens=generation_args["max_new_tokens"],
            )
        # self._sync_cuda()
        generate_time_sec = time.perf_counter() - generate_start

        prompt_length = inputs["input_ids"].shape[-1]
        outputs = outputs[:, prompt_length:]
        answers = self.processor.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        parsed = self._parse_output(answers[0])
        total_time_sec = time.perf_counter() - total_start
        self.last_prediction = {
            "step_id": step_id,
            "raw_output": answers[0],
            "parsed": parsed,
            "long_memory_size_before_update": len(self.long_memory),
            "recent_visible_size_before_update": len(self._visible_recent_entries()),
            "inference_time_sec": total_time_sec,
            "generate_time_sec": generate_time_sec,
        }
        self._update_memory(current_image, step_id, parsed)
        self.last_prediction["long_memory_size_after_update"] = len(self.long_memory)
        self.last_prediction["recent_visible_size_after_update"] = len(
            self._visible_recent_entries()
        )
        return answers


class ShortLabelVLNEvaluator(VLNEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_trace_path = None
        if self.args.save_step_traces:
            self.step_trace_path = os.path.join(
                self.output_path,
                f"step_traces_rank{get_rank()}.jsonl",
            )

    def _append_step_trace(self, record: Dict) -> None:
        if not self.step_trace_path:
            return
        with open(self.step_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def eval_action(self, idx):
        env = self.config_env()
        episodes = sorted(
            env.episodes,
            key=lambda episode: (str(episode.scene_id), str(episode.episode_id)),
        )
        if self.args.limit_episodes > 0:
            episodes = episodes[: self.args.limit_episodes]
        assigned_episodes = episodes[idx :: self.env_num]

        def episode_key(episode):
            scene_path = Path(str(episode.scene_id))
            scene_id = scene_path.parent.name or scene_path.stem
            instruction = (
                episode.instruction.instruction_text
                if "objectnav" not in self.config_path
                else episode.object_category
            )
            return scene_id, str(episode.episode_id), instruction

        assigned_keys = {episode_key(episode) for episode in assigned_episodes}
        completed = {}
        result_path = os.path.join(self.output_path, f"episodes_rank{get_rank()}.jsonl")
        result_needs_newline = False
        if os.path.exists(result_path):
            if os.path.getsize(result_path) > 0:
                with open(result_path, "rb") as raw_file:
                    raw_file.seek(-1, os.SEEK_END)
                    result_needs_newline = raw_file.read(1) != b"\n"
            with open(result_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        res = json.loads(line)
                    except json.JSONDecodeError:
                        print(
                            f"Ignoring malformed resume record in {result_path}:{line_number}"
                        )
                        continue
                    if {"scene_id", "episode_id", "episode_instruction"}.issubset(
                        res.keys()
                    ):
                        key = (
                            str(res["scene_id"]),
                            str(res["episode_id"]),
                            res["episode_instruction"],
                        )
                        if key in assigned_keys and {
                            "success",
                            "spl",
                            "os",
                            "ne",
                        }.issubset(res.keys()):
                            completed[key] = res

        sucs = [res["success"] for res in completed.values()]
        spls = [res["spl"] for res in completed.values()]
        oss = [res["os"] for res in completed.values()]
        ones = [res["ne"] for res in completed.values()]
        done_res = set(completed)
        process_bar = tqdm(
            assigned_episodes,
            desc=f"Habitat rank {get_rank()}",
            unit="episode",
        )
        try:
            for episode in process_bar:
                scene_id, episode_id, episode_instruction = episode_key(episode)
                print("episode start: ", episode_instruction)
                if (scene_id, episode_id, episode_instruction) in done_res:
                    continue

                env.current_episode = episode
                observations = env.reset()
                self.model.reset_episode()

                vis_frames = []
                step_id = 0
                should_save_video = self.save_video and (
                    random.random() < self.save_video_ratio
                )
                if should_save_video:
                    os.makedirs(
                        os.path.join(self.output_path, f"vis_{self.epoch}"),
                        exist_ok=True,
                    )

                while not env.episode_over:
                    rgb = observations["rgb"]
                    info = env.get_metrics()
                    with Image.fromarray(rgb).convert("RGB") as image:
                        action_text = self.model.call_model(
                            image, episode_instruction, step_id
                        )[0]
                    parsed_action = (
                        self.model.last_prediction["parsed"]["action"]
                        if self.model.last_prediction
                        else None
                    )
                    if (
                        self.args.log_step_time
                        and self.model.last_prediction is not None
                        and step_id % max(1, self.args.log_step_time_every) == 0
                    ):
                        print(
                            f"[step_time] scene={scene_id} episode={episode_id} step={step_id} "
                            f"infer={self.model.last_prediction['inference_time_sec']:.3f}s "
                            f"generate={self.model.last_prediction['generate_time_sec']:.3f}s"
                        )

                    if self.model.last_prediction is not None:
                        self._append_step_trace(
                            {
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "episode_instruction": episode_instruction,
                                "step_id": step_id,
                                "raw_output": self.model.last_prediction["raw_output"],
                                "parsed": self.model.last_prediction["parsed"],
                                "long_memory_size_before_update": self.model.last_prediction[
                                    "long_memory_size_before_update"
                                ],
                                "long_memory_size_after_update": self.model.last_prediction[
                                    "long_memory_size_after_update"
                                ],
                                "recent_visible_size_before_update": self.model.last_prediction[
                                    "recent_visible_size_before_update"
                                ],
                                "recent_visible_size_after_update": self.model.last_prediction[
                                    "recent_visible_size_after_update"
                                ],
                                "inference_time_sec": self.model.last_prediction[
                                    "inference_time_sec"
                                ],
                                "generate_time_sec": self.model.last_prediction[
                                    "generate_time_sec"
                                ],
                            }
                        )

                    if info.get("top_down_map") is not None and should_save_video:
                        from habitat.utils.visualizations.utils import (
                            observations_to_image,
                        )

                        frame = observations_to_image(
                            {"rgb": observations["rgb"]}, info
                        )
                        vis_frames.append(frame)

                    if parsed_action in self.actions2idx:
                        action = self.actions2idx[parsed_action][0]
                    else:
                        print(f"Unparsed action at step {step_id}: {action_text}")
                        action = 0

                    if step_id >= self.args.max_steps:
                        action = 0

                    observations = env.step(action)
                    step_id += 1

                metrics = env.get_metrics()
                if should_save_video:
                    from habitat.utils.visualizations.utils import images_to_video

                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, f"vis_{self.epoch}"),
                        f"{scene_id}_{episode_id}",
                        fps=6,
                        quality=9,
                    )
                vis_frames.clear()
                sucs.append(metrics["success"])
                spls.append(metrics["spl"])
                oss.append(metrics["oracle_success"])
                ones.append(metrics["distance_to_goal"])
                print(
                    f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, "
                    f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}"
                )
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics["oracle_success"],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                }
                with open(result_path, "a", encoding="utf-8") as f:
                    if result_needs_newline:
                        f.write("\n")
                        result_needs_newline = False
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        finally:
            env.close()
        return (
            torch.tensor(sucs, dtype=torch.float64, device=self.device),
            torch.tensor(spls, dtype=torch.float64, device=self.device),
            torch.tensor(oss, dtype=torch.float64, device=self.device),
            torch.tensor(ones, dtype=torch.float64, device=self.device),
            torch.tensor(len(sucs), dtype=torch.float64, device=self.device),
        )


def aggregate_evaluator(evaluator, args, extra_metrics: Optional[Dict] = None):
    """Run one evaluator per rank and reduce metric sums without padding."""
    world_size = get_world_size()
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank())
    totals = torch.stack([sucs.sum(), spls.sum(), oss.sum(), ones.sum(), ep_num])
    if is_dist_avail_and_initialized() and world_size > 1:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    episode_count = int(totals[4].item())
    if episode_count == 0:
        raise RuntimeError(
            "No episodes were evaluated. Check --data_root, --eval_split, and resume files."
        )
    result_all = {
        "success": (totals[0] / episode_count).item(),
        "spl": (totals[1] / episode_count).item(),
        "oracle_success": (totals[2] / episode_count).item(),
        "navigation_error": (totals[3] / episode_count).item(),
        "episodes": episode_count,
        "split": args.eval_split,
        "max_long_memory": args.max_long_memory,
        "recent_size": args.recent_size,
        "max_steps": args.max_steps,
        "limit_episodes": args.limit_episodes,
        "seed": args.seed,
        "world_size": world_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "torch_dtype": args.torch_dtype,
    }
    if extra_metrics:
        result_all.update(extra_metrics)

    print(result_all)
    if get_rank() == 0:
        with open(
            os.path.join(args.output_path, "metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(result_all, f, ensure_ascii=False, indent=2)
    return result_all


def evaluate(model, args):
    evaluator = ShortLabelVLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=get_world_size(),
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args,
    )
    return aggregate_evaluator(
        evaluator,
        args,
        extra_metrics={"eval_mode": "lookstep_event_fifo"},
    )


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    init_distributed_mode(args)
    os.makedirs(args.output_path, exist_ok=True)
    ensure_run_manifest(args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false."
        )
    device = args.device
    if device == "cuda":
        device = f"cuda:{args.local_rank}"

    model = ShortLabelEventFIFOInference(
        pretrained=args.model_path,
        processor_path=args.processor_path,
        device=device,
        args=args,
    )
    evaluate(model, args)


if __name__ == "__main__":
    main()
