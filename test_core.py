"""Fast, dependency-light regression tests for deterministic LookStep logic."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    from LookStep.data_construction import build_short_label_dataset as labels
    from LookStep.model_training import prepare_ms_swift_data as swift_data
    from LookStep.simulation import verify_paper_results as result_verifier
except ModuleNotFoundError:
    from data_construction import build_short_label_dataset as labels
    from model_training import prepare_ms_swift_data as swift_data
    from simulation import verify_paper_results as result_verifier


class LabelConstructionTests(unittest.TestCase):
    def setUp(self):
        self.actions = [
            "MOVE_FORWARD",
            "TURN_LEFT",
            "TURN_LEFT",
            "MOVE_FORWARD",
            "STOP",
        ]

    def test_turn_and_alignment_events(self):
        self.assertEqual(labels.event_label(self.actions, 1, 5), "turn_left_start")
        self.assertEqual(labels.event_label(self.actions, 2, 5), "turn_left_finishing")
        self.assertEqual(labels.event_label(self.actions, 3, 5), "post_turn_alignment")
        self.assertEqual(labels.event_label(self.actions, 4, 5), "stop_now")

    def test_memory_roles(self):
        self.assertEqual(
            labels.memory_write_label(self.actions, 0, 5),
            {"memory_write": "keep", "memory_role": "start_view"},
        )
        self.assertEqual(
            labels.memory_write_label(self.actions, 3, 5),
            {
                "memory_write": "keep",
                "memory_role": "post_turn_alignment",
            },
        )
        self.assertEqual(
            labels.memory_write_label(self.actions, 4, 5),
            {"memory_write": "keep", "memory_role": "stop_evidence"},
        )

    def test_candidate_outcomes(self):
        turn = labels.action_outcomes(self.actions, 1, 5)
        self.assertEqual(turn["TURN_LEFT"], "start_turn")
        self.assertEqual(turn["MOVE_FORWARD"], "premature_forward")
        stop = labels.action_outcomes(self.actions, 4, 5)
        self.assertEqual(stop["STOP"], "success_stop")
        self.assertEqual(stop["MOVE_FORWARD"], "overshoot_goal")

    def test_prompt_image_alignment(self):
        image_files = [Path(f"episode/step_{index:04d}.png") for index in range(5)]
        long_memory = [
            {
                "path": image_files[0],
                "step_index": 0,
                "action": self.actions[0],
                "memory_role": "start_view",
            }
        ]
        recent = [
            {
                "path": image_files[1],
                "step_index": 1,
                "action": self.actions[1],
                "memory_role": "recent_only",
            }
        ]
        sample = labels.make_event_fifo_sample(
            dataset="r2r",
            episode_id="test",
            instruction="turn left",
            image_files=image_files,
            target_actions=self.actions,
            step_idx=2,
            future_window=5,
            path_mode="relative",
            long_memory=long_memory,
            recent_buffer=recent,
            max_long_memory=6,
            recent_size=2,
        )
        prompt = sample["conversations"][0]["value"]
        self.assertEqual(prompt.count("<image>"), len(sample["images"]))
        json.dumps(sample)

class DataConversionTests(unittest.TestCase):
    def test_episode_split_is_deterministic_and_disjoint(self):
        episodes = [f"episode-{index}" for index in range(100)]
        first = swift_data.choose_val_episodes(episodes, 0.2, -1, 7, False)
        second = swift_data.choose_val_episodes(episodes, 0.2, -1, 7, False)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertFalse(first.intersection(set(episodes) - first))

    def test_converter_rejects_image_tag_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            image_path.touch()
            sample = {
                "id": "bad-image-alignment",
                "images": [str(image_path)],
                "conversations": [
                    {"from": "human", "value": "no image marker"},
                    {"from": "gpt", "value": "<action>STOP</action>"},
                ],
            }
            with self.assertRaisesRegex(ValueError, "<image> tags"):
                swift_data.convert_sample(
                    sample=sample,
                    path_mode="absolute",
                    image_root=Path(directory),
                    keep_extra=False,
                    allow_image_tag_mismatch=False,
                )


class TrainingDataProvenanceTests(unittest.TestCase):
    def test_public_constructor_rejects_scalevln(self):
        with self.assertRaisesRegex(ValueError, "Unsupported main-experiment dataset"):
            labels.selected_episode_sources(
                "scalevln", Path("."), Path("."), limit_episodes=-1
            )


class ResultVerificationTests(unittest.TestCase):
    def test_metric_normalization_and_episode_count(self):
        actual = result_verifier.normalized_metrics(
            {
                "episodes": 1839,
                "navigation_error": 5.34,
                "oracle_success": 0.559,
                "success": 0.497,
                "spl": 0.453,
            }
        )
        expected = {
            "episodes": 1839,
            "navigation_error": 5.34,
            "oracle_success": 55.9,
            "success": 49.7,
            "spl": 45.3,
        }
        failures, _ = result_verifier.compare("R2R", actual, expected, 1e-9, 1e-9)
        self.assertEqual(failures, 0)

        actual["episodes"] = 2
        failures, _ = result_verifier.compare("R2R", actual, expected, 1.0, 0.15)
        self.assertEqual(failures, 1)

    def test_recipe_verifier_rejects_smoke_configuration(self):
        recipe = {
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
        failures, _ = result_verifier.compare_recipe("R2R", recipe)
        self.assertEqual(failures, 0)
        recipe["limit_episodes"] = 2
        failures, _ = result_verifier.compare_recipe("R2R", recipe)
        self.assertEqual(failures, 1)


if __name__ == "__main__":
    unittest.main()
