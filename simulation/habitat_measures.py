"""Minimal Habitat measures required by the LookStep VLN-CE configs.

Importing this module registers the measures with Habitat's registry.  Keeping
the two small measures here avoids pulling in the repository's larger custom
measurement module (and its optional DTW dependencies).
"""

from typing import Any

from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import DistanceToGoal


@registry.register_measure
class OracleNavigationError(Measure):
    """Minimum distance-to-goal observed during an episode."""

    cls_uuid = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = min(self._metric, distance)


@registry.register_measure
class OracleSuccess(Measure):
    """Whether the agent ever entered the success radius."""

    cls_uuid = "oracle_success"

    def __init__(self, *args: Any, config: Any, **kwargs: Any):
        self._config = config
        self._success_distance = float(getattr(config, "success_distance", 3.0))
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(bool(self._metric) or distance < self._success_distance)
