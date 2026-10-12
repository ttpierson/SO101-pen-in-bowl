"""Inspect Robots entry points for the SO-101 pen-in-bowl setup (see pyproject.toml).

    inspect-robots run --task so101-pen-in-bowl --policy so101_ik --embodiment so101_pen_bowl \\
        --store-frames --no-prompt [-T scenes=10] [-E headless=false]

Importing this module is cheap: Isaac Sim only starts when the embodiment first resets.
"""

from __future__ import annotations

from typing import Any

from inspect_robots import Scene
from inspect_robots.scorer import episode_length, success_at_end
from inspect_robots.task import Task


def ik_policy(**kwargs: Any) -> Any:
    """Policy ``so101_ik``: scripted IK pick-and-place on privileged pen/bowl state."""
    from ik_policy import IKPickPlacePolicy

    return IKPickPlacePolicy(**kwargs)


def pen_bowl_embodiment(**kwargs: Any) -> Any:
    """Embodiment ``so101_pen_bowl``: the Isaac Lab scene; ``-E headless=false`` renders on DISPLAY."""
    from run_eval import SO101PenBowlEmbodiment

    return SO101PenBowlEmbodiment(**kwargs)


def pen_bowl_task(scenes: int = 5, max_steps: int = 330) -> Task:
    """Task ``so101-pen-in-bowl``: ``scenes`` pen layouts (seeds 0..scenes-1), success at the end."""
    return Task(
        name="so101-pen-in-bowl",
        scenes=[Scene(id=f"pen-layout-{i}", instruction="put the pen in the bowl", init_seed=i) for i in range(scenes)],
        scorer=[success_at_end(), episode_length()],
        max_steps=max_steps,  # 11 s at 30 Hz; the scripted trajectory takes ~8.5 s
    )
