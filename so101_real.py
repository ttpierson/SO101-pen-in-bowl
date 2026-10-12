"""Inspect Robots embodiment ``so101_real``: the physical SO-101 with its wrist camera.

A thin factory over ``inspect-robots-so101``'s ``SOArmEmbodiment`` that fills in what
its CLI can't (the camera config object) and derives the joint safety limits from
the arm's own LeRobot calibration file instead of the plugin's ±180° placeholders.

    inspect-robots run --instruction "put the pen in the bowl" --embodiment so101_real \\
        --policy agent -P model=openai/gpt-6-astra -P prior_learnings=real_notes.md --store-frames

Units are LeRobot degrees: arm joints in degrees around the calibrated mid-range,
gripper 0..100. ``-E key=value`` overrides any keyword below.
"""

from __future__ import annotations

import json
import os
from typing import Any

CALIB_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TICKS_PER_REV = 4095  # STS3215 resolution - 1, as LeRobot's degree conversion uses


def calibrated_limits(robot_id: str, margin_deg: float = 5.0) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Joint limits in LeRobot degrees from ``<robot_id>.json``, pulled in by ``margin_deg``.

    LeRobot maps ticks to degrees as ``(ticks - mid) * 360 / 4095`` with ``mid`` the middle
    of the calibrated range, so each joint spans ``±(max - min) / 2`` ticks around zero.
    """
    with open(os.path.join(CALIB_DIR, f"{robot_id}.json")) as f:
        calib = json.load(f)
    half = [(calib[j]["range_max"] - calib[j]["range_min"]) / 2 * 360 / TICKS_PER_REV - margin_deg for j in JOINTS]
    return tuple(-h for h in half) + (0.0,), tuple(half) + (100.0,)


def real_embodiment(
    port: str = "/dev/ttyACM0",
    robot_id: str = "my_follower",
    camera: str = "/dev/video0",
    cam_width: int = 640,
    cam_height: int = 480,
    fps: int = 30,
    max_relative_target: float = 3.0,  # degrees per step at 30 Hz: at most 90 deg/s per joint
    margin_deg: float = 5.0,
    **kwargs: Any,
) -> Any:
    """Build the SO-101 follower embodiment with one camera named ``wrist``."""
    from inspect_robots_so101 import SOArmConfig, SOArmEmbodiment
    from lerobot.cameras.opencv import OpenCVCameraConfig

    low, high = calibrated_limits(robot_id, margin_deg)
    cfg = SOArmConfig(
        port=port,
        robot_type="so101_follower",
        robot_id=robot_id,
        cameras=("wrist",),
        camera_configs={
            "wrist": OpenCVCameraConfig(index_or_path=camera, width=cam_width, height=cam_height, fps=fps)
        },
        cam_width=cam_width,
        cam_height=cam_height,
        joint_low=low,
        joint_high=high,
        max_relative_target=max_relative_target,
        **kwargs,
    )
    return SOArmEmbodiment(cfg)
