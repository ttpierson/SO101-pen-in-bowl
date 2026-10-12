"""Run the SO-101 pen-in-bowl task through inspect-robots with the scripted IK policy.

    python run_eval.py [--scenes 5] [--gui] [--log-dir /data/so101_isaac/logs]

The same run from the inspect-robots CLI (entry points in pyproject.toml / so101_inspect.py):

    inspect-robots run --task so101-pen-in-bowl --policy so101_ik --embodiment so101_pen_bowl --store-frames

With --gui it renders on the noVNC desktop (DISPLAY=:1) so you can watch.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from inspect_robots import ObservationSpace, StateField, StateSpec, eval  # noqa: E402
from inspect_robots_isaacsim import embodiment as isaac_embodiment  # noqa: E402
from inspect_robots_isaacsim.embodiment import IsaacSimEmbodiment  # noqa: E402

from ik_policy import IKPickPlacePolicy  # noqa: E402
from so101_kinematics import SO101Kinematics  # noqa: E402

TASK_ID = "Isaac-SO101-PenBowl-v0"
CAM_H, CAM_W = 240, 320


class SO101PenBowlEmbodiment(IsaacSimEmbodiment):
    """The generic Isaac adapter, plus privileged object state and camera rendering."""

    def __init__(self, *, headless: bool = True) -> None:
        super().__init__(
            task_id=TASK_ID,
            num_arm_joints=5,
            cameras=(("overhead", CAM_H, CAM_W, 3), ("wrist", CAM_H, CAM_W, 3)),
            control_hz=30.0,
            headless=headless,
            name="so101_pen_bowl",
        )
        obs = self.info.observation_space
        # Joint limits from the URDF plus the 0..1 gripper command, so the default guardrails
        # (bounds clamp + per-step delta limit) are active. 0.1 rad/step = 3 rad/s at 30 Hz.
        kin = SO101Kinematics()
        act = self.info.action_space
        action_space = dataclasses.replace(
            act,
            low=np.append(kin.lower, 0.0),
            high=np.append(kin.upper, 1.0),
            semantics=dataclasses.replace(act.semantics, max_step=(0.1,) * 5 + (1.0,)),
        )
        privileged = (
            StateField("pen_pos", (3,), "m"),
            StateField("pen_quat", (4,), "unit_quat"),
            StateField("bowl_pos", (3,), "m"),
        )
        self.info = dataclasses.replace(
            self.info,
            action_space=action_space,
            observation_space=ObservationSpace(cameras=obs.cameras, state=StateSpec(fields=obs.state.fields + privileged)),
            docs=(
                "SO-101 on a table; put the blue pen in the red bowl. Action: 5 absolute arm joint targets "
                "(rad) + binary gripper (>=0.5 closes). Positions are in the robot base frame; pen_pos, "
                "pen_quat and bowl_pos are privileged simulator state."
            ),
        )

    def _ensure_app(self):
        # Launch the app ourselves so camera rendering is enabled, then let the base class adopt it.
        if self._app is None and isaac_embodiment._ACTIVE_APP is None:
            from isaaclab.app import AppLauncher

            isaac_embodiment._ACTIVE_APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
        app = super()._ensure_app()
        import pen_bowl_env  # noqa: F401  (registers the gym id; needs the app running)

        return app

    def close(self) -> None:
        # SimulationApp.close() ends the process, so closing it here would swallow whatever the
        # caller prints afterwards (the inspect-robots CLI prints its results after close()).
        # Close the env now and leave the app to shut down at interpreter exit.
        app = self._app
        if app is not None:
            self._app = None
            if isaac_embodiment._ACTIVE_APP is app:
                isaac_embodiment._ACTIVE_APP = None
            atexit.register(app.close)
        super().close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--log-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    args = parser.parse_args()

    from so101_inspect import pen_bowl_task

    task = pen_bowl_task(scenes=args.scenes)
    embodiment = SO101PenBowlEmbodiment(headless=not args.gui)
    try:
        (log,) = eval(task, IKPickPlacePolicy(), embodiment, log_dir=args.log_dir, store_frames=True)
        print(f"[eval] status={log.status} metrics={log.results.metrics}", flush=True)
        for scene in log.samples:
            print(f"[eval] {scene.scene_id}: {scene.status} {scene.reduced} {scene.error or ''}", flush=True)
    finally:
        embodiment.close()


if __name__ == "__main__":
    main()
