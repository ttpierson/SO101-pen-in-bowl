"""Scripted inverse-kinematics pick-and-place policy for the SO-101 pen-in-bowl task.

An oracle baseline: it reads the privileged ``pen_pos`` / ``pen_quat`` / ``bowl_pos`` state
(no cameras) and plays back joint-space trajectories through fixed tool-frame waypoints in
chunks. It plans twice: the pick from the first observation, and the place once the pen is
lifted, aiming the pen's measured centre (wherever it ended up between the jaws) at the bowl.
"""

from __future__ import annotations

import logging

import numpy as np

from inspect_robots import ActionSemantics, Box, ObservationSpace
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from so101_kinematics import SO101Kinematics, quat_wxyz_to_matrix

logger = logging.getLogger(__name__)

ACTION_SPACE = Box(
    shape=(6,),
    semantics=ActionSemantics(control_mode="joint_pos", rotation_repr="none", gripper="binary", frame="base"),
)
OPEN, CLOSE = 0.0, 1.0

# (name, tool target relative to "pen" or "bowl" [m], gripper, steps at 30 Hz)
PICK = (
    ("above_pen", "pen", (0.0, 0.0, 0.08), OPEN, 45),
    ("at_pen", "pen", (0.0, 0.0, 0.011), OPEN, 30),
    ("grasp", "pen", (0.0, 0.0, 0.011), CLOSE, 25),
    ("lift", "pen", (0.0, 0.0, 0.11), CLOSE, 30),
)
# Bowl targets are for the pen's centre, not the tool point.
PLACE = (
    ("above_bowl", "bowl", (0.0, 0.0, 0.12), CLOSE, 45),
    ("lower", "bowl", (0.0, 0.0, 0.065), CLOSE, 30),  # pen just above the rim (~55 mm) before letting go
    ("release", "bowl", (0.0, 0.0, 0.065), OPEN, 30),
    ("retreat", "bowl", (0.0, 0.0, 0.15), OPEN, 30),
)


def _smoothstep(n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    return s * s * (3 - 2 * s)


class IKPickPlacePolicy:
    """Pick the pen with a top-down grasp across its long axis, drop it into the bowl."""

    def __init__(self, *, chunk_size: int = 10, urdf_path: str | None = None) -> None:
        self.kin = SO101Kinematics(urdf_path) if urdf_path else SO101Kinematics()
        self.chunk_size = chunk_size
        self.info = PolicyInfo(
            name="so101_ik_pick_place",
            action_space=ACTION_SPACE,
            observation_space=ObservationSpace(state_keys=frozenset({"joint_pos", "pen_pos", "pen_quat", "bowl_pos"})),
            control_hz=30.0,
        )
        self.config = PolicyConfig(action_horizon=chunk_size)
        self._plan: list[np.ndarray] = []
        self._phase = "pick"
        self._q_end: np.ndarray | None = None  # last commanded arm joints
        self.waypoints: list[dict] = []

    def reset(self, scene: Scene) -> None:
        """Drop the previous trial's plan; the next act() replans from the new observation."""
        self._plan = []
        self._phase = "pick"
        self._q_end = None
        self.waypoints = []

    def _segment(self, plan, q_prev, grip_prev, q_goal, grip, steps):
        for s in _smoothstep(steps):
            # Close/open the gripper only once the arm has arrived at the waypoint.
            g = grip if (grip == grip_prev or s >= 1.0) else grip_prev
            plan.append(np.append(q_prev + s * (q_goal - q_prev), g))

    def _log(self, name, target, err):
        self.waypoints.append({"name": name, "target": np.round(target, 4).tolist(), "ik_err_m": round(err, 4)})
        if err > 0.01:
            logger.warning("IK for %s misses its target by %.1f mm", name, err * 1000)

    def _plan_pick(self, obs: Observation) -> list[np.ndarray]:
        q = np.asarray(obs.state["joint_pos"], dtype=np.float64)
        pen = np.asarray(obs.state["pen_pos"], dtype=np.float64)
        pen[2] = 0.0  # targets are heights above the tabletop
        pen_axis = quat_wxyz_to_matrix(obs.state["pen_quat"])[:, 1]  # the pen lies along its local Y

        plan: list[np.ndarray] = []
        q_prev, grip_prev = q, OPEN
        for name, _, offset, grip, steps in PICK:
            target = pen + np.asarray(offset)
            q_goal, err = self.kin.ik(target, q_prev, grasp_axis=pen_axis)
            self._log(name, target, err)
            self._segment(plan, q_prev, grip_prev, q_goal, grip, steps)
            q_prev, grip_prev = q_goal, grip
        self._q_end = q_prev
        return plan

    def _plan_place(self, obs: Observation) -> list[np.ndarray]:
        # Where the pen's centre actually sits in the gripper_link frame after the lift.
        q_now = np.asarray(obs.state["joint_pos"], dtype=np.float64)
        G = self.kin.fk(q_now)
        pen_in_grip = G[:3, :3].T @ (np.asarray(obs.state["pen_pos"], dtype=np.float64) - G[:3, 3])
        bowl = np.asarray(obs.state["bowl_pos"], dtype=np.float64)

        plan: list[np.ndarray] = []
        q_prev, grip_prev = self._q_end, CLOSE
        for name, _, offset, grip, steps in PLACE:
            want = bowl + np.asarray(offset)  # desired pen centre
            target = want.copy()  # tool point target, corrected until the pen centre lands on `want`
            for _ in range(4):
                q_goal, _ = self.kin.ik(target, q_prev)
                q_goal[4] = q_prev[4]  # keep the wrist roll from the grasp while carrying
                Gg = self.kin.fk(q_goal)
                target -= Gg[:3, :3] @ pen_in_grip + Gg[:3, 3] - want
            self._log(name, want, float(np.linalg.norm(Gg[:3, :3] @ pen_in_grip + Gg[:3, 3] - want)))
            self._segment(plan, q_prev, grip_prev, q_goal, grip, steps)
            q_prev, grip_prev = q_goal, grip
        self._q_end = q_prev
        return plan

    def act(self, observation: Observation) -> ActionChunk:
        """Plan the pick, then (once the pen is lifted) the place, and hand them out chunk by chunk."""
        if not self._plan and self._phase in ("pick", "place"):
            self._plan = self._plan_pick(observation) if self._phase == "pick" else self._plan_place(observation)
            self._phase = "place" if self._phase == "pick" else "done"
        if not self._plan:  # trajectory finished: hold the last pose
            return ActionChunk(actions=[Action(data=self._last)])
        chunk, self._plan = self._plan[: self.chunk_size], self._plan[self.chunk_size :]
        self._last = chunk[-1]
        return ActionChunk(actions=[Action(data=a) for a in chunk])

    def transcript(self) -> dict:
        """Record the planned waypoints and their IK errors in the eval log."""
        return {"waypoints": [dict(w) for w in self.waypoints]}
