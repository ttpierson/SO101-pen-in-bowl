"""Isaac Lab task ``Isaac-SO101-PenBowl-v0``: the SO-101 must put a pen into a bowl.

Importing this module (after the Isaac app has launched) registers the gym id.

Action (6-D, 30 Hz): 5 absolute arm joint targets (rad, URDF order) + a binary gripper
command where >= 0.5 means CLOSE (the openpi / DROID convention).

Observation group ``"policy"`` (named terms, num_envs-batched, positions in the robot base
frame): joint_pos, joint_vel, eef_pos, eef_quat (wxyz), gripper (0 closed .. 1 open),
overhead / wrist RGB images, and the privileged pen_pos, pen_quat (wxyz) and bowl_pos
(centre of the bowl's bottom). ``info["success"]``: pen at rest inside the bowl with the
gripper open; the episode terminates when that happens.
"""

from __future__ import annotations

import math
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import trimesh
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_cfg import ARM_JOINTS, GRIPPER_JOINT, SO101_CFG  # noqa: E402

TASK_ID = "Isaac-SO101-PenBowl-v0"
TABLE_H = 0.75
CAM_H, CAM_W = 240, 320
BOWL_USD = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/024_bowl.usd"
GRIPPER_OPEN, GRIPPER_CLOSED = 1.2, -0.17  # rad; closing past the pen's width makes the servo squeeze


@clone
def spawn_hex_pen(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A hexagonal pencil along Y, flat face down. A round cylinder keeps rolling away in PhysX."""
    to_y = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = trimesh.creation.cylinder(radius=cfg.radius, height=cfg.height, sections=6, transform=to_y)
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation)
    return sim_utils.get_current_stage().GetPrimAtPath(prim_path)


@configclass
class PenBowlEnvCfg(DirectRLEnvCfg):
    decimation = 4  # 120 Hz physics, 30 Hz control
    episode_length_s = 20.0
    action_space = 6
    observation_space = {
        "joint_pos": 5, "joint_vel": 5, "eef_pos": 3, "eef_quat": 4, "gripper": 1, "joint_cmd": 6,
        "pen_pos": 3, "pen_quat": 4, "bowl_pos": 3,
        "overhead": [CAM_H, CAM_W, 3], "wrist": [CAM_H, CAM_W, 3],
    }  # fmt: skip
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    # GUI viewport (-E headless=false): a close three-quarter view of the arm, pen and bowl.
    viewer: ViewerCfg = ViewerCfg(eye=(0.75, -0.55, TABLE_H + 0.45), lookat=(0.17, 0.0, TABLE_H))

    robot = SO101_CFG.replace(
        prim_path="/World/envs/env_.*/Robot", init_state=SO101_CFG.init_state.replace(pos=(0.0, 0.0, TABLE_H))
    )
    pen = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pen",
        spawn=sim_utils.MeshCylinderCfg(
            func=spawn_hex_pen,
            radius=0.0063,  # corner radius; 11 mm across the flats, resting with its axis 5.5 mm up
            height=0.14,
            axis="Y",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, -0.09, TABLE_H + 0.0056)),
    )
    # Overhead: 0.7 m above the tabletop looking straight down ("world" convention looks along +X).
    overhead_cam = TiledCameraCfg(
        prim_path="/World/envs/env_.*/OverheadCam",
        height=CAM_H,
        width=CAM_W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.2, 0.0, TABLE_H + 0.7), rot=(0.7071068, 0.0, 0.7071068, 0.0), convention="world"),
    )
    # Wrist: just in front of the lens of the URDF's wrist camera (lens tip at z=26 mm, looking along +Z).
    wrist_cam = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/wrist_camera_link/wrist_cam",
        height=CAM_H,
        width=CAM_W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.01, 3.0)),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.028), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )

    bowl_pos = (0.17, 0.14)  # bowl centre on the table, robot base frame (fixed)
    pen_x_range = (0.15, 0.22)
    pen_y_range = (-0.13, -0.05)
    success_radius = 0.055  # pen centre within this of the bowl axis (bowl rim radius ~0.08)
    success_max_height = 0.05  # pen centre below this height above the tabletop (bowl rim ~0.055)


class PenBowlEnv(DirectRLEnv):
    cfg: PenBowlEnvCfg

    def __init__(self, cfg: PenBowlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._arm_ids, _ = self.robot.find_joints(list(ARM_JOINTS), preserve_order=True)
        self._grip_id = self.robot.find_joints(GRIPPER_JOINT)[0][0]
        self._eef_id = self.robot.find_bodies("gripper_frame_link")[0][0]
        self._targets = self.robot.data.default_joint_pos.clone()
        self._gripper_closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._bowl_pos_b = torch.tensor([*cfg.bowl_pos, 0.0], device=self.device).repeat(self.num_envs, 1)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.pen = RigidObject(self.cfg.pen)
        self.overhead_cam = TiledCamera(self.cfg.overhead_cam)
        self.wrist_cam = TiledCamera(self.cfg.wrist_cam)

        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
        table = sim_utils.CuboidCfg(
            size=(0.8, 1.0, TABLE_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.28)),
        )
        table.func("/World/envs/env_0/Table", table, translation=(0.2, 0.0, TABLE_H / 2))
        self._spawn_bowl("/World/envs/env_0/Bowl")

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["pen"] = self.pen
        self.scene.sensors["overhead_cam"] = self.overhead_cam
        self.scene.sensors["wrist_cam"] = self.wrist_cam

    def _spawn_bowl(self, path: str) -> None:
        """YCB bowl as a static, exact (concave) triangle-mesh collider resting on the tabletop."""
        bowl = sim_utils.UsdFileCfg(usd_path=BOWL_USD)
        # The YCB "Axis_Aligned" meshes are Y-up; rotate -90 deg about X so the bowl opens upward.
        bowl.func(path, bowl, translation=(*self.cfg.bowl_pos, TABLE_H), orientation=(0.7071068, -0.7071068, 0.0, 0.0))
        stage = sim_utils.get_current_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
        # The mesh origin is not at its base: lift it so the bottom rests on the tabletop.
        min_z = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
            stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMin()[2]  # fmt: skip
        stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Set((*self.cfg.bowl_pos, 2 * TABLE_H + 0.001 - min_z))

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        lim = self.robot.data.soft_joint_pos_limits[:, self._arm_ids]
        self._targets[:, self._arm_ids] = torch.clamp(actions[:, :5], lim[..., 0], lim[..., 1])
        self._gripper_closed = actions[:, 5] >= 0.5
        self._targets[:, self._grip_id] = torch.where(self._gripper_closed, GRIPPER_CLOSED, GRIPPER_OPEN)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._targets)

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        base = self.robot.data.root_pos_w
        grip = self.robot.data.joint_pos[:, self._grip_id : self._grip_id + 1]
        obs = {
            "joint_pos": self.robot.data.joint_pos[:, self._arm_ids],
            "joint_vel": self.robot.data.joint_vel[:, self._arm_ids],
            "eef_pos": self.robot.data.body_pos_w[:, self._eef_id] - base,
            "eef_quat": self.robot.data.body_quat_w[:, self._eef_id],
            "gripper": ((grip - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED)).clamp(0.0, 1.0),
            # The current state in action units: arm joints + commanded gripper (1 = closed).
            "joint_cmd": torch.cat(
                [self.robot.data.joint_pos[:, self._arm_ids], self._gripper_closed.float().unsqueeze(-1)], dim=-1
            ),
            "pen_pos": self.pen.data.root_pos_w - base,
            "pen_quat": self.pen.data.root_quat_w,
            "bowl_pos": self._bowl_pos_b,
            "overhead": self.overhead_cam.data.output["rgb"],
            "wrist": self.wrist_cam.data.output["rgb"],
        }
        return {"policy": obs}

    def _success(self) -> torch.Tensor:
        pen_b = self.pen.data.root_pos_w - self.robot.data.root_pos_w
        radial = torch.linalg.norm(pen_b[:, :2] - self._bowl_pos_b[:, :2], dim=-1)
        at_rest = torch.linalg.norm(self.pen.data.root_lin_vel_w, dim=-1) < 0.05
        return (
            (radial < self.cfg.success_radius)
            & (pen_b[:, 2] < self.cfg.success_max_height)
            & at_rest
            & ~self._gripper_closed
        )

    def _get_rewards(self) -> torch.Tensor:
        return self._success().float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = self._success()
        self.extras["success"] = success
        return success, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._grip_id] = GRIPPER_OPEN
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self._targets[env_ids] = joint_pos
        self._gripper_closed[env_ids] = False

        # Pen: random position in its patch and random yaw, lying flat on the table.
        pose = self.pen.data.default_root_state[env_ids].clone()
        pose[:, 0] = sample_uniform(*self.cfg.pen_x_range, (n,), self.device)
        pose[:, 1] = sample_uniform(*self.cfg.pen_y_range, (n,), self.device)
        yaw = sample_uniform(-math.pi / 2, math.pi / 2, (n,), self.device)
        pose[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        pose[:, :3] += self.scene.env_origins[env_ids]
        pose[:, 7:] = 0.0
        self.pen.write_root_state_to_sim(pose, env_ids=env_ids)


gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}:PenBowlEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:PenBowlEnvCfg"},
)
