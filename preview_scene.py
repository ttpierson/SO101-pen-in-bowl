"""Preview: SO-101 on a primitive table with a YCB bowl and a primitive pen, plus an overhead and a wrist camera.

Camera frames are written to /data/so101_isaac/frames/{overhead,wrist}.png every 5 s. The arm holds its
initial pose; pass --sweep to sweep the joints.

Run on the GPU desktop:  /data/run_isaaclab_gui.sh /data/so101_isaac/preview_scene.py
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--sweep", action="store_true", help="Sweep every joint instead of holding the initial pose.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import torch  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from PIL import Image  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_cfg import SO101_CFG  # noqa: E402

TABLE_H = 0.75
FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")
BOWL_USD = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/024_bowl.usd"


def make_static_trimesh_collider(stage: Usd.Stage, root: str) -> None:
    """Give every mesh under `root` an exact (concave) triangle-mesh collider. Fine for static bodies."""
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
            PhysxSchema.PhysxCollisionAPI.Apply(prim)


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim.set_camera_view(eye=[0.75, 0.75, TABLE_H + 0.5], target=[0.1, 0.0, TABLE_H])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    # Table: a primitive cuboid, static collider.
    table = sim_utils.CuboidCfg(
        size=(0.8, 1.0, TABLE_H),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.28)),
    )
    table.func("/World/table", table, translation=(0.2, 0.0, TABLE_H / 2))

    # Bowl: YCB mesh from the NVIDIA asset server, made a static concave collider.
    bowl = sim_utils.UsdFileCfg(usd_path=BOWL_USD)
    # YCB "Axis_Aligned" meshes are Y-up; rotate -90 deg about X so the bowl opens upward.
    bowl.func("/World/bowl", bowl, translation=(0.25, 0.12, TABLE_H + 0.001), orientation=(0.7071068, -0.7071068, 0.0, 0.0))
    stage = sim_utils.get_current_stage()
    make_static_trimesh_collider(stage, "/World/bowl")
    # The YCB mesh origin is not at its base: lift it so the bottom rests on the tabletop.
    bowl_min_z = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
        stage.GetPrimAtPath("/World/bowl")).ComputeAlignedRange().GetMin()[2]
    stage.GetPrimAtPath("/World/bowl").GetAttribute("xformOp:translate").Set((0.25, 0.12, 2 * TABLE_H + 0.002 - bowl_min_z))

    # Pen: a primitive cylinder, dynamic rigid body.
    pen = RigidObject(
        RigidObjectCfg(
            prim_path="/World/pen",
            spawn=sim_utils.CylinderCfg(
                radius=0.0055,
                height=0.14,
                axis="Y",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.8)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.1, TABLE_H + 0.006)),
        )
    )

    robot_cfg = SO101_CFG.replace(prim_path="/World/so101")
    robot_cfg.init_state.pos = (0.0, 0.0, TABLE_H)
    robot = Articulation(robot_cfg)

    # Overhead camera: 0.7 m above the tabletop, centred on the workspace, looking straight down
    # ("world" convention: the camera looks along +X, so rotate +90 deg about Y to face -Z).
    overhead = Camera(
        CameraCfg(
            prim_path="/World/overhead_cam",
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.22, 0.0, TABLE_H + 0.7), rot=(0.7071068, 0.0, 0.7071068, 0.0), convention="world"),
        )
    )
    # Wrist camera: rides on the URDF's wrist_camera_link. The lens points along the link's +Z with its tip at
    # z=26 mm, so sit just in front of it ("ros" convention: the camera looks along +Z).
    wrist = Camera(
        CameraCfg(
            prim_path="/World/so101/wrist_camera_link/wrist_cam",
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.01, 3.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.028), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        )
    )
    os.makedirs(FRAMES_DIR, exist_ok=True)

    sim.reset()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(stage.GetPrimAtPath("/World/bowl"))
    print("[so101] joints:", robot.joint_names)
    print("[so101] bodies:", robot.body_names)
    print("[so101] joint limits:", robot.data.soft_joint_pos_limits[0].tolist())
    print("[so101] bowl world bbox:", bbox.ComputeAlignedRange())
    print("[so101] pen pos:", pen.data.root_pos_w[0].tolist(), flush=True)

    lim = robot.data.soft_joint_pos_limits[0]
    mid, half = (lim[:, 0] + lim[:, 1]) / 2, (lim[:, 1] - lim[:, 0]) / 2
    t = 0.0
    while app.is_running():
        # Gentle sweep of every joint (40% of its range), each at a different rate.
        phase = torch.arange(robot.num_joints, device=sim.device, dtype=torch.float32)
        target = mid + 0.4 * half * torch.sin(t * (0.4 + 0.15 * phase) + phase)
        if not args.sweep:
            target = robot.data.default_joint_pos[0]
        robot.set_joint_position_target(target.unsqueeze(0))
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        pen.update(sim.get_physics_dt())
        overhead.update(sim.get_physics_dt())
        wrist.update(sim.get_physics_dt())
        t += sim.get_physics_dt()
        if int(t * 120) % 600 == 0:
            ee = robot.data.body_pos_w[0, -1].tolist()
            for name, cam in (("overhead", overhead), ("wrist", wrist)):
                Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy()).save(f"{FRAMES_DIR}/{name}.png")
            print(f"[so101] t={t:5.1f}s last-body pos={[round(v, 3) for v in ee]} pen={[round(v, 3) for v in pen.data.root_pos_w[0].tolist()]}", flush=True)


if __name__ == "__main__":
    main()
    app.close()
