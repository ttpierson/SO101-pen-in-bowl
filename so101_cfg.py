"""SO-101 articulation config for Isaac Lab (USD converted from SO-ARM100 so101_new_calib_camera.urdf,
the variant with the wrist camera mount; the camera body is `wrist_camera_link`)."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

SO101_USD = os.environ.get("SO101_USD", "/data/so101/usd_cam/so101_new_calib_camera.usd")

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SO101_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={j: 0.0 for j in ARM_JOINTS + [GRIPPER_JOINT]},
    ),
    actuators={
        # STS3215 hobby servos: soft position gains, ~10 Nm-scale limits (from the URDF).
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS, effort_limit_sim=10.0, velocity_limit_sim=10.0, stiffness=17.8, damping=0.6
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[GRIPPER_JOINT], effort_limit_sim=10.0, velocity_limit_sim=10.0, stiffness=17.8, damping=0.6
        ),
    },
)
