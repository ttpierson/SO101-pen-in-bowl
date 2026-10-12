"""Debug: compare numpy FK against Isaac's body poses while playing the IK plan up to the grasp."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_robots import Scene  # noqa: E402

from ik_policy import IKPickPlacePolicy  # noqa: E402
from run_eval import SO101PenBowlEmbodiment  # noqa: E402

emb = SO101PenBowlEmbodiment(headless=True)
pol = IKPickPlacePolicy()
scene = Scene(id="dbg", instruction="put the pen in the bowl")
obs = emb.reset(scene, seed=0)
pol.reset(scene)
GF = np.array([-0.0079, -0.000218, -0.0981274])  # gripper_frame_link in gripper_link
t = 0
while t < 100:
    for a in pol.act(obs).actions:
        res = emb.step(a)
        obs = res.observation
        t += 1
        if t in (1, 45, 60, 75, 100):
            q = obs.state["joint_pos"]
            G = pol.kin.fk(q)
            print(f"[dbg] t={t} q={np.round(q, 3)} cmd={np.round(a.data[:5], 3)}")
            print(f"[dbg]   isaac eef(gripper_frame)={np.round(obs.state['eef_pos'], 4)} fk={np.round(G[:3, :3] @ GF + G[:3, 3], 4)}")
            print(f"[dbg]   fk tcp={np.round(pol.kin.tcp(q), 4)} pen={np.round(obs.state['pen_pos'], 4)} gripper={obs.state['gripper']}")
emb.close()
