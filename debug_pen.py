"""Debug: hold the arm still and watch whether the pen moves on its own."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_robots import Scene  # noqa: E402
from inspect_robots.types import Action  # noqa: E402

from run_eval import SO101PenBowlEmbodiment  # noqa: E402

emb = SO101PenBowlEmbodiment(headless=True)
for seed in (0, 1):
    obs = emb.reset(Scene(id="dbg", instruction="x"), seed=seed)
    hold = np.append(obs.state["joint_pos"], 0.0)
    for t in range(121):
        if t % 30 == 0:
            env = emb._env.unwrapped
            pen = env.pen.data
            print(f"[pen] seed={seed} t={t} pos={np.round(obs.state['pen_pos'], 4)} quat={np.round(obs.state['pen_quat'], 3)}"
                  f" v={np.round(pen.root_lin_vel_w[0].cpu().numpy(), 4)} w={np.round(pen.root_ang_vel_w[0].cpu().numpy(), 3)}", flush=True)
        obs = emb.step(Action(data=hold)).observation
emb.close()
