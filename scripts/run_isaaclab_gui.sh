#!/bin/bash
# Launch an Isaac Lab script on the GPU desktop (:1). Usage: run_isaaclab_gui.sh [script.py] [args]
export DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=YES
source /data/env_isaaclab/bin/activate
cd /data/IsaacLab
./isaaclab.sh -p "${1:-scripts/demos/quadrupeds.py}" "${@:2}"
