#!/bin/bash
# Run the scripted-IK pen-in-bowl eval through the inspect-robots CLI, then render the replay pages.
#
#   ./ik_eval.sh [--gui] [any inspect-robots run args, e.g. -T scenes=10 --seed 3]
#
#   --gui  also render the sim live on the noVNC desktop (http://localhost:6080/vnc.html)
#
# Watch live / replay in the browser: http://localhost:8300 (served by `inspect-robots view logs --serve`).
set -e
source /data/env_isaaclab/bin/activate
cd "$(dirname "$(readlink -f "$0")")"
export OMNI_KIT_ACCEPT_EULA=YES

extra=()
if [[ "$1" == "--gui" ]]; then
    shift
    /data/start_desktop.sh > /dev/null
    export DISPLAY=:1
    extra=(-E headless=false)
fi

inspect-robots run --task so101-pen-in-bowl --policy so101_ik --embodiment so101_pen_bowl \
    --store-frames --no-prompt --log-dir logs "${extra[@]}" "$@"

# The live server only renders quick pages; this pass adds the per-trial MP4 replays.
inspect-robots view logs > /dev/null
echo "replay: http://localhost:8300  (or: inspect-robots view logs --open)"
