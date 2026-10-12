#!/bin/bash
# Run "put the pen in the bowl" on the PHYSICAL SO-101 with GPT-6 Astra as the policy.
# Run it yourself in a real terminal: it asks you to confirm the scene is ready before the arm
# moves, Esc ends a trial early, and you grade each trial (y/n/partial) when it ends.
#
#   ./real_eval.sh [extra inspect-robots run args, e.g. --epochs 3 -P effort=low]
#
# Needs: OPENROUTER_API_KEY in ./.env, read access to the wrist camera (/dev/video0).
set -e
cd "$(dirname "$(readlink -f "$0")")"
for dev in /dev/ttyACM0 /dev/video0; do
    [[ -r $dev && -w $dev ]] || { echo "no read/write access to $dev (see README: dialout/video groups)"; exit 1; }
done
grep -q '^OPENROUTER_API_KEY=' .env 2>/dev/null || { echo "missing OPENROUTER_API_KEY in $(pwd)/.env"; exit 1; }

~/miniconda3/envs/lerobot/bin/inspect-robots run \
    --instruction "put the pen in the bowl" --max-steps 1500 \
    --embodiment so101_real \
    --policy agent -P model=openai/gpt-6-astra -P prior_learnings=real_notes.md \
    --store-frames --log-dir logs_real "$@"

# Replay pages with video; open logs_real/html/index.html, or: inspect-robots view logs_real --open
~/miniconda3/envs/lerobot/bin/inspect-robots view logs_real > /dev/null
echo "replay: $(pwd)/logs_real/html/index.html"
