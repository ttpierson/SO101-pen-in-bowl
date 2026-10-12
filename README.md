# Eval logs: SO-101 pen-in-bowl

This branch holds the logs of every test run of the [`main`](../../tree/main) branch's task. The runs were made with
[Inspect Robots](https://github.com/robocurve/inspect-robots), and the HTML replay pages contain the videos.

| Run (UTC) | Log | Policy | Scenes | success_at_end | episode_length | Notes |
|---|---|---|---|---|---|---|
| 2026-10-11 22:04 | [so101-pen-in-bowl_9b518326](logs/html/so101-pen-in-bowl_9b518326.html) · [json](logs/so101-pen-in-bowl_9b518326.json) | so101_ik_pick_place | 1 | 0.00 | 330 | Early IK attempt; round pen rolled away before the grasp |
| 2026-10-12 05:40 | [so101-pen-in-bowl_10aba762](logs/html/so101-pen-in-bowl_10aba762.html) · [json](logs/so101-pen-in-bowl_10aba762.json) | so101_ik_pick_place | 5 | 0.60 | 264 | Pen made hexagonal (stops rolling) |
| 2026-10-12 05:42 | [so101-pen-in-bowl_dc740cbc](logs/html/so101-pen-in-bowl_dc740cbc.html) · [json](logs/so101-pen-in-bowl_dc740cbc.json) | so101_ik_pick_place | 5 | 0.80 | 255 | Release lowered to just above the rim |
| 2026-10-12 05:43 | [so101-pen-in-bowl_7f9050ce](logs/html/so101-pen-in-bowl_7f9050ce.html) · [json](logs/so101-pen-in-bowl_7f9050ce.json) | so101_ik_pick_place | 10 | 0.80 | 260 | Closed-loop place (aims the pen's centre at the bowl); **baseline result** |
| 2026-10-12 05:49 | [so101-pen-in-bowl_7f33e1f2](logs/html/so101-pen-in-bowl_7f33e1f2.html) · [json](logs/so101-pen-in-bowl_7f33e1f2.json) | so101_ik_pick_place | 2 | 1.00 | 244 | First run through the `inspect-robots run` CLI |
| 2026-10-12 05:51 | [so101-pen-in-bowl_c8f98c2c](logs/html/so101-pen-in-bowl_c8f98c2c.html) · [json](logs/so101-pen-in-bowl_c8f98c2c.json) | so101_ik_pick_place | 2 | 1.00 | 244 | CLI run with guardrails (joint bounds + delta limit) on |
| 2026-10-12 05:58 | [so101-pen-in-bowl_18bf2e61](logs/html/so101-pen-in-bowl_18bf2e61.html) · [json](logs/so101-pen-in-bowl_18bf2e61.json) | so101_ik_pick_place | 3 | 0.67 | 273 | `ik_eval.sh` test |
| 2026-10-12 06:00 | [so101-pen-in-bowl_05b17ca9](logs/html/so101-pen-in-bowl_05b17ca9.html) · [json](logs/so101-pen-in-bowl_05b17ca9.json) | so101_ik_pick_place | 1 | 1.00 | 241 | `ik_eval.sh --gui` live run (the README demo) |
| 2026-10-12 16:54 | [so101-pen-in-bowl_7e8920a0](logs/html/so101-pen-in-bowl_7e8920a0.html) · [json](logs/so101-pen-in-bowl_7e8920a0.json) | agent (`openai/gpt-6-astra`) | 1 | 0.00 | 183 | LLM policy; pushed the pen out of reach and gave up at step 183 |

`success_at_end` is the fraction of scenes where the pen ended at rest inside the bowl.
`episode_length` is the mean number of steps at 30 Hz; 330 means the step limit was reached.

## What's here

| Path | Contents |
|---|---|
| `logs/*.json` | Inspect Robots `EvalLog`s: config, per-scene scores, and per-trial step records and policy transcripts. |
| `logs/html/` | Self-contained HTML replays (video + transcript per trial). `index.html` lists every run. |
| `logs/actions/` | Per-step actions (JSONL), one folder per run. |
| `logs/transcripts/` | Policy transcripts written alongside the logs. |

These are not included:
- the raw per-step camera frames (`.npy`, about 3.4 GB); the HTML pages embed the videos instead
- the LLM run's raw API traffic

## Viewing

GitHub shows `.html` files as source code, not as pages. To view them:
- clone this branch and open `logs/html/index.html` in a browser:
  ```bash
  git clone -b eval-logs https://github.com/ttpierson/SO101-pen-in-bowl so101-logs
  ```
- or use the Inspect Robots CLI on a log: `inspect-robots inspect logs/<run>.json [--transcript]` or `inspect-robots view logs --open`.
