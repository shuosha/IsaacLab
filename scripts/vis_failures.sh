#!/usr/bin/env bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <task> <run_name> [num_envs]"
  echo " "
  exit 1
fi

task="$1"
ckpt_path="$2"
python scripts/reinforcement_learning/rl_games/visualize_failure_cases.py \
    --task Isaac-Factory-Xarm-${task}-Residual \
    --num_envs 20 \
    --checkpoint "${ckpt_path}/nn/FactoryXarm.pth" \
    --enable_cameras --output_name "$(basename "${ckpt_path}")"