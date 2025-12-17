#!/usr/bin/env bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <task> <run_name> [num_envs]"
  echo " "
  exit 1
fi

task="$1"
ckpt_path="$2"
output_dir="$3"
num_envs="${4:-1}"   # default to 1 if not provided
python scripts/reinforcement_learning/rl_games/visualize_failure_cases.py \
    --task Isaac-Factory-Xarm-${task}-Residual \
    --num_envs "${num_envs}" \
    --checkpoint "${ckpt_path}/nn/FactoryXarm.pth" \
    --enable_cameras --output_dir "${output_dir}"