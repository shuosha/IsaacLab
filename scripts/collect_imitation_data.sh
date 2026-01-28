#!/usr/bin/env bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <task> <run_name> [num_envs]"
  echo " "
  exit 1
fi

task="$1"
ckpt_path="$2"
output_dir="$3"
base="${4:-noisy_nn}" # default to nn if not provided
num_eps=2000
num_envs=100   # default to 1 if not provided
python scripts/reinforcement_learning/rl_games/collect_imitation_data.py \
    --task Isaac-Factory-Xarm-${task}-Residual \
    --num_envs "${num_envs}" \
    --checkpoint "${ckpt_path}/nn/FactoryXarm.pth" \
    --output_dir "${output_dir}" --num_episodes "${num_eps}" \
    --no_images --headless --base ${base}