#!/usr/bin/env bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <task> <ckpt_path> [num_envs]"
  echo " "
  exit 1
fi

task="$1"
ckpt_path="$2"
num_envs="${3:-1}"   # default to 1 if not provided

CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/play_dexgen.py \
    --task Isaac-Factory-Xarm-${task}-DexGen \
    --num_envs ${num_envs} \
    --policy_path ${ckpt_path} --headless
