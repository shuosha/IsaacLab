#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <ckpt_path> [base]"
  exit 1
fi

ckpt_path="$1"
base="${2:-nn}"

# extract directory name
dir="$(basename "$ckpt_path")"

# extract task token after timestamp (lowercase)
task_lc="$(echo "$dir" | sed -E 's/^[0-9-]+_[0-9-]+_([^_]+).*/\1/')"

# map to CamelCase task names
case "$task_lc" in
  peginsert)
    task="PegInsert"
    ;;
  gearmesh)
    task="GearMesh"
    ;;
  nutthread)
    task="NutThread"
    ;;
  *)
    echo "Unknown task name: $task_lc"
    exit 1
    ;;
esac

python scripts/reinforcement_learning/rl_games/vis_qual_results.py \
  --task "Isaac-Factory-Xarm-${task}-Residual" \
  --num_envs 4 \
  --checkpoint "${ckpt_path}/nn/FactoryXarm.pth" \
  --enable_cameras \
  --output_name "$(basename "${ckpt_path}")" \
  --base "${base}" \
