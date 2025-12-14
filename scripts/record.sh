if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <task> <run_name>"
  echo " "
  exit 1
fi

task="$1"
ckpt_path="$2"

CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/record.py \
    --task Isaac-Factory-Xarm-${task}-Residual-Sparse-New  --num_envs 20 \
    --checkpoint ${ckpt_path}/nn/FactoryXarm.pth --enable_cameras #--headless