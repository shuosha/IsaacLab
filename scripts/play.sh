if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <task> <run_name>"
  echo " "
  exit 1
fi

task="$1"
ckpt_path="$2"

CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Factory-Xarm-${task}-Residual --num_envs 1 \
    --checkpoint ${ckpt_path}/nn/FactoryXarm.pth