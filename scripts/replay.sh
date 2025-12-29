task="$1"

python scripts/reinforcement_learning/rl_games/replay.py \
    --task Isaac-Factory-Xarm-${task}-Residual --enable_cameras