task="$1"

python scripts/reinforcement_learning/rl_games/replay_${task}.py \
    --task Isaac-Factory-Xarm-${task}-Replay --enable_cameras