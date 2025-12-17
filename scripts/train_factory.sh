time="$(date +%Y-%m-%d_%H-%M-%S)"

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-PegInsert-Residual --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name ${time}_peginsert_only_task_rew --wandb-entity ss7050-columbia \
#     agent.params.config.full_experiment_name=${time}_peginsert_only_task_rew --headless

CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-Xarm-GearMesh-Residual --num_envs 128 \
    --track --wandb-project-name FactoryXarm --wandb-name ${time}_gearmesh_adm_clip_grip_20s --wandb-entity ss7050-columbia \
    agent.params.config.full_experiment_name=${time}_gearmesh_adm_clip_grip_20s --headless

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-NutThread-Residual --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name ${time}_nutthread_perf_grip_rot180 --wandb-entity ss7050-columbia \
#     agent.params.config.full_experiment_name=${time}_nutthread_perf_grip_rot180 --headless