time="$(date +%Y-%m-%d_%H-%M-%S)"

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-PegInsert-Residual --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name ${time}_peginsert_adm_1rew_knn --wandb-entity ss7050-columbia \
#     agent.params.config.full_experiment_name=${time}_peginsert_adm_1rew_knn --headless --base nn

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-GearMesh-Residual --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name ${time}_gearmesh_adm_3rew_bc_longer --wandb-entity ss7050-columbia \
#     agent.params.config.full_experiment_name=${time}_gearmesh_adm_3rew_bc_longer --headless --base bc

CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-Xarm-NutThread-Residual --num_envs 128 \
    --track --wandb-project-name FactoryXarm --wandb-name ${time}_nutthread_adm_3rew_nn_30deg_g240deg_40s_red10 --wandb-entity ss7050-columbia \
    agent.params.config.full_experiment_name=${time}_nutthread_adm_3rew_nn_30deg_g240deg_40s_red10 --headless --base nn