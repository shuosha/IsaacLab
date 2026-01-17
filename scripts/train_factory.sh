time="$(date +%Y-%m-%d_%H-%M-%S)"

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-PegInsert-Residual --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name ${time}_peginsert_constr_nn_term  --wandb-entity ss7050-columbia \
#     agent.params.config.full_experiment_name=${time}_peginsert_constr_nn_term  --headless --base nn

CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Factory-Xarm-GearMesh-Residual --num_envs 128 \
    --track --wandb-project-name FactoryXarm --wandb-name ${time}_gearmesh_constr_nn_term --wandb-entity ss7050-columbia \
    agent.params.config.full_experiment_name=${time}_gearmesh_constr_nn_term --headless --base nn

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-NutThread-Residual --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name ${time}_nutthread_constr_nn_term  --wandb-entity ss7050-columbia \
#     agent.params.config.full_experiment_name=${time}_nutthread_constr_nn_term  --headless --base nn