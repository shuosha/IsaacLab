export LD_PRELOAD=/home/shuo/projects/IsaacLab/env_isaac/lib/python3.11/site-packages/scikit_learn.libs/libgomp-e985bcbb.so.1.0.0:${LD_PRELOAD}

time="$(date +%Y-%m-%d_%H-%M-%S)"
base="nn" # choose from nn, bc, noisy_nn

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-PegInsert-Residual --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name ${time}_peginsert_train_${base}_v14.1   --wandb-entity ss7050-columbia \
#     agent.params.config.full_experiment_name=${time}_peginsert_train_${base}_v14.1 --headless --base ${base}

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-GearMesh-Residual --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name ${time}_gearmesh_train_${base}_v14.1 --wandb-entity ss7050-columbia \
#     agent.params.config.full_experiment_name=${time}_gearmesh_train_${base}_v14.1 --headless --base ${base}

# CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/rl_games/train.py \
#     --task Isaac-Factory-Xarm-NutThread-Residual --num_envs 128 \
#     --track --wandb-project-name FactoryXarm --wandb-name ${time}_nutthread_train_${base}_v14.1   --wandb-entity ss7050-columbia \
#     agent.params.config.full_experiment_name=${time}_nutthread_train_${base}_v14.1 --headless --base ${base}