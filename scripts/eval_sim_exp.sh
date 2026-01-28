num_envs=100
eval_eps=100
# log_path=logs/plot_data/sim_exp_residual_v2.json

# for task in GearMesh PegInsert NutThread; do
#     for base in noisy_nn nn bc_teleop bc_expert laggy_bc_expert noisy_bc_expert; do
#         for train_base in noisy_nn bc_teleop; do
#             ckpt_path="rrl_data_v3/models/residual/${task}_${train_base}"
#             python scripts/reinforcement_learning/rl_games/play.py \
#                 --task Isaac-Factory-Xarm-${task}-Residual \
#                 --num_envs ${num_envs} \
#                 --checkpoint ${ckpt_path}/nn/FactoryXarm.pth --headless \
#                 --base ${base} \
#                 --eval_episodes ${eval_eps} \
#                 --log_path ${log_path}
#             echo "Evaluated checkpoint ${ckpt_path} with base model ${base}"
#         done
#     done
# done

log_path=logs/plot_data/sim_exp_base_only_v2.json
for task in GearMesh PegInsert NutThread; do
    for base in noisy_nn nn bc_teleop bc_expert laggy_bc_expert noisy_bc_expert; do
        ckpt_path="rrl_data_v3/models/residual/${task}_noisy_nn"
        python scripts/reinforcement_learning/rl_games/play.py \
            --task Isaac-Factory-Xarm-${task}-Residual \
            --num_envs ${num_envs} \
            --checkpoint ${ckpt_path}/nn/FactoryXarm.pth --headless \
            --base ${base} --base_only \
            --eval_episodes ${eval_eps} \
            --log_path ${log_path}
        echo "Evaluated checkpoint ${ckpt_path} with base model ${base}"
    done
done

# log_path=logs/plot_data/sim_exp_diffusion_v2.json

# for task in GearMesh PegInsert NutThread; do
#     for base in noisy_nn nn bc_teleop bc_expert laggy_bc_expert noisy_bc_expert; do
#         for train_base in bc_teleop bc_expert; do
#             ckpt_path="rrl_data_v3/models/${train_base}/${task}_${train_base}"
#             python scripts/reinforcement_learning/rl_games/play_dexgen.py \
#                 --task Isaac-Factory-Xarm-${task}-DexGen \
#                 --num_envs ${num_envs} \
#                 --policy_path ${ckpt_path} --headless \
#                 --base ${base} \
#                 --eval_episodes ${eval_eps} \
#                 --log_path ${log_path}
#             echo "Evaluated checkpoint ${ckpt_path} with base model ${base}"
#         done
#     done
# done