# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv
from .factory_env_plain import FactoryEnvPlain
from .factory_env_replay import FactoryEnvReplay
from .factory_env_residual_no_base import FactoryEnvResidualNoBase
from .factory_env_residual import FactoryEnvResidual
from .factory_env_residual_teleop import FactoryEnvResidualTeleop
from .factory_env_cfg import FactoryTaskGearMeshCfg, FactoryTaskNutThreadCfg, FactoryTaskPegInsertCfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Factory-Xarm-Plain",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvPlain",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-GearMesh-Replay",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvReplay",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-PegInsert-Replay",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvReplay",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-NutThread-Replay",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvReplay",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-GearMesh-Residual-NoBase",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvResidualNoBase",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-PegInsert-Residual-NoBase",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvResidualNoBase",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-PegInsert-Residual",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvResidual",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-GearMesh-Residual",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvResidual",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-NutThread-Residual",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvResidual",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-GearMesh-Teleop",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvResidualTeleop",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-Xarm-PegInsert-Teleop",
    entry_point="isaaclab_tasks.direct.factory_xarm:FactoryEnvResidualTeleop",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)