import torch, numpy as np

def quat_geodesic_angle(q1, q2, eps=1e-8):
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(eps)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(eps)
    dot = (q1 * q2).sum(-1).abs().clamp(-1 + eps, 1 - eps)
    return 2 * torch.arccos(dot)


class NearestNeighborBuffer:
    """Nearest-neighbor action retriever with per-env horizon queues."""

    def __init__(self, path: str, num_envs: int,
                 min_horizon: int = 1,
                 max_horizon: int = 15,
                 device: str | torch.device = "cpu",
                 pad: bool = True):
        self._device = torch.device(device)

        if min_horizon < 1:
            raise ValueError(f"min_horizon must be >= 1, got {min_horizon}")
        if max_horizon < min_horizon:
            raise ValueError(
                f"max_horizon ({max_horizon}) must be >= min_horizon ({min_horizon})"
            )

        self._min_horizon = int(min_horizon)
        self._max_horizon = int(max_horizon)

        flat = np.load(path, allow_pickle=True)
        flat = {k: torch.as_tensor(v, dtype=torch.float32, device=self._device)
                for k, v in flat.items()}

        eps = sorted({k.split("/", 1)[0] for k in flat})
        data = {e: {s.split("/", 1)[1]: flat[s] for s in flat if s.startswith(e)}
                for e in eps}

        lengths = torch.tensor([len(data[e]["obs.gripper"]) for e in eps],
                               device=self._device)
        self._lengths = lengths
        T = int(lengths.max())

        def pad_last(key, d):  # optional padding
            out = torch.zeros((len(eps), T, d), device=self._device)
            for i, e in enumerate(eps):
                x = data[e][key]
                if pad and len(x) < T:
                    x = torch.cat([x, x[-1:].repeat(T - len(x), 1)], dim=0)
                out[i, :len(x)] = x
            return out

        self._obs_pos  = pad_last("obs.eef_pos", 3)
        self._obs_quat = pad_last("obs.eef_quat", 4)
        self._obs_grip = pad_last("obs.gripper", 1)
        self._act_pos  = pad_last("action.eef_pos", 3)
        self._act_quat = pad_last("action.eef_quat", 4)
        self._act_grip = pad_last("action.gripper", 1)
        self._mask     = (torch.arange(T, device=self._device)
                          .expand(len(eps), T) < lengths[:, None])

        self._num_envs = num_envs

        # Minimum timestep to consider per env when doing NN
        self._nn_tmin = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        self._current_eidx = torch.zeros(num_envs, dtype=torch.long, device=self._device)

        # Max buffer capacity is max_horizon; actual per-env length is sampled later.
        self._horizon_env = torch.full((num_envs,),
                                       self._max_horizon,
                                       dtype=torch.long,
                                       device=self._device)

        self._queued = None          # (N, H_max, 8), on self._device
        self._queued_idx = None      # (N, H_max)
        self._q_ptr = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        self._q_len = torch.zeros(num_envs, dtype=torch.long, device=self._device)

        self._total_episodes = len(eps)
        self._max_episode_length = T
        print(f"Loaded {len(eps)} episodes; max length {T} on {self._device}. "
              f"Horizon in [{self._min_horizon}, {self._max_horizon}].")

    # --- public helpers ---

    def get_total_episodes(self):
        return self._total_episodes

    def get_max_episode_length(self):
        return self._max_episode_length

    def get_max_per_episode_length(self):
        return self._lengths

    def clear(self,
              env_ids: torch.Tensor | np.ndarray | list):
        """
        Clear queues for the given env ids.
        Does NOT change per-env horizons; horizons are re-sampled at refill time.
        Also resets the per-env NN time pointer.
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        self._q_ptr[env_ids] = 0
        self._q_len[env_ids] = 0
        self._nn_tmin[env_ids] = 0

    # --- core NN ---

    def _nn_indices(self, eidx, pos, quat=None, grip=None, env_ids=None, verbose=False):
        obs_p = self._obs_pos[eidx]
        obs_q = self._obs_quat[eidx]
        obs_g = self._obs_grip[eidx]
        mask  = self._mask[eidx]  # (N, T)

        ep_len = self._lengths[eidx]  # (N,)

        if env_ids is not None:
            # max starting index to keep horizon near the end
            # use ep_len - self._max_horizon so t0 + (H_max-1) stays in range
            max_tmin = (ep_len - self._max_horizon).clamp(min=0)

            # raw pointer for these envs
            tmin_raw = self._nn_tmin[env_ids]

            # effective pointer is clamped so we always have some valid timesteps
            tmin_eff = torch.minimum(tmin_raw, max_tmin)  # (N,)

            t = torch.arange(mask.shape[1], device=self._device).unsqueeze(0)  # (1, T)
            mask = mask & (t >= tmin_eff.unsqueeze(1))  # (N, T)

        pos_term  = 10 * torch.norm(obs_p - pos[:, None, :], dim=-1)
        ang_term  = torch.zeros_like(pos_term)
        grip_term = torch.zeros_like(pos_term)

        if quat is not None:
            ang_term = torch.rad2deg(
                quat_geodesic_angle(obs_q, quat[:, None, :])
            ) / 50
        if grip is not None:
            grip_term = 5 * (obs_g.squeeze(-1) - grip.view(-1, 1)).abs()

        dist = (pos_term + ang_term + grip_term).masked_fill(~mask, float("inf"))
        t0   = dist.argmin(dim=1)

        if verbose:
            mmean = lambda x: x.masked_fill(~mask, torch.nan).nanmean().item()
            print(f"[NN contrib] pos_cm/10: {mmean(pos_term):.3f}, "
                f"ang_deg/10: {mmean(ang_term):.3f}, "
                f"grip_L1*2: {mmean(grip_term):.3f}")

        return t0, ep_len


    @torch.no_grad()
    def get_actions(self,
                    eidx: torch.Tensor,
                    pos: torch.Tensor,
                    quat: torch.Tensor | None = None,
                    grip: torch.Tensor | None = None,
                    verbose: bool = False) -> torch.Tensor:
        # Assume we always query all envs in order
        N = pos.shape[0]
        assert N == self._num_envs, "get_actions expects a batch over all envs"

        if self._queued is None:
            self._queued = torch.empty(
                (self._num_envs, self._max_horizon, 8), device=self._device
            )
            self._queued_idx = torch.empty(
                (self._num_envs, self._max_horizon),
                dtype=torch.long,
                device=self._device,
            )
            
        self._current_eidx[:] = eidx 

        refill = (self._q_ptr >= self._q_len)   # (num_envs,)
        if refill.any():
            ids = refill.nonzero(as_tuple=False).squeeze(-1)  # (M,)

            t0, ep_len = self._nn_indices(
                eidx=eidx[ids],
                pos=pos[ids],
                quat=None if quat is None else quat[ids],
                grip=None if grip is None else grip[ids],
                env_ids=ids,
                verbose=verbose,
            )

            ar = torch.arange(self._max_horizon, device=self._device)   # (H_max,)
            idx = t0[:, None] + ar[None, :]                             # (M, H_max)

            # cap by episode end: ep_len-1 is the last valid global timestep
            idx = torch.minimum(idx, (ep_len - 1).clamp(min=0)[:, None])

            ap = self._act_pos[eidx[ids]]   # (M, T, 3)
            aq = self._act_quat[eidx[ids]]  # (M, T, 4)
            ag = self._act_grip[eidx[ids]]  # (M, T, 1)

            gi3 = idx[..., None].expand(-1, -1, 3)
            gi4 = idx[..., None].expand(-1, -1, 4)
            gi1 = idx[..., None].expand(-1, -1, 1)

            a = torch.cat([
                torch.gather(ap, 1, gi3),
                torch.gather(aq, 1, gi4),
                torch.gather(ag, 1, gi1),
            ], dim=-1)  # (M, H_max, 8)

            self._queued[ids] = a
            self._queued_idx[ids] = idx

            # Sample a fresh horizon for these envs
            H_env = torch.randint(
                low=self._min_horizon,
                high=self._max_horizon + 1,  # upper bound is exclusive
                size=(ids.numel(),),
                device=self._device,
            )

            self._horizon_env[ids] = H_env
            self._q_ptr[ids] = 0
            self._q_len[ids] = H_env

            # 🔹 update per-env minimum timestep based on END of sampled horizon
            M = ids.numel()
            m_idx = torch.arange(M, device=self._device)
            end_t = idx[m_idx, (H_env - 1).clamp(min=0)]  # (M,)

            self._nn_tmin[ids] = end_t + 1

        env_ids = torch.arange(N, device=self._device)
        step_idx = torch.minimum(self._q_ptr, (self._q_len - 1).clamp(min=0))
        out = self._queued[env_ids, step_idx, :]  # (N, 8)

        has_data = (self._q_ptr < self._q_len)
        self._q_ptr[has_data] += 1

        return out


    def get_episode_traj(self, eps_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the full (obs_pos, obs_quat, act_pos, act_quat) trajectory for a given episode index.
        """
        if not (0 <= eps_idx < self._total_episodes):
            raise IndexError(
                f"Episode index {eps_idx} out of range [0, {self._total_episodes - 1}]"
            )

        T = int(self._lengths[eps_idx].item())
        obs_pos  = self._obs_pos[eps_idx, :T, :]   # (T, 3)
        obs_quat = self._obs_quat[eps_idx, :T, :]  # (T, 4)
        act_pos  = self._act_pos[eps_idx, :T, :]   # (T, 3)
        act_quat = self._act_quat[eps_idx, :T, :]  # (T, 4)
        return obs_pos, obs_quat, act_pos, act_quat

    @torch.no_grad()
    def get_closest_obs_pos(
        self,
        eidx: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor | None = None,
        grip: torch.Tensor | None = None,
        verbose: bool = False,
        return_idx: bool = False,
    ):
        """
        For each env, return the obs_pos closest to (pos, quat, grip) over the full episode.
        Does NOT use the per-env time pointer.
        """
        if pos.device != self._device:
            raise ValueError(f"pos.device={pos.device} but buffer.device={self._device}")
        if quat is not None and quat.device != self._device:
            raise ValueError("quat must be on the same device as the buffer")
        if grip is not None and grip.device != self._device:
            raise ValueError("grip must be on the same device as the buffer")

        t0, _ = self._nn_indices(
            eidx=eidx,
            pos=pos,
            quat=quat,
            grip=grip,
            verbose=verbose,
        )  # t0: (N,)

        obs_pos_nn = self._obs_pos[eidx, t0, :]  # (N, 3)

        if return_idx:
            return obs_pos_nn, t0
        return obs_pos_nn

    @torch.no_grad()
    def get_closest_obs(
        self,
        eidx: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor | None = None,
        grip: torch.Tensor | None = None,
        verbose: bool = False,
        return_idx: bool = False,
    ):
        """
        For each env, return the obs (pos, quat, grip) closest to (pos, quat, grip) over the full episode.
        Does NOT use the per-env time pointer.
        """
        if pos.device != self._device:
            raise ValueError(f"pos.device={pos.device} but buffer.device={self._device}")
        if quat is not None and quat.device != self._device:
            raise ValueError("quat must be on the same device as the buffer")
        if grip is not None and grip.device != self._device:
            raise ValueError("grip must be on the same device as the buffer")

        t0, _ = self._nn_indices(
            eidx=eidx,
            pos=pos,
            quat=quat,
            grip=grip,
            verbose=verbose,
        )  # t0: (N,)

        pos_nn   = self._obs_pos[eidx, t0, :]      # (N, 3)
        quat_nn  = self._obs_quat[eidx, t0, :]     # (N, 4)
        grip_nn  = self._obs_grip[eidx, t0, :]     # (N, 1)

        if return_idx:
            return pos_nn, quat_nn, grip_nn, t0
        return pos_nn, quat_nn, grip_nn
    
    def is_waiting(self, env_ids):
        """
        Return a boolean tensor of shape (len(env_ids),)
        indicating whether each env has reached the end of its episode.
        """
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

        # Per-env episode index
        e = self._current_eidx[env_ids]               # (N,)
        ep_len = self._lengths[e]                     # (N,)

        # Latest t0 that can still fit a horizon
        max_valid_t0 = ep_len - self._max_horizon - 1  # (N,)

        # Check
        return self._nn_tmin[env_ids] >= max_valid_t0
