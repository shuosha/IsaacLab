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
                 pad: bool = True,
                 offline_base: bool = False,
                 ):
        self._device = torch.device(device)

        if min_horizon < 1:
            raise ValueError(f"min_horizon must be >= 1, got {min_horizon}")
        if max_horizon < min_horizon:
            raise ValueError(
                f"max_horizon ({max_horizon}) must be >= min_horizon ({min_horizon})"
            )

        self._min_horizon = int(min_horizon)
        self._max_horizon = int(max_horizon)

        # --- kNN stochasticity knobs ---
        self._knn_k = 10
        self._knn_tau = 0.3  # larger => more stochastic (flatter probs); start ~0.2-0.5

        data = np.load(path, allow_pickle=True).item()
        eps = sorted(data.keys())

        # convert to torch
        data = {
            ep: {
                k: torch.as_tensor(v, dtype=torch.float32, device=self._device)
                for k, v in ep_dict.items()
            }
            for ep, ep_dict in data.items()
        }

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

        # robot states
        self._obs_pos  = pad_last("obs.fingertip_pos", 3)
        self._obs_quat = pad_last("obs.fingertip_quat", 4)
        self._obs_grip = pad_last("obs.gripper", 1)
        self._obs_linvel = pad_last("obs.ee_linvel_fd", 3)
        self._obs_angvel = pad_last("obs.ee_angvel_fd", 3)

        # env states
        self._obs_rel_held = pad_last("obs.fingertip_pos_rel_held", 3)
        self._obs_rel_fixed = pad_last("obs.fingertip_pos_rel_fixed", 3)

        # actions
        self._act_pos  = pad_last("action.fingertip_pos", 3)
        self._act_quat = pad_last("action.fingertip_quat", 4)
        self._act_grip = pad_last("action.gripper", 1)
        self._mask     = (torch.arange(T, device=self._device)
                          .expand(len(eps), T) < lengths[:, None])

        self._num_envs = num_envs

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

        self._replay_mode = bool(offline_base)
        self._replay_ptr = torch.zeros(num_envs, dtype=torch.long, device=self._device)


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
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        self._q_ptr[env_ids] = 0
        self._q_len[env_ids] = 0
        self._replay_ptr[env_ids] = 0

    # --- core NN ---

    def _nn_indices(self, eidx, pos, quat=None, grip=None, verbose=False):
        # All tensors already on self._device
        obs_p = self._obs_pos[eidx]
        obs_q = self._obs_quat[eidx]
        obs_g = self._obs_grip[eidx]
        mask  = self._mask[eidx]

        pos_term  = 5 * torch.norm(obs_p - pos[:, None, :], dim=-1)
        ang_term  = torch.zeros_like(pos_term)
        grip_term = torch.zeros_like(pos_term)

        if quat is not None:
            ang_term = torch.rad2deg(
                quat_geodesic_angle(obs_q, quat[:, None, :])
            ) / 50
        if grip is not None:
            grip_term = 5 * (obs_g.squeeze(-1) - grip.view(-1, 1)).abs()

        dist = (pos_term + ang_term + grip_term).masked_fill(~mask, float("inf"))

        # valid lengths (per episode in batch)
        L = mask.long().sum(dim=1)  # (N,)

        # ---- stochastic kNN: sample among k nearest with p(i) ∝ exp(-d_i / tau) ----
        # If tau is tiny / non-positive, fall back to argmin (deterministic 1-NN).
        if self._knn_tau <= 0.0:
            t0 = dist.argmin(dim=1)
        else:
            k = min(self._knn_k, dist.shape[1])

            # get k smallest distances
            d_k, idx_k = torch.topk(dist, k=k, largest=False, dim=1)  # (N,k), (N,k)

            # Convert to logits; subtract max for stability
            logits = -d_k / self._knn_tau
            logits = logits - logits.max(dim=1, keepdim=True).values

            probs = torch.softmax(logits, dim=1)  # (N,k)

            # Sample one neighbor index per env
            j = torch.multinomial(probs, num_samples=1).squeeze(1)      # (N,)
            t0 = idx_k.gather(1, j[:, None]).squeeze(1)                 # (N,)

        if verbose:
            mmean = lambda x: x.masked_fill(~mask, torch.nan).nanmean().item()
            print(f"[NN contrib] pos_cm*10: {mmean(pos_term):.3f}, "
                  f"ang_deg/10: {mmean(ang_term):.3f}, "
                  f"grip_L1*2: {mmean(grip_term):.3f}")
        return t0, L

    @torch.no_grad()
    def get_actions(self,
                    eidx: torch.Tensor,
                    pos: torch.Tensor,
                    quat: torch.Tensor | None = None,
                    grip: torch.Tensor | None = None,
                    verbose: bool = False) -> torch.Tensor:
        # ... (device checks etc. unchanged)

        N = pos.shape[0]

        # -----------------------------
        # Replay mode: open-loop actions from t=0
        # -----------------------------
        if self._replay_mode:
            # gather per-env step indices, clamped to episode length-1
            L = self._lengths[eidx]  # (N,)
            t = torch.minimum(self._replay_ptr[:N], (L - 1).clamp(min=0))  # (N,)

            # gather actions at time t for each env
            a_pos  = self._act_pos[eidx, t, :]   # (N,3)
            a_quat = self._act_quat[eidx, t, :]  # (N,4)
            a_grip = self._act_grip[eidx, t, :]  # (N,1)
            out = torch.cat([a_pos, a_quat, a_grip], dim=-1)  # (N,8)

            # advance pointer only if not past the true episode end
            self._replay_ptr[:N] = torch.minimum(self._replay_ptr[:N] + 1, (L - 1).clamp(min=0))
            return out

        if self._queued is None:
            self._queued = torch.empty(
                (self._num_envs, self._max_horizon, 8), device=self._device
            )
            self._queued_idx = torch.empty(
                (self._num_envs, self._max_horizon),
                dtype=torch.long,
                device=self._device,
            )

        refill = (self._q_ptr >= self._q_len)   # (num_envs,)
        if refill.any():
            ids = refill.nonzero(as_tuple=False).squeeze(-1)  # (M,)

            t0, L = self._nn_indices(
                eidx[ids],
                pos[ids],
                None if quat is None else quat[ids],
                None if grip is None else grip[ids],
                verbose,
            )

            ar = torch.arange(self._max_horizon, device=self._device)   # (H_max,)
            idx = t0[:, None] + ar[None, :]                             # (M, H_max)
            idx = torch.minimum(idx, (L - 1).clamp(min=0)[:, None])

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

            # 🔹 Sample a fresh horizon for these envs
            H_env = torch.randint(
                low=self._min_horizon,
                high=self._max_horizon + 1,  # upper bound is exclusive
                size=(ids.numel(),),
                device=self._device,
            )

            self._horizon_env[ids] = H_env
            self._q_ptr[ids] = 0
            self._q_len[ids] = H_env

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

        obs_pos = self._obs_pos[eps_idx, :T, :].clone()    # (T, 3)
        obs_quat = self._obs_quat[eps_idx, :T, :].clone()  # (T, 4)
        act_pos = self._act_pos[eps_idx, :T, :].clone()    # (T, 3)
        act_quat = self._act_quat[eps_idx, :T, :].clone()  # (T, 4)

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        For each env, return the obs_pos that is closest to the given (pos, quat, grip).
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
        )

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
        For each env, return the obs (pos, quat, grip) closest to the query.
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
        )

        pos_nn   = self._obs_pos[eidx, t0, :]
        quat_nn  = self._obs_quat[eidx, t0, :]
        grip_nn  = self._obs_grip[eidx, t0, :]

        if return_idx:
            return pos_nn, quat_nn, grip_nn, t0
        return pos_nn, quat_nn, grip_nn
