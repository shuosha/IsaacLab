import torch

class LastKPoints:
    def __init__(self, num_envs: int, K: int = 5, device="cuda"):
        self.K = K
        self.num_envs = num_envs
        self.device = torch.device(device)

        # (K, N, 3)
        self.buf = torch.zeros((K, num_envs, 3), device=self.device)
        # whether this env has ever received a point
        self.initialized = torch.zeros((num_envs,), device=self.device, dtype=torch.bool)

    @torch.no_grad()
    def push(self, pos: torch.Tensor):
        """
        pos: (N,3) fingertip positions for all envs
        """
        # envs that are new (never initialized)
        new_envs = ~self.initialized

        if new_envs.any():
            # fill all K slots with the first value
            self.buf[:, new_envs] = pos[new_envs][None, :, :]
            self.initialized[new_envs] = True

        # for already-initialized envs: shift older points down
        old_envs = self.initialized
        if old_envs.any():
            self.buf[:, old_envs] = torch.roll(self.buf[:, old_envs], shifts=1, dims=0)
            self.buf[0, old_envs] = pos[old_envs]

    @torch.no_grad()
    def clear_envs(self, env_ids: torch.Tensor):
        """
        env_ids: (M,) long tensor
        """
        self.initialized[env_ids] = False

    def get_points(self):
        """
        Returns:
            points: (K*N, 3)
        """
        return self.buf.reshape(-1, 3)


