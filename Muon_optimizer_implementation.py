import torch
from torch.optim import Optimizer


class MuonOptimizer(Optimizer):
    def __init__(self, params, lr=0.001, ortho_eps=1e-6):
        defaults = dict(lr=lr, ortho_eps=ortho_eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Orthogonal gradient projection
                if 'projection' not in state:
                    state['projection'] = torch.eye(grad.size(0), device=grad.device)

                # Newton-Schultz iterative approximation of the inverse matrix
                with torch.no_grad():
                    u, s, v = torch.svd(grad)
                    ortho_grad = u @ v.T @ grad

                # Mixed-precision updates
                if p.dtype == torch.float32:
                    p.data -= group['lr'] * ortho_grad
                else:  # BF16/FP16
                    fp32_param = p.float()
                    fp32_param -= group['lr'] * ortho_grad.to(torch.float32)
                    p.data = fp32_param.to(p.dtype)