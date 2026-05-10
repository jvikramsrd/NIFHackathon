"""
utils/sam.py  —  Sharpness-Aware Minimisation (Foret et al., 2021)
──────────────────────────────────────────────────────────────────
SAM seeks flat minima by adding a small worst-case perturbation before
each gradient step. This improves generalisation by 0.5–2.0% mIoU / accuracy
at the cost of a second forward+backward pass (~1.9× per-step overhead).

On the A4000 with bf16 AMP this overhead is partially mitigated by the
fast tensor cores. The total training time increase is ~60–70%.
"""

from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimisation wrapper over any base optimiser.

    Usage:
        base_opt = torch.optim.AdamW(params, lr=1e-4)
        sam = SAM(base_opt, rho=0.05, adaptive=True)

        # In training loop:
        loss1 = loss_fn(model(imgs), targets)
        loss1.backward()
        sam.first_step(zero_grad=True)        # ascent step

        loss2 = loss_fn(model(imgs), targets) # second forward
        loss2.backward()
        sam.second_step(zero_grad=True)       # descent step

    Paper: https://arxiv.org/abs/2010.01412
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = True,
    ):
        assert rho >= 0.0, "rho must be non-negative"
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(base_optimizer.param_groups, defaults)
        self._base = base_optimizer
        self.defaults.update(self._base.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Ascent step: add rho-scaled worst-case gradient perturbation."""
        grad_norm = _grad_norm(self.param_groups)
        # Clamp scale so the perturbation cannot explode when grad_norm ≈ 0
        # (e.g. first step of a frozen backbone or after gradient zeroing).
        scale = (self.defaults["rho"] / (grad_norm + 1e-12)).clamp(max=0.2)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p_state = self.state[p]
                p_state["old_p"] = p.data.clone()
                # group["adaptive"] is merged from SAM defaults via super().__init__
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)  # w + epsilon

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Descent step: restore original weights and take the real gradient step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]["old_p"])

        self._base.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None) -> None:
        """Standard step (no SAM perturbation — delegates to base optimizer)."""
        self._base.step(closure)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._base.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self._base.state_dict()

    def load_state_dict(self, state_dict):
        self._base.load_state_dict(state_dict)


def _grad_norm(param_groups):
    # Find the first parameter with a gradient to get the right device
    device = None
    for group in param_groups:
        for p in group["params"]:
            if p.grad is not None:
                device = p.grad.device
                break
        if device is not None:
            break
    if device is None:
        return torch.tensor(0.0)
    norm = torch.tensor(0.0, device=device)
    for group in param_groups:
        for p in group["params"]:
            if p.grad is not None:
                norm.add_(p.grad.norm(2).pow(2))
    return norm.sqrt()