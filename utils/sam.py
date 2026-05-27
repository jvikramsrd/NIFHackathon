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
        """Ascent step: add rho-scaled worst-case gradient perturbation.

        Implementation note: the original per-parameter loop issued one CUDA
        kernel per tensor. On a 64M-parameter SegFormer that's >500 kernel
        launches per first_step. We batch the same math through
        ``torch._foreach_*`` ops (one kernel per op total), and we keep a
        single reusable buffer of saved weights instead of allocating a
        fresh ``clone()`` per param per step.
        """
        grad_norm = _grad_norm(self.param_groups)
        # Clamp scale so the perturbation cannot explode when grad_norm ≈ 0
        # (e.g. first step of a frozen backbone or after gradient zeroing).
        scale = (self.defaults["rho"] / (grad_norm + 1e-12)).clamp(max=0.2)
        scale_f = float(scale.detach().item())

        for group in self.param_groups:
            params, grads = [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad)
            if not params:
                continue

            # Materialise (or refresh) the persistent old-weight buffers.
            old_list = []
            for p in params:
                buf = self.state[p].get("old_p")
                if buf is None or buf.shape != p.shape or buf.dtype != p.dtype:
                    buf = torch.empty_like(p, memory_format=torch.preserve_format)
                    self.state[p]["old_p"] = buf
                old_list.append(buf)
            try:
                torch._foreach_copy_(old_list, [p.data for p in params])
            except (AttributeError, RuntimeError):
                for b, p in zip(old_list, params):
                    b.copy_(p.data)

            # Build perturbation tensors in batched form, then add in place.
            if group["adaptive"]:
                try:
                    e_w = torch._foreach_mul(grads, [p.pow(2) for p in params])
                    torch._foreach_mul_(e_w, scale_f)
                except (AttributeError, RuntimeError):
                    e_w = [(p.pow(2) * g) * scale_f for p, g in zip(params, grads)]
            else:
                try:
                    e_w = torch._foreach_mul(grads, scale_f)
                except (AttributeError, RuntimeError):
                    e_w = [g * scale_f for g in grads]

            try:
                torch._foreach_add_([p.data for p in params], e_w)
            except (AttributeError, RuntimeError):
                for p, e in zip(params, e_w):
                    p.data.add_(e)

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Descent step: restore original weights and take the real gradient step.

        Batched restore via ``torch._foreach_copy_`` collapses N per-param
        kernels into 1.
        """
        for group in self.param_groups:
            params, olds = [], []
            for p in group["params"]:
                buf = self.state.get(p, {}).get("old_p")
                if buf is None:
                    continue
                params.append(p)
                olds.append(buf)
            if not params:
                continue
            try:
                torch._foreach_copy_([p.data for p in params], olds)
            except (AttributeError, RuntimeError):
                for p, b in zip(params, olds):
                    p.data.copy_(b)

        self._base.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

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
    """Global L2 norm of all gradients.

    Batched through ``torch._foreach_norm`` + a single stack reduction so the
    cost is one launch instead of one per parameter.
    """
    grads = [p.grad for group in param_groups for p in group["params"] if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    try:
        # _foreach_norm returns scalar tensors on the same device as their input
        # gradient, so when all params live on one GPU the .to(device) round-trip
        # is a no-op kernel launch per scalar. Drop it on the fast path; fall
        # back to the explicit copy only when the user actually mixed devices.
        norms = torch._foreach_norm(grads, 2)
        if all(n.device == device for n in norms):
            return torch.linalg.vector_norm(torch.stack(norms))
        return torch.linalg.vector_norm(torch.stack([n.to(device) for n in norms]))
    except (AttributeError, RuntimeError):
        sq = torch.zeros((), device=device)
        for g in grads:
            sq = sq + g.float().pow(2).sum()
        return sq.sqrt()