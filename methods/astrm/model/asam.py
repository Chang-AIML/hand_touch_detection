""" Adaptive Sharpness-Aware Minimization (ASAM).

Reference: Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for
Scale-Invariant Learning of Deep Neural Networks", ICML 2021.
Implementation follows the authors' official ASAM minimizer.

Usage (per optimization step):
    # 1) loss at current weights
    loss = compute_loss(); loss.backward()
    minimizer.ascent_step()            # climb to adaptive worst-case neighbour
    # 2) loss at perturbed weights
    compute_loss().backward()
    minimizer.descent_step()           # restore weights, optimizer.step()

The paper uses rho = 2.0 with AdamW as the base optimizer.
"""

from collections import defaultdict
import torch


class ASAM:
    def __init__(self, optimizer, model, rho=2.0, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if "weight" in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)        # adaptive scaling: grad *= |w|
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.0e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if "weight" in n:
                p.grad.mul_(t_w)        # grad *= |w| again -> total |w|^2
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)                 # ascend to worst-case point
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])    # restore original weights
        self.optimizer.step()
        self.optimizer.zero_grad()
