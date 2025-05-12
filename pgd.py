import torch
import torch.nn as nn


class PGD(nn.Module):
    '''
    arguments:
        - model (nn.Module): model to attack
        - loss_fn (callable): loss function
        - p (float): distance measure
        - adv_radius (float): maximum perturbation
        - step_size (float): step size
        - nsteps (int): number of steps

    shape:
        - X: (N, P) where N is the number of samples and P is the number of features
        - y: (N,)
        - output: (N, P)
    '''

    def __init__(
        self,
        model,
        loss_fn=nn.MSELoss(),
        p=torch.inf,
        adv_radius=8 / 255,
        step_size=2 / 255,
        nsteps=10,
    ):
        super().__init__()
        self.model, self.loss_fn = model, loss_fn
        self.adv_radius, self.step_size, self.nsteps = adv_radius, step_size, nsteps
        self.p = p
        assert p >= 1 or p == torch.inf, "p must be >= 1 or p == inf"
        self.q = 1 if p == torch.inf else (torch.inf if p == 1 else p / (p - 1))

    def forward(self, X, y, debug=False):
        X_adv = X.clone().requires_grad_(True)
        if debug:
            advs, losses = [], []

        for _ in range(self.nsteps):
            loss = self.loss_fn(self.model(X_adv), y)
            if debug:
                losses.append(loss.item())
                advs.append(X_adv.detach().clone())

            (grad,) = torch.autograd.grad(
                loss, X_adv, retain_graph=False, create_graph=False
            )
            grad = self.normalize_(grad)

            # flatten all but the batch dimension; check if norms are ok
            flat = grad.view(grad.shape[0], -1)
            norms = torch.norm(flat, self.p, dim=1)
            tol = 1e-1
            ok = (norms < tol) | (torch.abs(norms - 1.0) < tol)
            assert torch.all(ok), f"L_p norms not approx. 0/1: {norms}"

            # update X_adv with the gradient step
            X_adv = X_adv + self.step_size * grad

            # project X_adv onto the p-ball around X
            X_adv = self.project_(X, X_adv)

        if debug:
            advs.append(X_adv.detach().clone())
            losses.append(self.loss_fn(self.model(X_adv), y).item())

        init_loss = self.loss_fn(self.model(X), y)
        adv_loss = self.loss_fn(self.model(X_adv), y)
        assert (
            adv_loss >= init_loss
        ), f" adversarial loss ({adv_loss}) lower than initial loss ({init_loss})"

        if debug:
            return advs, losses

        return X_adv

    def normalize_(self, grad):
        if self.p == torch.inf:
            return grad.sign()
        if self.p == 1:  # put all mass in the direction of the maximum absolute value
            flat = grad.view(grad.shape[0], -1)
            # find index of maximum absolute value for each sample
            idx = flat.abs().argmax(dim=1)
            out = torch.zeros_like(grad)
            # scatter 1.0 at the index of maximum absolute value; everything else is 0
            out.view(grad.shape[0], -1).scatter_(1, idx.unsqueeze(1), 1.0)
            return out * grad.sign()
        # 1 < p < inf
        flat = grad.view(grad.shape[0], -1)
        power = 1.0 / (self.p - 1)
        norm_q = torch.norm(flat, p=self.q, dim=1, keepdim=True) + 1e-12
        # d = sign(g) * |g|^{1/(p-1)} / ||g||_{q}^{1/(p-1)}
        d = flat.sign() * flat.abs().pow(power) / norm_q.pow(power)
        return d.view_as(grad)

    def project_(self, X, X_adv):
        delta = X_adv - X
        if self.p == torch.inf:
            delta = delta.clamp(-self.adv_radius, self.adv_radius)
        elif self.p == 1:
            delta = self._proj_l1_ball(delta, self.adv_radius)
        else:  # 1 < p < inf
            flat = delta.view(delta.shape[0], -1)
            p_norm = torch.norm(flat, p=self.p, dim=1, keepdim=True)
            # compute scaling factor for each sample; factor is 1 if we are within the p-ball
            factor = (self.adv_radius / (p_norm + 1e-12)).clamp(max=1.0)
            # scale and reshape back to original shape
            delta = (flat * factor).view_as(delta)

        # check if norms are ok
        flat = delta.view(delta.shape[0], -1)
        if self.p == torch.inf:
            p_norm = torch.max(flat.abs(), dim=1, keepdim=True).values
        else:
            p_norm = torch.norm(flat, p=self.p, dim=1, keepdim=True)
        assert (
            torch.max(p_norm) <= self.adv_radius + 1e-6
        ), f"p-norm = {p_norm.max()} > adv_radius = {self.adv_radius} (tolerance = 1e-6)"

        return (X + delta).requires_grad_(True)

    @staticmethod
    def _proj_l1_ball(v, eps):
        '''figure 1 from Duchi et al. (2008)'''
        flat = v.view(v.size(0), -1)
        abs_v = flat.abs()
        l1_norm = abs_v.sum(dim=1, keepdim=True)

        # 1) check if we are inside the l1-ball
        inside_mask = l1_norm <= eps
        if inside_mask.all():
            return v  # nothing to do

        # 2) run Duchi only on outside rows
        work = flat[~inside_mask.squeeze()]  # (M, P)
        s = torch.sort(work, dim=1, descending=True).values
        cssv = s.cumsum(dim=1) - eps
        ind = torch.arange(1, work.size(1) + 1, device=v.device).float().unsqueeze(0)
        rho = (s - cssv / ind > 0).float().sum(dim=1, keepdim=True) - 1
        theta = cssv.gather(1, rho.long()) / (rho + 1)
        proj = torch.sign(work) * torch.clamp(work - theta, min=0.0)

        flat[~inside_mask.squeeze()] = proj
        return flat.view_as(v)
