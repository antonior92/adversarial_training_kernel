import torch
import torch.nn as nn


class PGD(nn.Module):
    '''
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    distance measure: Linf

    arguments:
        - model (nn.Module): model to attack
        - loss_fn (callable): loss function
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
        loss_fn,
        adv_radius=8 / 255,
        step_size=2 / 255,
        nsteps=10,
    ):
        super(PGD, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.adv_radius = adv_radius
        self.step_size = step_size
        self.nsteps = nsteps

    def forward(self, X, y):
        X_adv = X.clone()
        X_adv.requires_grad = True

        for _ in range(self.nsteps):
            loss = self.loss_fn(self.model(X_adv), y)
            grad = torch.autograd.grad(
                loss, X_adv, retain_graph=False, create_graph=False
            )[0]

            # update X_adv with the gradient step
            X_adv = X_adv + self.step_size * grad.sign()

            # project X_adv to the ball around X
            X_adv = torch.clamp(X_adv, X - self.adv_radius, X + self.adv_radius)

        init_loss = self.loss_fn(self.model(X), y)
        adv_loss = self.loss_fn(self.model(X_adv), y)
        assert (
            adv_loss >= init_loss
        ), f" adversarial loss ({adv_loss}) lower than initial loss ({init_loss})"

        return X_adv
