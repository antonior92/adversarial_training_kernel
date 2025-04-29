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
        - a (float): lower bound of the input
        - b (float): upper bound of the input
        - eps (float): maximum perturbation
        - alpha (float): step size
        - steps (int): number of steps
        - random_start (bool): using random initialization of delta

    shape:
        - X: (N, C, H, W) where N = batch size, C = number of channels, H = height and W = width
        - y: (N,)
        - output: (N, C, H, W)
    '''

    def __init__(
        self,
        model,
        loss_fn,
        a,
        b,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        random_start=False,
    ):
        super(PGD, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.a = a
        self.b = b
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, X, y):
        X_adv = X.clone()
        X_adv.requires_grad = True

        if self.random_start:
            # starting at a uniformly random point
            X_adv = X_adv + torch.empty_like(X_adv).uniform_(-self.eps, self.eps)
            X_adv = torch.clamp(X_adv, min=self.a, max=self.b)

        for _ in range(self.steps):
            loss = self.loss_fn(self.model(X_adv), y)
            grad = torch.autograd.grad(
                loss, X_adv, retain_graph=False, create_graph=False
            )[0]

            # update X_adv with the gradient step
            X_adv = X_adv + self.alpha * grad.sign()

            # project X_adv to the epsilon-ball around X
            X_adv = torch.clamp(X_adv, X - self.eps, X + self.eps)

            # ensure X_adv stays within the valid input range [a, b]
            X_adv = torch.clamp(X_adv, min=self.a, max=self.b)

        return X_adv
