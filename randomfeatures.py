import torch
import torch.nn as nn


class RBFRandomFourierFeatures(nn.Module):
    '''
    compute R random fourier features for each sample in X
    '''

    def __init__(self, R, p, sigmasq=1, track_grads=False):
        '''
        R: number of random fourier features
        p: number of input features
        '''
        super().__init__()
        self.R = R
        self.p = p
        self.sigmasq = sigmasq
        self.track_grads = track_grads

    def fit(self):
        self.W = torch.normal(
            mean=0.0,
            std=1.0 / torch.sqrt(torch.tensor(self.sigmasq)),
            size=(self.R, self.p),
        )  # R x p
        self.b = torch.rand(self.R) * 2 * torch.pi  # (R,)

        self.W = nn.Parameter(self.W, requires_grad=self.track_grads)
        self.b = nn.Parameter(self.b, requires_grad=self.track_grads)

    def transform(self, X):
        # N x p @ p x R + R x 1
        Z = X @ self.W.T + self.b
        Z = torch.sqrt(torch.tensor(2 / self.R)) * torch.cos(Z)
        return Z

    def fit_transform(self, X):
        '''
        note: W has R N(0, I * (1 / sigma_sq)) vectors on its rows
        '''
        self.W = torch.normal(
            mean=0.0,
            std=1.0 / torch.sqrt(torch.tensor(self.sigmasq)),
            size=(self.R, self.p),
        )  # R x p
        self.b = torch.rand(self.R) * 2 * torch.pi  # (R,)

        self.W = nn.Parameter(self.W, requires_grad=self.track_grads)
        self.b = nn.Parameter(self.b, requires_grad=self.track_grads)

        # N x p @ p x R + R x 1
        Z = X @ self.W.T + self.b  # n x R
        Z = torch.sqrt(torch.tensor(2 / self.R)) * torch.cos(Z)
        return Z
