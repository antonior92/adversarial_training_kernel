import torch
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
import numpy as np


class RBFRandomFourierFeatures:
    '''
    compute R random fourier features for each sample in X
    '''

    def __init__(self, R, p, sigmasq=1, track_grads=False, device=None):
        '''
        R: number of random fourier features
        p: number of input features
        '''
        self.R = R
        self.p = p
        self.sigmasq = sigmasq
        self.track_grads = track_grads
        self.device = device

    def fit(self):
        self.W = torch.normal(
            mean=0.0,
            std=1.0 / torch.sqrt(torch.tensor(self.sigmasq)),
            size=(self.R, self.p),
        )  # R x p
        self.b = torch.rand(self.R) * 2 * torch.pi  # (R,)
        self.W.requires_grad = self.track_grads
        self.b.requires_grad = self.track_grads
        if self.device is not None:
            self.W = self.W.to(self.device)
            self.b = self.b.to(self.device)

    def transform(self, X):
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
        self.W.requires_grad = self.track_grads
        self.b.requires_grad = self.track_grads
        if self.device is not None:
            self.W = self.W.to(self.device)
            self.b = self.b.to(self.device)

        Z = X @ self.W.T + self.b  # n x R
        Z = torch.sqrt(torch.tensor(2 / self.R)) * torch.cos(Z)
        return Z


if __name__ == "__main__":
    # sample N points in the range [a, b); a <= b
    a, b = 3, 10
    N = 50
    X = (a - b) * np.random.rand(N, 1) + b
    # generate noisy observations of sin(x)
    mu, sigmasq = 0, 0.1
    y = np.sin(X).reshape(-1, 1) + np.random.normal(
        loc=mu, scale=np.sqrt(sigmasq), size=(N, 1)
    )

    # generate a smooth curve for sin(x) over the same range
    X_eval = np.linspace(a, b, 1000).reshape(-1, 1)
    y_eval = np.sin(X_eval)

    # fit a kernel ridge regression model to the data
    model = KernelRidge(kernel='rbf', gamma=1 / 2, alpha=0.1)
    model.fit(X, y)
    y_kr = model.predict(X_eval)

    R = 100000
    randomfeatures = RBFRandomFourierFeatures(R, X.shape[1])
    Z = randomfeatures.fit_transform(torch.from_numpy(X.astype(np.float32))).numpy()
    model = Ridge(alpha=0.1)
    model.fit(Z, y)
    y_r = model.predict(
        randomfeatures.transform(torch.from_numpy(X_eval.astype(np.float32)))
    )

    plt.plot(X, y, 'o')  # data
    plt.plot(X_eval, y_eval, label='sin(x)', color='red')  # truth
    plt.plot(X_eval, y_kr, label='kernel ridge', color='blue')  # kernel ridge fit
    plt.plot(X_eval, y_r, label=f'rff + ridge, R = {R}', color='green')  # rff fit
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
