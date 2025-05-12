from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.linear_model._ridge import _ridge_regression, Ridge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from pgd import PGD
from randomfeatures import RBFRandomFourierFeatures
import torch
import torch.nn as nn


def eta_trick(values, eps=1e-12):
    """Implement eta trick."""
    values = np.atleast_2d(values)
    # for exact solution use eps=0 so that np.abs(values)
    # this might lead to numerical instabilities tho
    abs_values = np.sqrt(values**2 + eps)
    sum_of_values = np.sum(abs_values, axis=0)
    c = sum_of_values / (abs_values)
    return c


def get_norm(krr, K, eps=1e-15):
    norm_squared = (K @ krr.dual_coef_) @ krr.dual_coef_
    # if norm_squared > 0:
    #    return np.sqrt(norm_squared)
    # else:
    return np.sqrt(norm_squared + eps * krr.dual_coef_ @ krr.dual_coef_)


def get_update_size(krr, krr_old):
    return np.linalg.norm(krr.dual_coef_ - krr_old.dual_coef_, ord=2)


def kernel_adversarial_training(
    X,
    y,
    adv_radius=None,
    verbose=True,
    utol=1e-12,
    max_iter=100,
    kernel='linear',
    kernel_params=None,
):
    n_train, n_features = X.shape

    w_samples = 1 / n_train * np.ones(n_train)
    regul_correction = 1

    if kernel_params is None:
        kernel_params = {}
    K = pairwise_kernels(X, metric=kernel, **kernel_params)
    if adv_radius is None:
        adv_radius = 0.4 * np.sqrt(np.trace(K)) / n_train
    for i in range(max_iter):
        # ------- 1. Solve reweighted ridge regression ------
        reg = regul_correction * adv_radius**2
        krr = KernelRidge(alpha=reg, kernel='precomputed')
        krr.fit(K, y, sample_weight=w_samples)

        # ------- 2. Perform eta trick  -------
        abs_error = np.abs(krr.predict(K) - y)
        param_norm = get_norm(krr, K)  ## UPDATE!!!!
        M = np.abs([abs_error, adv_radius * param_norm * np.ones(n_train)])
        c = eta_trick(M)
        regul_correction = np.sum(c[1])
        w_samples = c[0]

        # Fix regularization parameter
        regul_correction = regul_correction / np.sum(w_samples)
        w_samples = w_samples / np.sum(w_samples)

        # -------  Generate report ------
        if i > 1:
            update_size = get_update_size(krr, krr_old)
        else:
            update_size = utol + 100
        if verbose == True:
            mean_regul = (
                np.mean(w_samples) * regul_correction
                if w_samples is not None
                else regul_correction
            )
            print(
                f'Iteration {i} | update size: {update_size:4.3e} | regul: {reg:4.3e} | '
                f'param norm: {param_norm:4.3e} | mean abs error: {np.mean(abs_error):4.3e} | '
                f'loss: {np.mean((abs_error + adv_radius * param_norm) ** 2)}'
            )
        info = {
            'w_samples': w_samples,
            'regul_correction': regul_correction,
            'update_size': update_size,
            'n_iter': i,
        }
        krr_old = krr  # update parameters

        # ------- Termination criterion -------
        if update_size < utol:
            break

    # modifications so one can safely use krr.predict()
    def my_get_kernel(Z, W):
        return pairwise_kernels(Z, W, metric=kernel, **kernel_params)

    krr._get_kernel = my_get_kernel
    krr.X_fit_ = X
    krr.n_features_in_ = X.shape[1]

    return krr


class AdvKernelTrain(BaseEstimator, RegressorMixin):
    def __init__(
        self, kernel='rbf', adv_radius=None, verbose=False, gamma=None, **kernel_params
    ):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.adv_radius = adv_radius
        self.gamma = gamma
        self.model_ = None

    def fit(self, X, y):
        if self.gamma is None:
            kernel_params = self.kernel_params
        else:
            kernel_params = {**self.kernel_params, 'gamma': self.gamma}
        self.model_ = kernel_adversarial_training(
            X,
            y,
            verbose=self.verbose,
            adv_radius=self.adv_radius,
            kernel=self.kernel,
            kernel_params=kernel_params,
        )
        return self

    def predict(self, X):
        return self.model_.predict(X)


def mkl_adversarial_training(
    X,
    y,
    adv_radius=None,
    verbose=True,
    utol=1e-12,
    max_iter=100,
    kernel=[
        'linear',
    ],
    kernel_params=None,
):
    n_train, n_features = X.shape

    w_samples = 1 / n_train * np.ones(n_train)
    regul_correction = 1

    n_kernels = len(kernel)
    w_params = 1 / n_kernels * np.ones(n_kernels)
    if kernel_params is None:
        kernel_params = n_kernels * [{}]

    if adv_radius is None:
        adv_radius = 0.4 / np.sqrt(n_train)

    kernel_list = []
    for kernel_i, kparams_i in zip(kernel, kernel_params):
        kernel_list.append(pairwise_kernels(X, metric=kernel_i, **kparams_i))
    for i in range(max_iter):
        # ------- 1. Solve reweighted ridge regression ------
        reg = regul_correction * adv_radius**2
        K = sum(wi * kernel_i for (wi, kernel_i) in zip(w_params, kernel_list))
        krr = KernelRidge(alpha=reg, kernel='precomputed')
        krr.fit(K, y, sample_weight=w_samples)

        # ------- 2. Perform eta trick  -------
        abs_error = np.abs(krr.predict(K) - y)
        param_norm = get_norm(krr, K)
        params_norms = [get_norm(krr, kernel_i) for kernel_i in kernel_list]
        M = np.abs(
            [abs_error, *[adv_radius * p * np.ones(n_train) for p in params_norms]]
        )
        c = eta_trick(M)
        regul_correction = np.sum(c[1])
        w_samples = c[0]
        w_params = np.sum(c[1:], axis=1)
        # Fix regularization parameter
        regul_correction = np.max(w_params)
        w_params = w_params / np.max(w_params)

        # Fix regularization parameter
        regul_correction = regul_correction / np.sum(w_samples)
        w_samples = w_samples / np.sum(w_samples)

        # -------  Generate report ------
        if i > 1:
            update_size = get_update_size(krr, krr_old)
        else:
            update_size = utol + 100
        if verbose == True:
            mean_regul = (
                np.mean(w_samples) * regul_correction
                if w_samples is not None
                else regul_correction
            )
            print(
                f'Iteration {i} | update size: {update_size:4.3e} | regul: {reg:4.3e} | '
                f'param norm: {param_norm:4.3e} | mean abs error: {np.mean(abs_error):4.3e} | '
                f'loss: {np.mean((abs_error + adv_radius * param_norm) ** 2)}'
            )
        info = {
            'w_samples': w_samples,
            'regul_correction': regul_correction,
            'update_size': update_size,
            'n_iter': i,
        }
        krr_old = krr  # update parameters

        # ------- Termination criterion -------
        if update_size < utol:
            break

    def my_get_kernel(X, Y):
        K = sum(
            wi * pairwise_kernels(X, Y, metric=kernel_i, **kparams_i)
            for wi, kernel_i, kparams_i in zip(w_params, kernel, kernel_params)
        )
        return K

    krr._get_kernel = my_get_kernel
    krr.X_fit_ = X
    krr.n_features_in_ = X.shape[1]

    return krr


# Define sklearn like wrapper
class AdvMultipleKernelTrain(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=[
            'linear',
        ],
        adv_radius=None,
        verbose=False,
        kernel_params=None,
    ):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.adv_radius = adv_radius
        self.model_ = None

    def fit(self, X, y):
        self.model_ = mkl_adversarial_training(
            X,
            y,
            verbose=self.verbose,
            adv_radius=self.adv_radius,
            kernel=self.kernel,
            kernel_params=self.kernel_params,
        )
        return self

    def predict(self, X):
        return self.model_.predict(X)


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
        )

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        return self.model(X).view(-1)


class LinearAdvFourierFeatures(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        R,
        adv_radius,
        p=torch.inf,
        nsteps=10,
        step_size=2 / 255,
        nepochs=100,
        lr=1e-3,
        verbose=False,
    ):
        self.R = R
        self.adv_radius = adv_radius
        self.nsteps = nsteps
        self.step_size = step_size
        self.nepochs = nepochs
        self.lr = lr
        self.verbose = verbose
        self.loss_fn = nn.MSELoss()
        self.lin_net = LinearNet(input_dim=R, output_dim=1)
        self.p = p

    def fit(self, X, y):
        n_train, n_features = X.shape

        self.randomfeatures = RBFRandomFourierFeatures(self.R, n_features)
        Z = self.randomfeatures.fit_transform(torch.tensor(X, dtype=torch.float32))

        self.attack = PGD(
            model=self.lin_net,
            loss_fn=self.loss_fn,
            p=self.p,
            adv_radius=self.adv_radius,
            step_size=self.step_size,
            nsteps=self.nsteps,
        )
        
        self.laff_adversarial_training(
            Z, torch.tensor(y, dtype=torch.float32), verbose=self.verbose
        )
        return self
    
    def create_attack(self):
        attack = PGD(
            model=self.lin_net,
            loss_fn=self.loss_fn,
            p=self.p,
            adv_radius=self.adv_radius,
            step_size=self.step_size,
            nsteps=self.nsteps,
        )

        return attack
    

    def laff_adversarial_training(self, X, y, verbose=False):
        self.lin_net.train()
        optimizer = torch.optim.Adam(self.lin_net.parameters(), lr=self.lr)
        for epoch in range(self.nepochs):
            optimizer.zero_grad()
            X_adv = self.attack(X, y)
            y_adv = self.lin_net(X_adv)
            loss = self.loss_fn(y_adv, y)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.nepochs}] | Loss = {loss.item():.4f}")

    def predict(self, X):
        Z = self.randomfeatures.transform(torch.tensor(X, dtype=torch.float32))
        pred = self.lin_net(Z)
        return pred
