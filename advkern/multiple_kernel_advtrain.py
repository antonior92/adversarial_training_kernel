import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_kernels
from torch import nn as nn

from advkern.kernel_advtrain import mfactor, get_norm, eta_trick, get_update_size


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
        adv_radius = mfactor / np.sqrt(n_train)

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
