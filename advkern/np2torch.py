import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from torch import nn as nn

from advkern.kernel_advtrain import AdvKernelTrain
from advkern.kernels import get_kernel


class KernelRidgeModel(nn.Module):
    def __init__(self, kernel, dual_coef_, X_fit, best_params=None):
        super().__init__()
        self.kernel = kernel
        self.dual_coef_ = torch.nn.Parameter(torch.tensor(dual_coef_))
        self.X_fit = torch.tensor(X_fit)
        self.best_params = best_params

    def forward(self, x):
        kernel, kernel_params = get_kernel(kernel=self.kernel, usetorch=True)
        if self.best_params is not None:
            for k, v in self.best_params.items():
                if k in kernel_params:
                    kernel_params[k] = v
        else:
            kernel_params = kernel_params
        K = kernel(x, self.X_fit, **kernel_params)
        return K @ self.dual_coef_

    @classmethod
    def from_sklearn(cls, kernel_name, krr_sklearn, best_params=None):
        if isinstance(krr_sklearn, GridSearchCV):
            return cls.from_sklearn(kernel_name, krr_sklearn.best_estimator_, best_params=krr_sklearn.best_params_)
        elif isinstance(krr_sklearn, KernelRidge):
            return cls(kernel_name, krr_sklearn.dual_coef_, krr_sklearn.X_fit_, best_params=best_params)
        elif isinstance(krr_sklearn, AdvKernelTrain):
            return cls.from_sklearn(kernel_name, krr_sklearn.model_, best_params=best_params)
        else:
            raise NotImplementedError()
