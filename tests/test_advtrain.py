# Import
import pytest
from advkern.kernel_advtrain import kernel_adversarial_training, AdvKernelTrain
from advkern.multiple_kernel_advtrain import mkl_adversarial_training, AdvMultipleKernelTrain
import cvxpy as cp
import numpy as np


def compute_q(p):
    if p != np.Inf and p > 1:
        q = p / (p - 1)
    elif p == 1:
        q = np.Inf
    else:
        q = 1
    return q


class AdversarialTrainingCVX:
    def __init__(self, X, y, p):
        m, n = X.shape
        q = compute_q(p)
        # Formulate problem
        param = cp.Variable(n)
        param_norm = cp.pnorm(param, p=q)
        adv_radius = cp.Parameter(name='adv_radius', nonneg=True)
        abs_error = cp.abs(X @ param - y)
        adv_loss = 1 / m * cp.sum((abs_error + adv_radius * param_norm) ** 2)
        prob = cp.Problem(cp.Minimize(adv_loss))
        self.prob = prob
        self.adv_radius = adv_radius
        self.param = param
        self.warm_start = False

    def __call__(self, adv_radius, **kwargs):
        try:
            self.adv_radius.value = adv_radius
            self.prob.solve(warm_start=self.warm_start, **kwargs)
            v = self.param.value
        except:
            v = np.zeros(self.param.shape)
        return v


def test_kernel_advtrain():
    params = np.array([1, 2, 3, 4, 5])
    adv_radius = 0.1
    n_train, n_params = 100, len(params)
    rng = np.random.RandomState(0)
    X = rng.randn(n_train, n_params)
    y = X @ params + 0.1 * rng.randn(n_train)

    krr = kernel_adversarial_training(X, y, adv_radius=0.1, kernel='linear')
    coefs = X.T @ krr.dual_coef_
    coefs_lin_advtrain = AdversarialTrainingCVX(X, y, p=2)(adv_radius=0.1) # Resul from linear adversarial training
    print(coefs)
    print(coefs_lin_advtrain)

    assert np.allclose(coefs, coefs_lin_advtrain, rtol=1e-3)

def test_akr_wrapper():
    params = np.array([1, 2, 3, 4, 5])
    n_train, n_params = 100, len(params)
    rng = np.random.RandomState(0)
    X = rng.randn(n_train, n_params)
    y = X @ params + 0.1 * rng.randn(n_train)

    model = AdvKernelTrain(adv_radius=0.1, kernel='linear')
    model.fit(X, y)

    # Assuming model.model_ is the object returned by kernel_adversarial_training
    coefs = X.T @ model.model_.dual_coef_
    coefs_lin_advtrain = AdversarialTrainingCVX(X, y, p=2)(adv_radius=0.1)

    assert np.allclose(coefs, coefs_lin_advtrain, rtol=1e-3)


def test_mkl_advtrain():
    params = np.array([1, 2, 3, 4, 5])
    n_train, n_params = 100, len(params)
    rng = np.random.RandomState(0)
    X = rng.randn(n_train, n_params)
    y = X @ params + 0.1 * rng.randn(n_train)

    krr = mkl_adversarial_training(X, y, kernel=['linear','linear'])
    coefs = X.T @ krr.dual_coef_
    coefs_lin_advtrain = AdversarialTrainingCVX(X, y, p=2)(adv_radius=0.1) # Resul from linear adversarial training
    assert np.allclose(2* coefs, coefs_lin_advtrain, rtol=1e-3)


def test_mkl_advtrain_wrapper():
    params = np.array([1, 2, 3, 4, 5])
    n_train, n_params = 100, len(params)
    rng = np.random.RandomState(0)
    X = rng.randn(n_train, n_params)
    y = X @ params + 0.1 * rng.randn(n_train)

    model = AdvMultipleKernelTrain(adv_radius=0.1, kernel=['linear','linear'])
    model.fit(X, y)

    # Assuming model.model_ is the object returned by kernel_adversarial_training
    coefs = X.T @ model.model_.dual_coef_
    coefs_lin_advtrain = AdversarialTrainingCVX(X, y, p=2)(adv_radius=0.1)

    assert np.allclose(2* coefs, coefs_lin_advtrain, rtol=1e-3)

if __name__ == "__main__":
    pytest.main()

