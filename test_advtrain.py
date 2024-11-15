# Import
import numpy as np
import pytest
from kernel_advtrain import kernel_adversarial_training, mkl_adversarial_training



def test_kernel_advtrain():
    params = np.array([1, 2, 3, 4, 5])
    n_train, n_params = 100, len(params)
    rng = np.random.RandomState(0)
    X = rng.randn(n_train, n_params)
    y = X @ params + 0.1 * rng.randn(n_train)

    krr = kernel_adversarial_training(X, y, kernel='linear')
    coefs = X.T @ krr.dual_coef_
    coefs_lin_advtrain = [1.00487159, 2.01035135, 3.02053247, 3.99403511, 4.98359855] # Resul from linear adversarial training 
    assert np.allclose(coefs, coefs_lin_advtrain)


def test_mkl_advtrain():
    params = np.array([1, 2, 3, 4, 5])
    n_train, n_params = 100, len(params)
    rng = np.random.RandomState(0)
    X = rng.randn(n_train, n_params)
    y = X @ params + 0.1 * rng.randn(n_train)

    krr = mkl_adversarial_training(X, y, kernel=['linear','linear'])
    coefs = X.T @ krr.dual_coef_
    print(coefs)
    coefs_lin_advtrain = [1.00487159, 2.01035135, 3.02053247, 3.99403511, 4.98359855] # Resul from linear adversarial training 
    assert np.allclose(2* coefs, coefs_lin_advtrain)


if __name__ == "__main__":
    pytest.main()

