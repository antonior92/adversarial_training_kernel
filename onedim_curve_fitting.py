#%% Imports and definition
from kernel_advtrain import kernel_adversarial_training, mkl_adversarial_training
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


def curve1(rng, n):
    X = 5 * rng.rand(n, 1)
    y = np.sin(X).ravel()
    # Add noise to targets
    y += 0.1 * (0.5 - rng.rand(X.shape[0]))
    X_plot = np.linspace(0, 5, 10000)[:, None]
    y_plot = np.sin(X_plot).ravel()
    return X, y, X_plot, y_plot


def curve2(rng, n):
    ntest = 400
    X = rng.rand(n)
    Xtest = np.linspace(0, 1, ntest)
    std_noise = 0.2
    y = np.sin(4 * np.pi * X) + std_noise * np.random.randn(n)
    ytest = np.sin(4 * np.pi * Xtest)
    return  X.reshape(-1, 1), y, Xtest.reshape(-1, 1), ytest


def curve3(rng, n):
    ntest = 400
    X = rng.rand(n)
    Xtest = np.linspace(0, 1, ntest)
    std_noise = 0.2
    y = np.sign(np.sin(4 * np.pi * X)) + std_noise * np.random.randn(n)
    ytest = np.sign(np.sin(4 * np.pi * Xtest))
    return X.reshape(-1, 1), y, Xtest.reshape(-1, 1), ytest



def get_curve(rng, n, curve):
    if curve == 1:
        return curve1(rng, n)
    elif curve == 2:
        return curve2(rng, n)
    elif curve == 3:
        return curve3(rng, n)
    else:
        raise ValueError("Invalid curve")


def sq_dist(a, b):
    C = ((a[:, None] - b[None, :]) ** 2)
    return C

def get_kernel(kernel, gamma = 12):
    if kernel == 'rbf':
        return "rbf", {'gamma': gamma}
    elif kernel == 'mattern1/2':
        def kernel(x, y, a = 2):
            return np.exp(-np.sqrt(sq_dist(x, y)) * gamma)
        return kernel, {}
    elif kernel == 'mattern3/2':
        def kernel(x, y ):
            temp = np.sqrt(sq_dist(x, y))
            return (1 + np.sqrt(3) * temp * gamma) * np.exp(-np.sqrt(3) *temp * gamma)
        return kernel, {}
    elif kernel == 'mattern5/2':
        def kernel(x, y):
            temp = np.sqrt(sq_dist(x, y))
            return (1 + np.sqrt(5) * temp * gamma + 5 * temp**2 * gamma ** 2 / 3) * np.exp(- np.sqrt(5) * temp * gamma)
        return kernel, {}
    else:
        raise ValueError("Invalid kernel")

if __name__ == "__main__":
    import argparse

    # Write me an argument parser
    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--curve', default=2, type=int, choices=[1, 2, 3],  help='Curve type (1, 2, or 3)')
    parser.add_argument('--n', type=int, default=100,  help='Number of data points')
    parser.add_argument('--train_size', type=int, default=100, help='Training size')
    parser.add_argument('--include_MKL', action='store_true', help='Include MKL in the fitting')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['rbf', 'mattern1/2','mattern3/2', 'mattern5/2'],help='Kernel type')
    args = parser.parse_args()

    rng = np.random.RandomState(42)
    n = args.n
    include_MKL = args.include_MKL

    X, y, X_plot, y_plot = get_curve(rng, n, curve=args.curve)

    def sq_dist(a, b):
        C = ((a[:, None] - b[None, :]) ** 2)
        return C

    kernel, kernel_params = get_kernel(args.kernel, gamma = 12)

    train_size = 100

    kr = GridSearchCV(
        KernelRidge(kernel=kernel, **kernel_params),
        param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]},
    )

    # Fit three models
    kr.fit(X, y)
    y_kr = kr.predict(X_plot)
    gamma = kr.best_estimator_.gamma
    print(gamma)

    akr = kernel_adversarial_training(X, y, verbose=False, kernel=kernel,
                                      kernel_params=kernel_params)
    y_akr = akr.predict(X_plot)

    if include_MKL:
        amkl = mkl_adversarial_training(X, y, adv_radius=1e-2, verbose=False,
                                        kernel=["rbf", "chi2"])
        y_amkl = amkl.predict(X_plot)

    import matplotlib.pyplot as plt

    plt.scatter(X, y, c="k", label="data", zorder=1, edgecolors=(0, 0, 0))

    plt.plot(X_plot, y_plot, c='k', ls=':', label='True')
    plt.plot(X_plot, y_kr, c="g", label="Kernel Ridge Regression with CV")

    plt.plot(X_plot, y_akr, c="b",  label="Adversarial Kernel regression")

    if include_MKL:
        plt.plot(X_plot, y_amkl, c="r",  label="Adversarial MKL")

    plt.xlabel("data")
    plt.ylabel("target")
    _ = plt.legend()
    plt.show()

