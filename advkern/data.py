import numpy as np


def curve1(rng, n, std_noise=0.2):
    X = 5 * rng.rand(n, 1)
    y = np.sin(X).ravel()
    # Add noise to targets
    y += 0.1 * (0.5 - rng.rand(X.shape[0]))
    X_plot = np.linspace(0, 5, 10000)[:, None]
    y_plot = np.sin(X_plot).ravel()
    return X, y, X_plot, y_plot


def curve2(rng, n, std_noise=0.2):
    ntest = 400
    X = rng.rand(n)
    Xtest = np.linspace(0, 1, ntest)
    y = np.sin(4 * np.pi * X) + std_noise * rng.randn(n)
    ytest = np.sin(4 * np.pi * Xtest)
    return  X.reshape(-1, 1), y, Xtest.reshape(-1, 1), ytest


def curve3(rng, n, std_noise=0.2):
    ntest = 400
    X = rng.rand(n)
    Xtest = np.linspace(0, 1, ntest)
    y = np.sign(np.sin(4 * np.pi * X)) + std_noise * rng.randn(n)
    ytest = np.sign(np.sin(4 * np.pi * Xtest))
    return X.reshape(-1, 1), y, Xtest.reshape(-1, 1), ytest


def get_curve(rng, n, curve, std_noise=0.2):
    if curve == 1:
        return curve1(rng, n, std_noise)
    elif curve == 2:
        return curve2(rng, n, std_noise)
    elif curve == 3:
        return curve3(rng, n, std_noise)
    else:
        raise ValueError("Invalid curve")
