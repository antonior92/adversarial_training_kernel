#%% Imports and definition
from kernel_advtrain import kernel_adversarial_training, mkl_adversarial_training, AdvKernelTrain, LinearAdvFourierFeatures
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


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



valid_kernels = ['rbf', 'linear', 'matern1-2', 'matern3-2', 'matern5-2']

def get_kernel(kernel, usetorch=False):
    default_gamma = 12
    if usetorch:
        import torch
        myexp = torch.exp
        mysqrt = torch.sqrt

        def sq_dist(x, y):
            """
            Compute the squared Euclidean distance between all rows of x and y using broadcasting.

            Parameters:
            x: torch.Tensor of shape (n_samples_x, n_features)
            y: torch.Tensor of shape (n_samples_y, n_features)

            Returns:
            A distance matrix of shape (n_samples_x, n_samples_y)
            """
            x = x.unsqueeze(1)  # (n_samples_x, 1, n_features)
            y = y.unsqueeze(0)  # (1, n_samples_y, n_features)
            return torch.sum((x - y) ** 2, dim=2)
    else:
        myexp = np.exp
        mysqrt = np.sqrt

        def sq_dist(X, Y):
            """
            Compute the squared Euclidean distance between all rows of X and Y using broadcasting.

            Parameters:
            X: np.ndarray of shape (n_samples_x, n_features)
            Y: np.ndarray of shape (n_samples_y, n_features)

            Returns:
            A distance matrix of shape (n_samples_x, n_samples_y)
            """
            X = np.atleast_2d(X)
            Y = np.atleast_2d(Y)
            x_exp = X[:, np.newaxis, :]  # shape: (n_samples_x, 1, n_features)
            y_exp = Y[np.newaxis, :, :]  # shape: (1, n_samples_y, n_features)
            S = np.sum((x_exp - y_exp) ** 2, axis=2)
            return S

    if kernel == 'rbf':
        def kernel(x, y, gamma=default_gamma):
            return myexp(-sq_dist(x, y) * gamma)
        return kernel, {}
    elif kernel == 'linear':
        def kernel(x, y):
            return x @ y.T
        return kernel, {}
    elif kernel == 'matern1-2':
        def kernel(x, y, gamma=default_gamma):
            return myexp(-mysqrt(sq_dist(x, y)) * gamma)
        return kernel, {}
    elif kernel == 'matern3-2':
        def kernel(x, y, gamma=default_gamma):
            temp = mysqrt(sq_dist(x, y))
            return (1 + mysqrt(3) * temp * gamma) * myexp(-mysqrt(3) *temp * gamma)
        return kernel, {}
    elif kernel == 'matern5-2':
        def kernel(x, y, gamma=default_gamma):
            temp = mysqrt(sq_dist(x, y))
            return (1 + mysqrt(5) * temp * gamma + 5 * temp**2 * gamma ** 2 / 3) * myexp(- mysqrt(5) * temp * gamma)
        return kernel, {}
    else:
        raise ValueError("Invalid kernel")

def get_estimate(X, y, kernel, method='akr', kernel_params=None):
    n_train, n_features = X.shape
    if method == 'akr':
        est = AdvKernelTrain(verbose=False, kernel=kernel, **kernel_params)
    elif method == 'akr_cv':
        est = AdvKernelTrain(verbose=False, kernel=kernel, **kernel_params)
        est = GridSearchCV(est, param_grid={"gamma": [10, 1e0, 0.1, 1e-2, 1e-3]})
    elif method == 'kr_cv':
        est = KernelRidge(kernel=kernel, **kernel_params)
        est = GridSearchCV(est, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]})
    elif method == 'kr':
        est = KernelRidge(kernel=kernel, alpha=1/np.sqrt(n_train), **kernel_params)

    est.fit(X, y)
    return est




if __name__ == "__main__":
    import argparse

    # Write me an argument parser
    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--curve', default=3, type=int, choices=[1, 2, 3],  help='Curve type (1, 2, or 3)')
    parser.add_argument('--n', type=int, default=40,  help='Number of data points')
    parser.add_argument('--train_size', type=int, default=100, help='Training size')
    parser.add_argument('--kernel', type=str, default='rbf', choices=valid_kernels, help='Kernel type')
    parser.add_argument('--save_figure', type=str, default='', help='Output figure')
    parser.add_argument('--gamma', type=float, default=12, help='Gamma parameter for the kernel')
    parser.add_argument('--style', type=str, nargs='+',  default=[], help='Style file to be used')
    parser.add_argument('--std_noise', type=float, default=0.2, help='Standard deviation of the noise')
    parser.add_argument('--rng', type=int, default=4, help='Random number generator')
    args = parser.parse_args()


    rng = np.random.RandomState(args.rng)
    n = args.n

    X, y, X_plot, y_plot = get_curve(rng, n, curve=args.curve, std_noise=args.std_noise)


    kernel, kernel_params = get_kernel(args.kernel)
    train_size = 100

    kr = GridSearchCV(
        KernelRidge(kernel=kernel, **kernel_params),
        param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]},
    )


    import matplotlib.pyplot as plt
    plt.style.use(args.style)


    plt.figure()
    plt.scatter(X, y, c="k", label="Data", zorder=1, edgecolors=(0, 0, 0))

    for method in ['akr', 'kr_cv']:
        estimator = get_estimate(X, y, kernel, method=method, kernel_params=kernel_params)
        y_pred = estimator.predict(X_plot)
        label = 'Adv Kern' if method == 'akr' else 'Ridge CV'
        plt.plot(X_plot, y_pred, label=label)
    plt.plot(X_plot, y_plot, c='k', ls=':', label='True')

    #plt.xlabel("$$x$$")
    #plt.ylabel("$$f(x)$$")
    _ = plt.legend(loc='upper right')

    if args.save_figure == '':
        plt.show()
    else:
        plt.savefig(args.save_figure)

