#%% Imports and definition
from kernel_advtrain import kernel_adversarial_training, mkl_adversarial_training, AdvKernelTrain, LinearAdvFourierFeatures
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
    y = np.sin(4 * np.pi * X) + std_noise * rng.randn(n)
    ytest = np.sin(4 * np.pi * Xtest)
    return  X.reshape(-1, 1), y, Xtest.reshape(-1, 1), ytest


def curve3(rng, n):
    ntest = 400
    X = rng.rand(n)
    Xtest = np.linspace(0, 1, ntest)
    std_noise = 0.2
    y = np.sign(np.sin(4 * np.pi * X)) + std_noise * rng.randn(n)
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

valid_kernels = ['rbf', 'linear', 'matern1-2', 'matern3-2', 'matern5-2']

def get_kernel(kernel):
    default_gamma = 12
    if kernel == 'rbf':
        return "rbf", {'gamma': default_gamma}
    elif kernel == 'linear':
        return 'linear', {}
    elif kernel == 'matern1-2':
        def kernel(x, y, gamma=default_gamma):
            return np.exp(-np.sqrt(sq_dist(x, y)) * gamma)
        return kernel, {}
    elif kernel == 'matern3-2':
        def kernel(x, y, gamma=default_gamma):
            temp = np.sqrt(sq_dist(x, y))
            return (1 + np.sqrt(3) * temp * gamma) * np.exp(-np.sqrt(3) *temp * gamma)
        return kernel, {}
    elif kernel == 'matern5-2':
        def kernel(x, y, gamma=default_gamma):
            temp = np.sqrt(sq_dist(x, y))

            return (1 + np.sqrt(5) * temp * gamma + 5 * temp**2 * gamma ** 2 / 3) * np.exp(- np.sqrt(5) * temp * gamma)
        return kernel, {}
    else:
        raise ValueError("Invalid kernel")

def get_estimate(X, y, kernel, method='akr', kernel_params=None):
    if method == 'akr':
        est = AdvKernelTrain(verbose=False, kernel=kernel, **kernel_params)
    elif method == 'akr_cv':
        est = AdvKernelTrain(verbose=False, kernel=kernel, **kernel_params)
        est = GridSearchCV(est, param_grid={"gamma": [10, 1e0, 0.1, 1e-2, 1e-3]})
    elif method == 'kr_cv':
        est = KernelRidge(kernel=kernel, **kernel_params)
        est = GridSearchCV(est, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]})
    elif method == 'laff':
        est = LinearAdvFourierFeatures(R=1000, adv_radius=0, verbose=True)

    est.fit(X, y)
    return est




if __name__ == "__main__":
    import argparse

    # Write me an argument parser
    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--curve', default=3, type=int, choices=[1, 2, 3],  help='Curve type (1, 2, or 3)')
    parser.add_argument('--n', type=int, default=40,  help='Number of data points')
    parser.add_argument('--train_size', type=int, default=100, help='Training size')
    parser.add_argument('--kernel', type=str, default='matern5-2', choices=valid_kernels, help='Kernel type')
    parser.add_argument('--save_figure', type=str, default='', help='Output figure')
    parser.add_argument('--gamma', type=float, default=12, help='Gamma parameter for the kernel')
    parser.add_argument('--style', type=str, nargs='+',  default=[], help='Style file to be used')
    args = parser.parse_args()


    rng = np.random.RandomState(42)
    n = args.n

    X, y, X_plot, y_plot = get_curve(rng, n, curve=args.curve)

    def sq_dist(a, b):
        C = ((a[:, None] - b[None, :]) ** 2)
        return C

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

