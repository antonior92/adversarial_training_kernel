#%% Imports and definition
from advkern.data import get_curve
from advkern.kernel_advtrain import AdvKernelTrain
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

from advkern.kernels import valid_kernels, get_kernel


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

