from onedim_curve_fitting import *
from kernel_advtrain import mkl_adversarial_training

if __name__ == "__main__":
    import argparse

    # Write me an argument parser
    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--curve', default=1, type=int, choices=[1, 2, 3],  help='Curve type (1, 2, or 3)')
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

    import matplotlib.pyplot as plt
    plt.style.use(args.style)


    plt.figure()
    plt.scatter(X, y, c="k", label="Data", zorder=1, edgecolors=(0, 0, 0))

    estimator = mkl_adversarial_training(X, y, kernel=['rbf', 'rbf', 'rbf'], kernel_params=[{'gamma': 3}, {'gamma': 0.3}, {'gamma': 0.03}],)
    y_pred = estimator.predict(X_plot)
    plt.plot(X_plot, y_pred,)

    #plt.xlabel("$$x$$")
    #plt.ylabel("$$f(x)$$")
    _ = plt.legend(loc='upper right')

    plt.show()

