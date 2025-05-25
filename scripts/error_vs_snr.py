import numpy as np
import pandas as pd

from scripts.onedim_curve_fitting import get_estimate
from advkern.data import get_curve
from advkern.kernels import valid_kernels, get_kernel


def get_quantiles(xaxis, r, quantileslower=0.25, quantilesupper=0.75):
    new_xaxis, inverse, counts = np.unique(xaxis, return_inverse=True, return_counts=True)
    r_values = np.zeros([len(new_xaxis), max(counts)])
    secondindex = np.zeros(len(new_xaxis), dtype=int)
    for n in range(len(xaxis)):
        i = inverse[n]
        j = secondindex[i]
        r_values[i, j] = r[n]
        secondindex[i] += 1
    m = np.median(r_values, axis=1)
    lerr = m - np.quantile(r_values, quantileslower, axis=1)
    uerr = np.quantile(r_values, quantilesupper, axis=1) - m
    return new_xaxis, m, lerr, uerr

def get_linear(rng, train_size, std_noise=0.1, input_size=1, test_size=100, n_params=10, insample=True):
    true_parameter = rng.randn(n_params)
    true_parameter *= input_size / np.linalg.norm(true_parameter)
    X = rng.randn(train_size, len(true_parameter))
    e = rng.randn(train_size)
    y = X @ true_parameter + std_noise * e
    if insample:
        X_test = X
    else:
        X_test = rng.randn(test_size, len(true_parameter))
    y_test = X_test @ true_parameter

    return X, y, X_test, y_test


rng = np.random.RandomState(42)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--kernel', type=str, default='linear', choices=valid_kernels, help='Kernel type')
    parser.add_argument('--estimate', type=str, nargs='+',  default=['kr_cv', 'akr', 'kr' ], choices=['akr', 'kr_cv', 'amkl', 'kr' ], help='Estimation method')
    parser.add_argument('--n_points', type=int, default=10, help='Number of grid points to plot')
    parser.add_argument('--n_reps', type=int, default=20, help='Number of repetitions')
    parser.add_argument('--csv_file', type=str,  default='out/error_vs_sample_size.csv', help='Output file')
    parser.add_argument('--min_log_range', type=int, default=-7, help='Minimum range')
    parser.add_argument('--max_log_range', type=int, default=0, help='Maximum range')
    parser.add_argument('--min_linear_range', type=float, default=0, help='Minimum range')
    parser.add_argument('--max_linear_range', type=float, default=0.5, help='Maximum range')
    parser.add_argument('--load', action='store_true', help='Load data from file')
    parser.add_argument('--dataset', type=str, default='linear', choices=['sine_1d', 'squarewave', 'linear'], help='what type of data to use')
    parser.add_argument('--dont_plot_figure', action='store_true', help='Plot figure')
    parser.add_argument('--save_figure', type=str, default='', help='Output figure')
    parser.add_argument('--style', type=str, nargs='+',  default=[], help='Style file to be used')
    parser.add_argument('--train_size', type=int, default=100, help='Training size')
    parser.add_argument('--linear_range', action='store_true', help='Linear data')
    args = parser.parse_args()
    print(args)

    if args.load:
        print('Loading data from file')
        df = pd.read_csv(args.csv_file)
        print(df)
    else:
        kernel, kernel_params = get_kernel(args.kernel)
        df = pd.DataFrame(columns=['method', 'dataset', 'rep', 'train_size', 'mse', ])
        for i, method in enumerate(args.estimate):
            if args.linear_range:
                ss = np.linspace(args.min_linear_range, args.max_linear_range, args.n_points)
            else:
                ss = np.logspace(args.min_log_range, args.max_log_range, args.n_reps)
            for std_noise in ss:
                for rep in np.arange(args.n_reps):
                    # Generate input
                    if args.dataset == 'sine_1d':
                        X, y, X_test, y_test = get_curve(rng, args.train_size, curve=2, std_noise=std_noise)
                    elif args.dataset == 'squarewave':
                        X, y, X_test, y_test = get_curve(rng, args.train_size, curve=3, std_noise=std_noise)
                    else:
                        X, y, X_test, y_test = get_linear(rng, args.train_size, std_noise=std_noise)
                    estimator = get_estimate(X, y, kernel, method=method, kernel_params=kernel_params)
                    y_pred = estimator.predict(X_test)
                    mse = np.mean((y_pred - y_test) ** 2)

                    print('rep', rep, 'dataset', args.dataset, 'std_noise', std_noise, 'mse', mse, 'method', method)
                    df = df.append({'method': method, 'dataset': args.dataset, 'rep': rep, 'train_size': args.train_size,'std_noise': std_noise , 'mse': mse}, ignore_index=True)
        df.to_csv(args.csv_file)

    if not args.dont_plot_figure:
        print('Plotting figure')
        import matplotlib.pyplot as plt

        plt.style.use(args.style)
        plt.figure()

        method = {0: 'Adv Kern', 1: 'Ridge CV', 2: 'Ridge'}
        color= {0: 'b', 1: 'g', 2: 'r'}
        sss = pd.DataFrame(columns=['method', 'kernel', 'dataset', 'rate'])
        for ii, mm in enumerate(args.estimate):
            df_curve2 = df[(df['method'] == mm)]
            x, y, lerr, uerr = get_quantiles(df_curve2['std_noise'], np.array(df_curve2['mse']))
            plt.plot(x, y,  label =method[ii], color=color[ii])
            plt.fill_between(x, y - lerr, y + uerr, alpha=0.2, color=color[ii])

        plt.title('Kernel: ' + args.kernel.capitalize())
        plt.legend()
        plt.xlabel('Noise standard deviation')
        plt.ylabel('Test MSE')
        if not args.linear_range:
            plt.xscale('log')
            plt.yscale('log')
        if args.save_figure == '':
            plt.show()
        else:
            plt.savefig(args.save_figure)
