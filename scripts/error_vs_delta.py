import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from advkern.kernel_advtrain import AdvKernelTrain
from sklearn.model_selection import GridSearchCV
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
        est = KernelRidge(kernel=kernel, **kernel_params)

    est.fit(X, y)
    return est


rng = np.random.RandomState(42)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--kernel', type=str, default='linear', choices=valid_kernels, help='Kernel type')
    parser.add_argument('--estimate', type=str, nargs='+',  default=['kr_cv', 'akr', 'kr' ], choices=['akr', 'kr_cv', 'amkl', 'kr' ], help='Estimation method')
    parser.add_argument('--n_points', type=int, default=10, help='Number of grid points to plot')
    parser.add_argument('--n_reps', type=int, default=20, help='Number of repetitions')
    parser.add_argument('--csv_file', type=str,  default='out/error_vs_sample_size.csv', help='Output file')
    parser.add_argument('--min_log_range', type=int, default=-5, help='Minimum range')
    parser.add_argument('--max_log_range', type=int, default=0, help='Maximum range')
    parser.add_argument('--min_linear_range', type=float, default=1e-5, help='Minimum range')
    parser.add_argument('--max_linear_range', type=float, default=1, help='Maximum range')
    parser.add_argument('--load', action='store_true', help='Load data from file')
    parser.add_argument('--dataset', type=str, default='linear', choices=['sine_1d', 'squarewave', 'linear'], help='what type of data to use')
    parser.add_argument('--dont_plot_figure', action='store_true', help='Plsot figure')
    parser.add_argument('--save_figure', type=str, default='', help='Output figure')
    parser.add_argument('--style', type=str, nargs='+',  default=[], help='Style file to be used')
    parser.add_argument('--train_size', type=int, default=100, help='Training size')
    parser.add_argument('--linear_range', action='store_true', help='Linear data')
    args = parser.parse_args()
    print(args)

    kernel, kernel_params = get_kernel(args.kernel)
    df = pd.DataFrame(columns=['method', 'dataset', 'rep', 'train_size', 'mse', ])
    for i, method in enumerate(args.estimate):
        kernel_params = {}
        if args.linear_range:
            ss = np.linspace(args.min_linear_range, args.max_linear_range, args.n_points)
        else:
            ss = np.logspace(args.min_log_range, args.max_log_range, args.n_reps)
        for delta, alpha in zip(ss,ss):
            if method == 'akr':
                kernel_params['adv_radius'] = delta
            elif method == 'kr':
                kernel_params['alpha'] = alpha
            for rep in np.arange(args.n_reps):
                # Generate input
                if args.dataset == 'sine_1d':
                    X, y, X_test, y_test = get_curve(rng, args.train_size, curve=2, std_noise=0.1)
                elif args.dataset == 'squarewave':
                    X, y, X_test, y_test = get_curve(rng, args.train_size, curve=3, std_noise=0.1)
                else:
                    X, y, X_test, y_test = get_linear(rng, args.train_size)
                estimator = get_estimate(X, y, kernel, method=method, kernel_params=kernel_params)
                y_pred = estimator.predict(X_test)
                mse = np.mean((y_pred - y_test) ** 2)

                if method == 'akr':
                    print('rep', rep, 'dataset', args.dataset, 'delta', delta, 'mse', mse, 'method', method)
                    new_row = pd.DataFrame([{'method': method, 'dataset': args.dataset, 'rep': rep, 'train_size': args.train_size, 'delta/alpha': delta, 'mse': mse}])
                else:
                    print('rep', rep, 'dataset', args.dataset, 'alpha', alpha, 'mse', mse, 'method', method)
                    new_row = pd.DataFrame([{'method': method, 'dataset': args.dataset, 'rep': rep, 'train_size': args.train_size, 'delta/alpha': alpha, 'mse': mse}])
                df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(args.csv_file)

    if not args.dont_plot_figure:
        print('Plotting figure')
        import matplotlib.pyplot as plt

        plt.style.use(args.style)

        method = {0: 'Ridge CV', 1: 'Adv Kern'}
        color= {0: 'b', 1: 'g'}
        sss = pd.DataFrame(columns=['method', 'kernel', 'dataset', 'rate'])
        print('\nakr')
        for ii, mm in enumerate(['kr_cv', 'akr']):
            df_curve2 = df[(df['method'] == mm)]
            x, y, lerr, uerr = get_quantiles(df_curve2['delta/alpha'], np.array(df_curve2['mse']))
            if mm == 'kr_cv':  # Make Ridge CV constant
                y[:] = y[0]
                lerr[:] = lerr[0]
                uerr[:] = uerr[0]
            print(f'{method[ii]}, x: x = {np.round(x, 6)}')
            print(f'{method[ii]}, y: y = {np.round(y, 3)}\n with lb = {np.round(y - lerr, 3)}\n and ub = {np.round(y + uerr, 3)}')
            plt.plot(x, y,  label=method[ii], color=color[ii])
            plt.fill_between(x, y - lerr, y + uerr, alpha=0.2, color=color[ii])
        #plt.title('Kernel: ' + args.kernel.capitalize())
        plt.legend()
        plt.xlabel('$\\delta$')
        plt.ylabel('Test MSE')
        plt.ylim(5e-4, 1)
        if not args.linear_range:
            plt.xscale('log')
            plt.yscale('log')
        if args.save_figure == '':
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(args.save_figure + '_akr.pdf')

        plt.figure()
        method = {0: 'Ridge CV', 1: 'Ridge'}
        color= {0: 'b', 1: 'g'}
        sss = pd.DataFrame(columns=['method', 'kernel', 'dataset', 'rate'])
        print('\nkr')
        for ii, mm in enumerate(['kr_cv', 'kr']):
            df_curve2 = df[(df['method'] == mm)]
            x, y, lerr, uerr = get_quantiles(df_curve2['delta/alpha'], np.array(df_curve2['mse']))
            if mm == 'kr_cv':  # Make Ridge CV constant
                y[:] = y[0]
                lerr[:] = lerr[0]
                uerr[:] = uerr[0]
            print(f'{method[ii]}, x: x = {np.round(x, 6)}')
            print(f'{method[ii]}, y: y = {np.round(y, 3)}\n with lb = {np.round(y - lerr, 3)}\n and ub = {np.round(y + uerr, 3)}')
            plt.plot(x, y,  label=method[ii], color=color[ii])
            plt.fill_between(x, y - lerr, y + uerr, alpha=0.2, color=color[ii])

        #plt.title('Kernel: ' + args.kernel.capitalize())
        plt.legend()
        plt.xlabel('$\\lambda$')
        plt.ylabel('Test MSE')
        plt.ylim(5e-4, 1)
        if not args.linear_range:
            plt.xscale('log')
            plt.yscale('log')
        if args.save_figure == '':
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(args.save_figure + '_kr.pdf')
