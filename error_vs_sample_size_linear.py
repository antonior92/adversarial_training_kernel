import numpy as np
import pandas as pd
from onedim_curve_fitting import get_curve, get_kernel, get_estimate, valid_kernels

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


def fit_line(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    theta = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = theta[0] * x + theta[1]
    return y_pred, theta[0], theta[1]

rng = np.random.RandomState(42)
df = pd.DataFrame(columns=['config', 'rep', 'train_size', 'mse'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--kernel', type=str, default='linear', choices=valid_kernels, help='Kernel type')
    parser.add_argument('--estimate', type=str, default='akr', choices=['akr', 'kr_cv', 'amkl'], help='Estimation method')
    parser.add_argument('--n_points', type=int, default=10, help='Number of grid points to plot')
    parser.add_argument('--n_reps', type=int, default=3, help='Number of repetitions')
    parser.add_argument('--csv_file', type=str, default='out/error_vs_sample_size.csv', help='Output file')
    parser.add_argument('--min_log_range', type=int, default=0, help='Minimum range')
    parser.add_argument('--max_log_range', type=int, default=1, help='Maximum range')
    parser.add_argument('--load', action='store_true', help='Load data from file')
    parser.add_argument('--dont_plot_figure', action='store_true', help='Plot figure')
    parser.add_argument('--save_figure', type=str, default='', help='Output figure')
    args = parser.parse_args()
    print(args)

    n_params = 50
    noise_std = 1
    input_size = 1
    test_size = 300
    configs = {'noise_std': [0.1, 1,  10]}
    n_configs = len(configs['noise_std'])

    if args.load:
        print('Loading data from file')
        df = pd.read_csv(args.csv_file)
        print(df)
    else:
        kernel, kernel_params = get_kernel(args.kernel, gamma=12)
        for rep in np.arange(args.n_reps):
            for c in range(n_configs):
                noise_std = configs['noise_std'][c]
                true_parameter = np.random.randn(n_params)
                true_parameter *= input_size / np.linalg.norm(true_parameter)
                for train_size in np.logspace(args.min_log_range, args.max_log_range, num=args.n_points):
                    train_size = int(train_size * n_params)
                    # Generate input
                    X = np.random.randn(train_size, len(true_parameter))
                    e = np.random.randn(train_size)
                    y = X @ true_parameter + noise_std * e
                    X_test = np.random.randn(test_size, len(true_parameter))
                    y_test = X_test @ true_parameter
                    # estimate
                    estimator = get_estimate(X, y, kernel, method=args.estimate, kernel_params=kernel_params)
                    y_pred = estimator.predict(X_test)
                    mse = np.mean((y_pred - y_test) ** 2)
                    df = df.append({'config': c, 'rep': rep, 'train_size': train_size, 'mse': mse}, ignore_index=True)
        print(df)
        df.to_csv(args.csv_file)


    if not args.dont_plot_figure:
        print('Plotting figure')
        import matplotlib.pyplot as plt
        plt.figure()
        for c in range(n_configs):
            df_curve1 = df[df['config'] == c]
            x, y, lerr, uerr = get_quantiles(df_curve1['train_size'], np.array(df_curve1['mse']))
            yp, t1, t2 = fit_line(np.log(x), np.log(y))
            nstd = configs['noise_std'][c]
            plt.plot(x, y, color='b', label=f'noise std = {nstd:.2f}')
            plt.fill_between(x, y - lerr, y + uerr, color='b', alpha=0.2)
            plt.plot(x, np.exp(yp), color='b', ls='--', lw=1, label=f'MSE $\propto$ n^({t1:.2f})')

        plt.title('Kernel: ' + args.kernel.capitalize())
        plt.legend()
        plt.xlabel('Training size')
        plt.ylabel('MSE')
        plt.xscale('log')
        plt.yscale('log')
        if args.save_figure == '':
            plt.show()
        else:
            plt.savefig(args.save_figure)