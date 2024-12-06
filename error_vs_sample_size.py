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
df = pd.DataFrame(columns=['curve', 'rep', 'train_size', 'mse'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--kernel', type=str, default='matern5-2', choices=valid_kernels, help='Kernel type')
    parser.add_argument('--estimate', type=str, default='kr_cv', choices=['akr', 'kr_cv', 'amkl'], help='Estimation method')
    parser.add_argument('--n_points', type=int, default=10, help='Number of grid points to plot')
    parser.add_argument('--n_reps', type=int, default=5, help='Number of repetitions')
    parser.add_argument('--csv_file', type=str, default='out/error_vs_sample_size.csv', help='Output file')
    parser.add_argument('--min_log_range', type=int, default=1, help='Minimum range')
    parser.add_argument('--max_log_range', type=int, default=3, help='Maximum range')
    parser.add_argument('--load', action='store_true', help='Load data from file')
    parser.add_argument('--dont_plot_figure', action='store_true', help='Plot figure')
    parser.add_argument('--save_figure', type=str, default='', help='Output figure')
    args = parser.parse_args()
    print(args)

    if args.load:
        print('Loading data from file')
        df = pd.read_csv(args.csv_file)
        print(df)
    else:
        kernel, kernel_params = get_kernel(args.kernel, gamma=12)
        for c in [2, 3]:
            for train_size in np.logspace(args.min_log_range, args.max_log_range, num=args.n_points):
                for rep in np.arange(args.n_reps):
                    train_size = int(train_size)
                    # Generate input
                    X, y, X_test, y_test = get_curve(rng, train_size, curve=c)
                    estimator = get_estimate(X, y, kernel, method=args.estimate, kernel_params=kernel_params)
                    y_pred = estimator.predict(X_test)
                    mse = np.mean((y_pred - y_test) ** 2)
                    df = df.append({'curve': c, 'rep': rep, 'train_size': train_size, 'mse': mse}, ignore_index=True)
        print(df)
        df.to_csv(args.csv_file)
    if not args.dont_plot_figure:
        print('Plotting figure')
        import matplotlib.pyplot as plt
        plt.figure()
        df_curve2 = df[df['curve'] == 3]
        x, y, lerr, uerr = get_quantiles(df_curve2['train_size'], np.array(df_curve2['mse']))
        yp, t1, t2 = fit_line(np.log(x), np.log(y))
        plt.plot(x, y, color='r', label=f'non-smooth')
        plt.fill_between(x, y - lerr, y + uerr, color='r', alpha=0.2)
        plt.plot(x, np.exp(yp), color='r', ls='--', lw=1, label=f'MSE $\propto$ n^({t1:.2f})')

        df_curve1 = df[df['curve'] == 2]
        x, y, lerr, uerr = get_quantiles(df_curve1['train_size'], np.array(df_curve1['mse']))
        yp, t1, t2 = fit_line(np.log(x), np.log(y))
        plt.plot(x, y, color='b', label=f'smooth')
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