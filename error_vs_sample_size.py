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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--kernel', type=str, default='matern5-2', choices=valid_kernels, help='Kernel type')
    parser.add_argument('--estimate', type=str, nargs='+',  default=['kr_cv',], choices=['akr', 'kr_cv', 'amkl'], help='Estimation method')
    parser.add_argument('--n_points', type=int, default=10, help='Number of grid points to plot')
    parser.add_argument('--n_reps', type=int, default=5, help='Number of repetitions')
    parser.add_argument('--csv_file', type=str, nargs='+',  default=['out/error_vs_sample_size.csv',], help='Output file')
    parser.add_argument('--min_log_range', type=int, default=1, help='Minimum range')
    parser.add_argument('--max_log_range', type=int, default=3, help='Maximum range')
    parser.add_argument('--load', action='store_true', help='Load data from file')
    parser.add_argument('--dont_plot_figure', action='store_true', help='Plot figure')
    parser.add_argument('--save_figure', type=str, default='', help='Output figure')
    parser.add_argument('--style', type=str, nargs='+',  default=[], help='Style file to be used')
    args = parser.parse_args()
    print(args)

    if args.load:
        print('Loading data from file')
        df_list = [pd.read_csv(cc) for cc in args.csv_file]
        print(df_list)
    else:
        kernel, kernel_params = get_kernel(args.kernel, gamma=12)
        df_list = []
        for method in args.estimate:
            df = pd.DataFrame(columns=['curve', 'rep', 'train_size', 'mse'])
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
            df_list.append(df)

    if not args.dont_plot_figure:
        print('Plotting figure')
        import matplotlib.pyplot as plt

        plt.style.use(args.style)
        plt.figure()


        method = {0: 'Adv Kern', 1: 'Ridge CV'}
        label = {2:  'Smooth', 3:'Non-smooth'}
        colors = {(0, 2): 'b', (0, 3): 'r', (1, 2): 'g', (1, 3): 'y'}
        for curve in [2, 3]:
            for ii, df in enumerate(df_list):
                df_curve2 = df[df['curve'] == curve]
                x, y, lerr, uerr = get_quantiles(df_curve2['train_size'], np.array(df_curve2['mse']))
                yp, t1, t2 = fit_line(np.log(x), np.log(y))
                plt.plot(x, y, color=colors[(ii, curve)], label=label[curve] )
                plt.fill_between(x, y - lerr, y + uerr, color=colors[(ii, curve)], alpha=0.2)
                plt.plot(x, np.exp(yp), color=colors[(ii, curve)], ls='--', lw=1)
                print(f'MSE  n^{t1:.2f}')



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