from datasets import *
import time
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error)
from onedim_curve_fitting import get_kernel
from kernel_advtrain import AdvKernelTrain, AdvMultipleKernelTrain, LinearAdvFourierFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import os
from pgd import PGD
import torch
from pgd_attack_krr import KernelRidgeModel, fine_tunne_advtrain


def bootstrap(y_test, y_pred, metric, quantiles, n_boot=500):
    value_r2_bootstrap = np.zeros(n_boot)
    for i in range(n_boot):
        indexes = np.random.choice(range(len(y_pred)), len(y_pred))
        value_r2_bootstrap[i] = metric(y_test[indexes], y_pred[indexes])
    return [np.quantile(value_r2_bootstrap, q) for q in quantiles]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', choices=['regr', 'regr_short' ], default='regr')
    parser.add_argument('--dont_plot', action='store_true', help='Enable plotting')
    parser.add_argument('--dont_show', action='store_true', help='dont show plot, but maybe save it')
    parser.add_argument('--load_data', action='store_true', help='Enable data loading')
    parser.add_argument('--csv_file', type=str, default='out/performance_regr.csv',
                        help='Output data')
    parser.add_argument('--figure_dir', type=str, default='img',
                        help='Output figures')
    parser.add_argument('--adv_radius', default=0, help='adversarial radius to evaluate the data')
    parser.add_argument('--style', type=str, nargs='+', default=[], help='Style file to be used')
    parser.add_argument('--dont_plot_figure', action='store_true', help='Plot figure')
    args = parser.parse_args()

    adv_radius = float(args.adv_radius)
    if args.setting == 'regr':
        all_methods = ['akr', 'kr_cv', 'amkl']
        datasets = [polution, diabetes, us_crime, wine, abalone]
        tp = 'regression'
        metrics_names = ['r2_score', 'mape']
        metrics_of_interest = [r2_score, mean_absolute_percentage_error]
        metric_show = 'r2_score'
        ord = np.inf
        ylabel = 'R2-score'
        methods_to_show = ['akr', 'kr_cv']
        methods_name = ['Adv Kern', 'Kernel Ridge']
    elif args.setting == 'regr_short':
        all_methods = ['akr', 'kr_cv', 'amkl', 'adv-inp-2', 'adv-inp-inf']
        datasets = [polution, diabetes, us_crime, wine, abalone]
        tp = 'regression'
        metrics_names = ['r2_score', 'mape']
        metrics_of_interest = [r2_score, mean_absolute_percentage_error]
        metric_show = 'mape'
        ord = np.inf
        ylabel = 'MAPE'
        methods_to_show = ['akr', 'kr_cv']
        methods_name = ['Adv Kern', 'Kernel Ridge']
    else:
        raise ValueError('Setting not implemented')

    if args.load_data:
        print('Loading data from file')
        df = pd.read_csv(args.csv_file)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    else:
        columns_names = ['dset', 'method', 'adv', 'p', 'radius'] + metrics_names + \
                        [nn + q for nn in metrics_names for q in ['q1', 'q3']] + \
                        ['exec_time']

        if not os.path.exists(os.path.dirname(args.csv_file)):
            os.makedirs(os.path.dirname(args.csv_file))

        if not os.path.exists(args.figure_dir):
            os.makedirs(args.figure_dir)

        all_results = []
        for dset in datasets:
            X_train, X_test, y_train, y_test = dset()
            for method in all_methods:
                print('method:' + method)
                n_test = len(y_test)
                start_time = time.time()
                kernel, kernel_params = get_kernel('rbf')
                if method == 'akr':
                    est = AdvKernelTrain(verbose=False, kernel=kernel, **kernel_params)
                    est = GridSearchCV(est, param_grid={"gamma": [10, 1e0, 0.1, 1e-2, 1e-3]})
                elif method in ['kr_cv', 'adv-inp-2', 'adv-inp-inf']:
                    est = KernelRidge(kernel='rbf',
                                      **kernel_params)  # Needs to have rbf here, otherwise cross-validation will not work, because kernel ridge already define parameter
                    est = GridSearchCV(est, param_grid={"alpha": [10, 1e0, 0.1, 1e-2, 1e-3],
                                                        "gamma": [10, 1e0, 0.1, 1e-2, 1e-3]})
                elif method == 'amkl':
                    est = AdvMultipleKernelTrain(verbose=False, kernel=5 * ['rbf'],
                                                 kernel_params=[{'gamma': 10}, {'gamma': 1e0}, {'gamma': 0.1},
                                                                {'gamma': 1e-2}, {'gamma': 1e-3}])
                estimator = est.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)

                if method == 'adv-inp-2':
                    model = KernelRidgeModel.from_sklearn('rbf', estimator)
                    model = fine_tunne_advtrain(model, X_train, y_train, p=2, nepochs=200)
                    y_pred = model(torch.tensor(X_test))
                elif method == 'adv-inp-inf':
                    model = KernelRidgeModel.from_sklearn('rbf', estimator)
                    model = fine_tunne_advtrain(model, X_train, y_train, p=np.inf, nepochs=200)
                    y_pred = model(torch.tensor(X_test))

                if isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.detach().numpy()

                exec_time = time.time() - start_time

                ms = [dset.__name__, method, 'False', 0, 0]
                ms += [m(y_test, y_pred) for m in metrics_of_interest]
                for m in metrics_of_interest:
                    ms += bootstrap(y_test, y_pred, m, [0.25, 0.75])
                ms += [exec_time]
                all_results.append(ms)

                if method != 'amkl':
                    if method in ['akr', 'kr_cv']:
                        model = KernelRidgeModel.from_sklearn('rbf', estimator)
                    params = [
                        {'loss_fn': torch.nn.MSELoss(), 'p': 2, 'adv_radius': rad, 'step_size': rad / 50, 'nsteps': 100}
                        for rad in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]]
                    params += [{'loss_fn': torch.nn.MSELoss(), 'p': np.inf, 'adv_radius': rad, 'step_size': rad / 10,
                                'nsteps': 100} for rad in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]]
                    for param in params:
                        attack = PGD(model, **param)
                        X_adv = attack(torch.tensor(X_test), torch.tensor(y_test))
                        print(model(torch.tensor(X_test)))
                        y_pred_adv = model(X_adv).detach().numpy()
                        ms = [dset.__name__, method, 'True', param['p'], param['adv_radius']]
                        ms += [m(y_test, y_pred_adv) for m in metrics_of_interest]
                        for m in metrics_of_interest:
                            ms += bootstrap(y_test, y_pred_adv, m, [0.25, 0.75])
                        ms += [exec_time]
                        all_results.append(ms)

                df = pd.DataFrame(all_results, columns=columns_names)
                df.to_csv(args.csv_file)
                print(df)

    # Load data
    print(df)
    df = df[~df['adv']]
    for nn in metrics_names:
        print(nn)
        # Also print preprocessed version for the paper
        ddf = df[['dset', 'method', nn]].set_index(['dset', 'method']).iloc[:, 0]
        ddf = ddf.unstack('method')
        ddf.index.name = None
        print(ddf.to_latex(columns=[m for m in all_methods], float_format="%.2f"))

    # Plot figure
    if not args.dont_plot:
        print('Plotting figure')
        from matplotlib import ticker

        plt.style.use(args.style)

        fig, ax = plt.subplots()
        width = 0.6
        bar_width = width / len(methods_to_show)
        ind = np.arange(len(datasets))

        for i in range(len(methods_to_show)):
            ddf = df[df['method'] == methods_to_show[i]]
            y = ddf[metric_show]
            yerr = [ddf[metric_show] - ddf[metric_show + 'q1'], ddf[metric_show + 'q3'] - ddf[metric_show]]
            # Center the bars around each x position
            offset = (i - len(methods_to_show) / 2) * bar_width + bar_width / 2
            bar_positions = ind + offset

            ax.bar(bar_positions, y, bar_width, yerr=yerr, label=methods_name[i])

        names = [d.__name__.replace('_', ' ').capitalize() for d in datasets]
        plt.xticks(range(len(datasets)), names)
        plt.ylabel(ylabel)
        plt.legend(title='', bbox_to_anchor=(0.55, 0.75))

        import matplotlib as mpl

        ax = plt.gca()
        major_names = [n for i, n in enumerate(names) if i % 2 == 0]
        minor_names = [n for i, n in enumerate(names) if i % 2 == 1]
        major_loc = [i for i, d in enumerate(datasets) if i % 2 == 0]
        minor_loc = [i for i, d in enumerate(datasets) if i % 2 == 1]
        ax.xaxis.set_major_locator(ticker.FixedLocator(major_loc))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_loc))
        ax.xaxis.set_minor_formatter(ticker.FixedFormatter(major_names))
        ax.xaxis.set_minor_formatter(ticker.FixedFormatter(minor_names))
        ax.tick_params(axis='x', which='minor', length=-200)
        ax.tick_params(axis='x', which='both', color='lightgrey')
        ax.autoscale(enable=True, axis='x', tight=True)
        mpl.rcParams['xtick.major.pad'] = 12
        mpl.rcParams['xtick.minor.pad'] = 32
        mpl.rcParams['xtick.direction'] = 'in'

        plt.savefig(f'{args.figure_dir}/performace_{tp}.pdf')
        plt.show()