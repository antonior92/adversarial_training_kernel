from kernel_advtrain import kernel_adversarial_training
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from onedim_curve_fitting import get_curve, get_kernel

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


rng = np.random.RandomState(42)
df = pd.DataFrame(columns=['curve', 'rep', 'train_size', 'mse'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="One-dimensional curve fitting")
    parser.add_argument('--kernel', type=str, default='mattern5/2', choices=['rbf', 'mattern1/2','mattern3/2', 'mattern5/2'], help='Kernel type')
    args = parser.parse_args()


    kernel, kernel_params = get_kernel(args.kernel, gamma=12)

    for c in [2, 3]:
        for train_size in np.logspace(1, 3, num=10):
            for rep in np.arange(5):
                train_size = int(train_size)
                # Generate input
                X, y, X_test, y_test = get_curve(rng, train_size, curve=c)

                akr = kernel_adversarial_training(X, y, adv_radius=None, verbose=False, kernel=kernel,
                                                  kernel_params=kernel_params)
                y_pred = akr.predict(X_test)

                mse = np.mean((y_pred - y_test) ** 2)

                df = df.append({'curve': c, 'rep': rep, 'train_size': train_size, 'mse': mse}, ignore_index=True)
        print(df)

    df_curve1 = df[df['curve'] == 2]
    x, y, lerr, uerr = get_quantiles(df_curve1['train_size'], np.array(df_curve1['mse']))
    plt.plot(x, y, color='b', label='smooth')
    plt.fill_between(x, y - lerr, y + uerr, color='b', alpha=0.2)

    df_curve2 = df[df['curve'] == 3]
    x, y, lerr, uerr = get_quantiles(df_curve2['train_size'], np.array(df_curve2['mse']))
    plt.plot(x, y, color='r', label='non-smooth')
    plt.fill_between(x, y - lerr, y + uerr, color='r', alpha=0.2)

    plt.title('Kernel: ' + args.kernel)
    plt.legend()
    plt.xlabel('Training size')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()