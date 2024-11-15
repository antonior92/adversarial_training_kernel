from kernel_advtrain import kernel_adversarial_training
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import pandas as pd

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

test_size = 1000

df = pd.DataFrame(columns=['rep', 'train_size', 'mse'])

for train_size in np.logspace(1, 3, num=10):
    for rep in np.arange(10):
        train_size = int(train_size)
        # Generate input
        X = 5 * rng.rand(train_size + test_size, 1)
        y = np.sin(X).ravel() + 0.5 * rng.randn(X.shape[0])

        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        akr = kernel_adversarial_training(X[:train_size], y[:train_size], adv_radius=1e-2, verbose=False, kernel="rbf", gamma=0.1)
        y_pred = akr.predict(X_test)

        kr =  KernelRidge(kernel="rbf", gamma=0.1).fit(X_train, y_train)
        y_kr = kr.predict(X_test)

        y_true = np.sin(X_test).ravel()

        mse = np.mean((y_true - y_pred)** 2)
        mse_kr = np.mean((y_kr - y_pred)** 2)

        df = df.append({'rep': rep, 'train_size': train_size, 'mse': mse, 'mse_kr':  mse_kr}, ignore_index=True)
print(df)

import matplotlib.pyplot as plt

x, y, lerr, uerr = get_quantiles(df['train_size'], df['mse'])
plt.errorbar(x, y, yerr=[lerr, uerr], color='b')

#x, y, lerr, uerr = get_quantiles(df['train_size'], df['mse_kr'])
#plt.errorbar(x, y, yerr=[lerr, uerr], color='r')
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e-3, 1e0])
plt.xlim([1e0, 1e3])
plt.show()