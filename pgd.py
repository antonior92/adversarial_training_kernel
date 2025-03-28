import torch
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
import numpy as np


class RBFRandomFourierFeatures:
    '''
    compute R random fourier features for each sample in X
    '''

    def __init__(self, R, sigmasq=1):
        '''
        R: number of random fourier features
        '''
        self.R = R
        self.sigmasq = sigmasq

    def fit_transform(self, X):
        '''
        note: W has R N(0, I * (1 / sigma_sq)) vectors on its rows
        '''
        p = X.shape[1]
        self.W = np.random.normal(
            loc=0.0, scale=1.0 / np.sqrt(self.sigmasq), size=(self.R, p)
        )  # R x p
        self.b = np.random.uniform(0, 2 * np.pi, size=(self.R,))  # (R,)
        Z = X @ self.W.T + self.b  # n x R
        Z = np.sqrt(2 / self.R) * np.cos(Z)
        return Z

    def transform(self, X):
        Z = X @ self.W.T + self.b
        Z = np.sqrt(2 / self.R) * np.cos(Z)
        return Z


# sample n points in the range [a, b); a <= b
a, b = 3, 10
N = 50
X = (a - b) * np.random.rand(N, 1) + b
# generate noisy observations of sin(x)
mu, sigmasq = 0, 0.1
y = np.sin(X).reshape(-1, 1) + np.random.normal(
    loc=mu, scale=np.sqrt(sigmasq), size=(N, 1)
)

# generate a smooth curve for sin(x) over the same range
X_eval = np.linspace(a, b, 1000).reshape(-1, 1)
y_eval = np.sin(X_eval)

# fit a kernel ridge regression model to the data
model = KernelRidge(kernel='rbf', gamma=1 / 2, alpha=0.1)
model.fit(X, y)
y_kr = model.predict(X_eval)

R = 10
randomfeatures = RBFRandomFourierFeatures(R)
Z = randomfeatures.fit_transform(X)
model = Ridge(alpha=0.1)
model.fit(Z, y)
y_r = model.predict(randomfeatures.transform(X_eval))

plt.plot(X, y, 'o')  # data
plt.plot(X_eval, y_eval, label='sin(x)', color='red')  # truth
plt.plot(X_eval, y_kr, label='kernel ridge', color='blue')  # kernel ridge fit
plt.plot(X_eval, y_r, label=f'rff + ridge, R = {R}', color='green')  # rff fit
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
