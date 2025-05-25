import numpy as np

valid_kernels = ['rbf', 'linear', 'matern1-2', 'matern3-2', 'matern5-2']


def get_kernel(kernel, usetorch=False):
    if usetorch:
        import torch
        myexp = torch.exp
        mysqrt = torch.sqrt

        def sq_dist(x, y):
            """
            Compute the squared Euclidean distance between all rows of x and y using broadcasting.

            Parameters:
            x: torch.Tensor of shape (n_samples_x, n_features)
            y: torch.Tensor of shape (n_samples_y, n_features)

            Returns:
            A distance matrix of shape (n_samples_x, n_samples_y)
            """
            x = x.unsqueeze(1)  # (n_samples_x, 1, n_features)
            y = y.unsqueeze(0)  # (1, n_samples_y, n_features)
            return torch.sum((x - y) ** 2, dim=2)
    else:
        myexp = np.exp
        mysqrt = np.sqrt

        def sq_dist(X, Y):
            """
            Compute the squared Euclidean distance between all rows of X and Y using broadcasting.

            Parameters:
            X: np.ndarray of shape (n_samples_x, n_features)
            Y: np.ndarray of shape (n_samples_y, n_features)

            Returns:
            A distance matrix of shape (n_samples_x, n_samples_y)
            """
            X = np.atleast_2d(X)
            Y = np.atleast_2d(Y)
            x_exp = X[:, np.newaxis, :]  # shape: (n_samples_x, 1, n_features)
            y_exp = Y[np.newaxis, :, :]  # shape: (1, n_samples_y, n_features)
            S = np.sum((x_exp - y_exp) ** 2, axis=2)
            return S

    if kernel == 'rbf':
        default_gamma = 20
        def kernel(x, y, gamma=default_gamma):
            return myexp(-sq_dist(x, y) * gamma)
        return kernel, {'gamma': default_gamma}
    elif kernel == 'linear':
        def kernel(x, y):
            return x @ y.T
        return kernel, {}
    elif kernel == 'matern1-2':
        default_gamma = 10
        def kernel(x, y, gamma=default_gamma):
            return myexp(-mysqrt(sq_dist(x, y)) * gamma)
        return kernel, {'gamma': default_gamma}
    elif kernel == 'matern3-2':
        default_gamma = 10
        def kernel(x, y, gamma=default_gamma):
            temp = mysqrt(sq_dist(x, y))
            return (1 + mysqrt(3) * temp * gamma) * myexp(-mysqrt(3) *temp * gamma)
        return kernel, {'gamma': default_gamma}
    elif kernel == 'matern5-2':
        default_gamma = 10
        def kernel(x, y, gamma=default_gamma):
            temp = mysqrt(sq_dist(x, y))
            return (1 + mysqrt(5) * temp * gamma + 5 * temp**2 * gamma ** 2 / 3) * myexp(- mysqrt(5) * temp * gamma)
        return kernel, {'gamma': default_gamma}
    else:
        raise ValueError("Invalid kernel")
