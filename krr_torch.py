import numpy as np
import torch
from onedim_curve_fitting import get_kernel




if __name__ == "__main__":
    # In pytorch
    def get_kernel(X, Y, gamma=12):
        return torch.exp(-torch.sqrt(sq_dist(X, Y)) * gamma)

    X = torch.tensor([[1.0, 2.0],[3.0, 4.0]])
    Y = torch.tensor([[1.0, 2.0],[3.0, 4.0]])

    x = X.unsqueeze(1)  # shape: (n_samples_x, 1, n_features)
    y = Y.unsqueeze(0)  # shape: (1, n_samples_y, n_features)
    S = torch.sum((x - y) ** 2, dim=2)


    print(S)
    print(get_kernel(X, Y))