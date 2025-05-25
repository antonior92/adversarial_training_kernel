import torch
import torch.nn as nn
from advkern.pgd import PGD
import torch
from torch import nn
import matplotlib.pyplot as plt



class LinearNet(nn.Module):
    def __init__(self, randomfeatures):
        super().__init__()
        self.randomfeatures = randomfeatures
        self.randomfeatures.fit()
        self.linear = nn.Linear(randomfeatures.R, 1, bias=False)

    def forward(self, x):
        x = self.randomfeatures.transform(x)
        x = self.linear(x)
        return x


class RBFRandomFourierFeatures(nn.Module):
    '''
    compute R random fourier features for each sample in X
    '''

    def __init__(self, R, p, sigmasq=1, track_grads=False):
        '''
        R: number of random fourier features
        p: number of input features
        '''
        super().__init__()
        self.R = R
        self.p = p
        self.sigmasq = sigmasq
        self.track_grads = track_grads

    def fit(self):
        self.W = torch.normal(
            mean=0.0,
            std=1.0 / torch.sqrt(torch.tensor(self.sigmasq)),
            size=(self.R, self.p),
        )  # R x p
        self.b = torch.rand(self.R) * 2 * torch.pi  # (R,)

        self.W = nn.Parameter(self.W, requires_grad=self.track_grads)
        self.b = nn.Parameter(self.b, requires_grad=self.track_grads)

    def transform(self, X):
        # N x p @ p x R + R x 1
        Z = X @ self.W.T + self.b
        Z = torch.sqrt(torch.tensor(2 / self.R)) * torch.cos(Z)
        return Z

    def fit_transform(self, X):
        '''
        note: W has R N(0, I * (1 / sigma_sq)) vectors on its rows
        '''
        self.W = torch.normal(
            mean=0.0,
            std=1.0 / torch.sqrt(torch.tensor(self.sigmasq)),
            size=(self.R, self.p),
        )  # R x p
        self.b = torch.rand(self.R) * 2 * torch.pi  # (R,)

        self.W = nn.Parameter(self.W, requires_grad=self.track_grads)
        self.b = nn.Parameter(self.b, requires_grad=self.track_grads)

        # N x p @ p x R + R x 1
        Z = X @ self.W.T + self.b  # n x R
        Z = torch.sqrt(torch.tensor(2 / self.R)) * torch.cos(Z)
        return Z



if __name__ == "__main__":
    # sample N points in the range [a, b); a <= b
    a, b = 0, 1
    N = 50
    X = (a - b) * torch.rand(N, 1) + b

    # generate noisy observations of sin(4 * pi * x)
    mu, sigmasq = 0, 0.01
    y = torch.sin(4 * torch.pi * X).reshape(-1, 1) + torch.normal(
        mean=mu, std=(sigmasq**0.5), size=(N, 1)
    )

    # generate a smooth curve for sin(4 * pi * x) over the same range
    X_eval = torch.linspace(a, b, 1000).reshape(-1, 1)
    y_eval = torch.sin(4 * torch.pi * X_eval)

    device = torch.device(
        'mps'
        if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    R = 500
    p = X.shape[1]
    randomfeatures = RBFRandomFourierFeatures(R, p, sigmasq=0.05, track_grads=False)
    randomfeatures.to(device)

    model = LinearNet(randomfeatures)
    model.to(device)

    # count the number of trainable parameters
    for name, param in model.named_parameters():
        print(
            f'param name = {name}, param shape = {param.shape}, requires grad: {param.requires_grad}'
        )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = int(3e3)
    model.train()

    X = X.to(device)
    y = y.to(device)
    attack = PGD(model, loss_fn)
    for epoch in range(epochs):
        optimizer.zero_grad()
        adv = attack(X, y)
        predictions = model(adv)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    plt.plot(X.cpu().numpy(), y.cpu().numpy(), 'o', label='Data')  # data
    plt.plot(X_eval, y_eval, label='sin(x)', color='red')  # truth

    X_eval = X_eval.to(device)
    plt.plot(
        X_eval.cpu().numpy(),
        model(X_eval).detach().cpu().numpy(),
        label='Predictions',
        color='green',
    )  # predictions
    plt.show()
