import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pgd import PGD
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from randomfeatures import RBFRandomFourierFeatures
from kernel_advtrain import LinearNet


def f(x):
    return np.sin(x)


def test_pgd():
    # sample N points in the range [a, b); a <= b
    a, b = 3, 10
    N = 50
    X = (a - b) * np.random.rand(N, 1) + b
    # generate noisy observations of sin(x)
    mu, sigmasq = 0, 0.1
    y = f(X).reshape(-1, 1) + np.random.normal(
        loc=mu, scale=np.sqrt(sigmasq), size=(N, 1)
    )

    # generate a smooth curve for sin(x) over the same range
    X_eval = np.linspace(a, b, 1000).reshape(-1, 1)
    y_eval = f(X_eval)

    # define a simple MLP model
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MLP, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.model(x)

    device = torch.device(
        'mps'
        if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    input_dim = 1
    hidden_dim = 2**6
    output_dim = 1
    model = MLP(input_dim, hidden_dim, output_dim)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    epochs = int(1e3)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    model.eval()
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_eval_tensor).cpu().numpy()

    # Plot the adversarial examples
    plt.plot(X_tensor.cpu().numpy(), y_tensor.cpu().numpy(), 'o', label='Data')  # data
    plt.plot(X_eval, y_eval, label='sin(x)', color='red')  # truth
    plt.plot(X_eval, y_pred, label='MLP Fit', color='blue')  # model prediction
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    pgd_attack = PGD(
        model=model,
        loss_fn=nn.MSELoss(),
        adv_radius=0.5,
        step_size=0.01,
        nsteps=100,
    )

    for i in range(10):
        x = torch.tensor((a - b) * np.random.rand(1, 1) + b, dtype=torch.float32).to(
            device
        )
        # x = torch.tensor([10], dtype=torch.float32).to(device)
        y = model(x)
        x_adv = pgd_attack(
            x, f(x.detach().cpu()).to(device)
        ).reshape(1)
        y_adv = model(x_adv)

        x = x.cpu().item()
        y = y.detach().cpu().item()
        x_adv = x_adv.cpu().item()
        y_adv = y_adv.detach().cpu().item()

        print(
            f'x = {x:.4f}, pred y = {y:.4f}, true y = {f(x):.4f}, delta y = {np.abs(y - f(x)):.4f}'
        )
        print(
            f'x_adv = {x_adv:.4f}, pred y_adv = {y_adv:.4f}, true y = {f(x):.4f}, delta y = {np.abs(y_adv - f(x)):.4f}'
        )
        print(
            f'diff = {(np.abs(y - f(x_adv)) - (np.abs(y - f(x)))):.4f} (this should be > 0)\n'
        )


def test_randomfeatures():
    # sample N points in the range [a, b); a <= b
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

    R = 100000
    randomfeatures = RBFRandomFourierFeatures(R, X.shape[1])
    Z = randomfeatures.fit_transform(torch.from_numpy(X.astype(np.float32))).numpy()
    model = Ridge(alpha=0.1)
    model.fit(Z, y)
    y_r = model.predict(
        randomfeatures.transform(torch.from_numpy(X_eval.astype(np.float32)))
    )

    plt.plot(X, y, 'o')  # data
    plt.plot(X_eval, y_eval, label='sin(x)', color='red')  # truth
    plt.plot(X_eval, y_kr, label='kernel ridge', color='blue')  # kernel ridge fit
    plt.plot(X_eval, y_r, label=f'rff + ridge, R = {R}', color='green')  # rff fit
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def test_multidim():
    N, p = 100, 3
    X = np.random.rand(N, p)
    R = 1000
    randomfeatures = RBFRandomFourierFeatures(R, X.shape[1])
    Z = randomfeatures.fit_transform(torch.from_numpy(X.astype(np.float32))).numpy()
    assert Z.shape == (N, R)


def test_linear():
    N, p = 100, 3
    X = np.random.rand(N, p)
    R = 1000
    randomfeatures = RBFRandomFourierFeatures(R, X.shape[1])
    Z = randomfeatures.fit_transform(torch.from_numpy(X.astype(np.float32))).numpy()
    model = LinearNet(input_dim=R, output_dim=1)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    model(torch.tensor(Z))
    print("finished")
