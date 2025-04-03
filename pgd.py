import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class PGD(nn.Module):
    '''
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    distance measure: Linf

    arguments:
        - model (nn.Module): model to attack
        - loss_fn (callable): loss function
        - a (float): lower bound of the input
        - b (float): upper bound of the input
        - eps (float): maximum perturbation
        - alpha (float): step size
        - steps (int): number of steps
        - random_start (bool): using random initialization of delta

    shape:
        - X: (N, C, H, W) where N = batch size, C = number of channels, H = height and W = width
        - y: (N,)
        - output: (N, C, H, W)
    '''

    def __init__(
        self,
        model,
        loss_fn,
        a,
        b,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        random_start=False,
    ):
        super(PGD, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.a = a
        self.b = b
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, X, y):
        X_adv = X.clone()
        X_adv.requires_grad = True

        if self.random_start:
            # starting at a uniformly random point
            X_adv = X_adv + torch.empty_like(X_adv).uniform_(-self.eps, self.eps)
            X_adv = torch.clamp(X_adv, min=self.a, max=self.b)

        for _ in range(self.steps):
            loss = self.loss_fn(self.model(X_adv), y)
            grad = torch.autograd.grad(
                loss, X_adv, retain_graph=False, create_graph=False
            )[0]

            # update X_adv with the gradient step
            X_adv = X_adv + self.alpha * grad.sign()

            # project X_adv to the epsilon-ball around X
            X_adv = torch.clamp(X_adv, X - self.eps, X + self.eps)

            # ensure X_adv stays within the valid input range [a, b]
            X_adv = torch.clamp(X_adv, min=self.a, max=self.b)

        return X_adv


if __name__ == "__main__":

    def f(x):
        return np.sin(x)

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
        a=a,
        b=b,
        eps=5,
        alpha=0.01,
        steps=100,
        random_start=False,
    )

    for i in range(10):
        # x = torch.tensor((a - b) * np.random.rand(1, 1) + b, dtype=torch.float32).to(device)
        x = torch.tensor([10], dtype=torch.float32).to(device)
        y = model(x)
        x_adv = pgd_attack(
            x.reshape(1, 1, 1, 1), f(x.detach().cpu()).to(device)
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
    print("finished")
