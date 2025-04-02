from pgd import PGD
from randomfeatures import RBFRandomFourierFeatures
import torch
from torch import nn
import matplotlib.pyplot as plt


class LinearNet(nn.Module):
    def __init__(self, randomfeatures):
        super().__init__()
        self.randomfeatures = randomfeatures
        self.randomfeatures.fit()
        self.linear = nn.Linear(R, 1, bias=False)

    def forward(self, x):
        x = self.randomfeatures.transform(x)
        x = self.linear(x)
        return x


# sample N points in the range [a, b); a <= b
a, b = 0, 1
N = 1000
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
R = 1000
randomfeatures = RBFRandomFourierFeatures(
    R, X.shape[1], track_grads=False, device=device
)

model = LinearNet(randomfeatures)
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = int(3e3)
model.train()

# count the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

X = X.to(device)
y = y.to(device)
attack = PGD(model, loss_fn, a, b)
for epoch in range(epochs):
    optimizer.zero_grad()
    adv = attack(X, y)
    predictions = model(adv)
    loss = loss_fn(predictions, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


plt.plot(X.cpu().numpy(), y.cpu().numpy(), 'o', label='Data')  # data
plt.plot(X_eval, y_eval, label='sin(x)', color='red')  # truth

# sort X in ascending order
sorted_indices = torch.argsort(X, dim=0).squeeze()
X_sorted = X[sorted_indices]

plt.plot(
    X_sorted.cpu().numpy(),
    model(X_sorted).detach().cpu().numpy(),
    label='Predictions',
    color='green',
)  # predictions
plt.show()

print("finished")
