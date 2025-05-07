import torch
import torch.nn as nn
from pgd import PGD
import matplotlib.pyplot as plt
from pgd_demo_1d import DummyModel


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        return torch.tanh(4 * pred[0] + 4 * pred[1]) + max(0.4 * pred[0] ** 2, 1) + 1


def loss_fn(grid):
    return (
        torch.tanh(4 * grid[..., 0] + 4 * grid[..., 1])
        + torch.maximum(0.4 * grid[..., 0] ** 2, torch.tensor(1.0))
        + 1
    )


def circle(r, num_points=100, center=(0, 0)):
    theta = torch.linspace(0, 2 * torch.pi, num_points)
    x1 = r * torch.cos(theta) + center[0]
    x2 = r * torch.sin(theta) + center[1]
    return x1, x2


def square(r, center=(0, 0)):
    x1 = torch.tensor([-r, r, r, -r, -r]) + center[0]
    x2 = torch.tensor([-r, -r, r, r, -r]) + center[1]
    return x1, x2


def rotated_square(r, center=(0, 0)):
    x1 = torch.tensor([-r, 0, r, 0, -r]) + center[0]
    x2 = torch.tensor([0, -r, 0, r, 0]) + center[1]
    return x1, x2


if __name__ == "__main__":
    model = DummyModel()

    # --------------------------------- Demo for Loss ---------------------------------

    loss = Loss()
    x1_init, x2_init = 0.0, -0.1  # must be float
    adv_radius = 3
    step_size = 0.8
    nsteps = 50
    p = 2

    attack = PGD(
        model=model,
        loss_fn=loss,
        p=p,
        adv_radius=adv_radius,
        step_size=step_size,
        nsteps=nsteps,
    )

    X = torch.tensor([[x1_init, x2_init]])
    y = torch.tensor(0.0)
    advs, losses = attack(X, y, debug=True)
    advs = [adv.squeeze() for adv in advs]

    x1s = torch.linspace(-7, 7, 100)
    x2s = torch.linspace(-7, 7, 100)
    x1, x2 = torch.meshgrid(x1s, x2s, indexing='ij')
    loss = loss_fn(torch.stack([x1, x2], dim=-1))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x1, x2, loss, alpha=0.6, cmap='cividis')
    ax.set_xlabel('x1'), ax.set_ylabel('x2'), ax.set_zlabel('loss')
    advs = torch.stack(advs)
    ax.scatter(
        advs[:, 0],  # x1
        advs[:, 1],  # x2
        losses,
        color='blue',
        label='intermediate advs',
        s=60,  # size of points
        edgecolors='black',
        linewidths=0.5,
    )

    if p == 1:
        x1, x2 = rotated_square(adv_radius, center=(x1_init, x2_init))
    elif p == 2:
        x1, x2 = circle(adv_radius, center=(x1_init, x2_init))
    elif p == torch.inf:
        x1, x2 = square(adv_radius, center=(x1_init, x2_init))

    if p in [1, 2, torch.inf]:
        max_loss = torch.max(loss).item()
        heights = torch.linspace(0, max_loss, steps=4)
        for h in heights:
            ax.plot(
                x1,
                x2,
                [h] * len(x1),
                color='blue',
                alpha=0.5,
                label=f'adv radius = {adv_radius}' if h == heights[0] else None,
            )
    ax.legend()
    plt.show()
