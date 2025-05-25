import torch
import torch.nn as nn
from advkern.pgd import PGD
import matplotlib.pyplot as plt


class Loss1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        return (
            torch.maximum(torch.tensor(0), (3 * pred - 2.3) ** 3 + 1) ** 2
            + torch.maximum(torch.tensor(0), (-3 * pred + 0.7) ** 3 + 1) ** 2
        )


class Loss2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        return pred**4 + 0.1


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X.squeeze()


if __name__ == "__main__":
    model = DummyModel()

    # --------------------------------- Demo for Loss 1 ---------------------------------

    loss1 = Loss1()
    x_init = 0.5  # must be float
    adv_radius = 0.4
    step_size = 0.01
    nsteps = 10
    p = 1

    attack = PGD(
        model=model,
        loss_fn=loss1,
        p=p,
        adv_radius=adv_radius,
        step_size=step_size,
        nsteps=nsteps,
    )

    X = torch.tensor([[x_init]])
    y = torch.tensor(0.0)
    advs, losses = attack(X, y, debug=True)
    advs = [adv.squeeze() for adv in advs]

    x1_arr = torch.linspace(0, 1, 1000).reshape(-1, 1)
    plt.plot(x1_arr, loss1(x1_arr, y), label="loss1")
    for i, (adv, loss) in enumerate(zip(advs, losses)):
        plt.plot(adv, loss, 'o', color=plt.cm.viridis(i / len(advs)))
    plt.axvline(x=x_init - adv_radius, color='red', linestyle='--')
    plt.axvline(x=x_init + adv_radius, color='red', linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()

    # --------------------------------- Demo for Loss 2 ---------------------------------

    loss2 = Loss2()
    x_init = 0.2  # must be float
    adv_radius = 0.2
    step_size = 0.02
    nsteps = 10
    p = 2

    attack = PGD(
        model=model,
        loss_fn=loss2,
        p=p,
        adv_radius=adv_radius,
        step_size=step_size,
        nsteps=nsteps,
    )

    X = torch.tensor([[x_init]])
    y = torch.tensor(0.0)
    advs, losses = attack(X, y, debug=True)
    advs = [adv.squeeze() for adv in advs]

    x2_arr = torch.linspace(-1, 1, 1000).reshape(-1, 1)
    plt.plot(x2_arr, loss2(x2_arr, y), label="loss2")
    for i, (adv, loss) in enumerate(zip(advs, losses)):
        plt.plot(adv, loss, 'o', color=plt.cm.viridis(i / len(advs)))
    plt.axvline(x=x_init - adv_radius, color='red', linestyle='--')
    plt.axvline(x=x_init + adv_radius, color='red', linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()
