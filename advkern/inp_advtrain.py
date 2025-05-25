import copy

import torch
from torch import nn as nn

from advkern.pgd import PGD


def fine_tunne_advtrain(model_base, X, y, lr=1e-2, nepochs=300, p=2, adv_radius=0.05, step_size=0.002,nsteps=100):
    model = copy.deepcopy(model_base)
    pgd = PGD(
        model=model,
        p=p,
        adv_radius=adv_radius,
        step_size=step_size,
        nsteps=nsteps
    )
    X = torch.tensor(X)
    y = torch.tensor(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    myloss = nn.MSELoss()
    for epoch in range(nepochs):
        optimizer.zero_grad()
        X_adv = pgd(X, y)
        y_adv =  model(X_adv)
        loss = myloss(y_adv, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{nepochs}] | Loss = {loss.item():.4f}")

    return model
