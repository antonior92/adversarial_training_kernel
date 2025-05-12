from onedim_curve_fitting import get_curve, get_kernel, get_estimate, valid_kernels
from pgd import PGD
import torch
import torch.nn as nn
import numpy as np
from kernel_advtrain import kernel_adversarial_training, mkl_adversarial_training, AdvKernelTrain, LinearAdvFourierFeatures
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import copy



class KernelRidgeModel(nn.Module):
    def __init__(self, kernel, dual_coef_, X_fit, best_params=None):
        super().__init__()
        self.kernel = kernel
        self.dual_coef_ = torch.nn.Parameter(torch.tensor(dual_coef_))
        self.X_fit = torch.tensor(X_fit)
        self.best_params = best_params

    def forward(self, x):
        kernel, kernel_params = get_kernel(kernel=self.kernel, usetorch=True)
        if self.best_params is not None:
            for k, v in self.best_params.items():
                if k in kernel_params:
                    kernel_params[k] = v
        else:
            kernel_params = kernel_params
        K = kernel(x, self.X_fit, **kernel_params)
        return K @ self.dual_coef_

    @classmethod
    def from_sklearn(cls, kernel_name, krr_sklearn, best_params=None):
        if isinstance(krr_sklearn, GridSearchCV):
            return cls.from_sklearn(kernel_name, krr_sklearn.best_estimator_, best_params=krr_sklearn.best_params_)
        elif isinstance(krr_sklearn, KernelRidge):
            return cls(kernel_name, krr_sklearn.dual_coef_, krr_sklearn.X_fit_, best_params=best_params)
        elif isinstance(krr_sklearn, AdvKernelTrain):
            return cls.from_sklearn(kernel_name, krr_sklearn.model_, best_params=best_params)
        else:
            raise NotImplementedError()

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    kernel_name='rbf'
    kernel, kernel_params = get_kernel(kernel_name)

    rng = np.random.RandomState(6)
    X, y, X_plot, y_plot = get_curve(rng, 100, curve=3)
    kr = GridSearchCV(
            KernelRidge(kernel=kernel, **kernel_params),
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]},
        )

    kr.fit(X, y)

    X_test, y_test, _X_plot, y_plot = get_curve(rng, 5, curve=3, std_noise=0)
    y_pred = kr.predict(X_test)


    krrtorch = KernelRidgeModel.from_sklearn(kernel_name, kr)
    pgd = PGD(
        model=krrtorch,
        p=2,
        adv_radius=0.05,
        step_size=0.002,
        nsteps=100,
    )
    x_attack = pgd(torch.tensor(X_test), torch.tensor(y_test))
    y_attack = krrtorch(x_attack)
    plt.figure()
    plt.scatter(X_test, y_pred, c="k", label="Data", marker='o', zorder=1, edgecolors=(0, 0, 0))
    plt.scatter(x_attack.detach().numpy(), y_attack.detach().numpy(), c="k", label="Data", marker='x')
    plt.plot(X_plot, y_plot, c='k', ls=':', label='True')
    y_pred_plot = krrtorch(torch.tensor(X_plot)).detach().numpy()
    plt.plot(X_plot, y_pred_plot)
    plt.savefig(f'img/adv_attacks_before.pdf')
    plt.show()


    adv_train_model = fine_tunne_advtrain(krrtorch, X_plot, y_plot)

    print('adv train')
    pgd = PGD(
        model=krrtorch,
        p=2,
        adv_radius=0.05,
        step_size=0.002,
        nsteps=100,
    )
    x_attack = pgd(torch.tensor(X_test), torch.tensor(y_test))
    y_attack = krrtorch(x_attack)
    y_pred = krrtorch(torch.tensor(X_test)).detach().numpy()
    plt.figure()
    plt.scatter(X_test, y_pred, c="k", label="Data", marker='o', zorder=1, edgecolors=(0, 0, 0))
    plt.scatter(x_attack.detach().numpy(), y_attack.detach().numpy(), c="k", label="Data", marker='x')
    plt.plot(X_plot, y_plot, c='k', ls=':', label='True')
    y_pred_plot = krrtorch(torch.tensor(X_plot)).detach().numpy()
    plt.plot(X_plot, y_pred_plot)
    plt.savefig(f'img/adv_attacks_after.pdf')
    plt.show()
