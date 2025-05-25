from advkern.inp_advtrain import fine_tunne_advtrain
from advkern.np2torch import KernelRidgeModel
from advkern.data import get_curve
from advkern.kernels import get_kernel
from advkern.pgd import PGD
import torch
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

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
    plt.show()
