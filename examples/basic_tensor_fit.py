import numpy as np

from hppca import fit_hppca


def main():
    rng = np.random.default_rng(0)
    n, J, p = 20, 4, 6
    Y = rng.normal(size=(n, J, p))
    Y -= Y.mean(axis=(0, 1), keepdims=True)

    missing = rng.random(size=Y.shape) < 0.2
    Y[missing] = np.nan

    (
        W1,
        W2,
        sigma2,
        ell,
        *_,
    ) = fit_hppca(
        Y_obs=Y,
        d1=1,
        d2=1,
        kernel_method="gp_iid",
        init_method="random",
        max_iter=3,
        seed=0,
    )

    print("W1 shape:", W1.shape)
    print("W2 shape:", W2.shape)
    print("sigma2:", sigma2)
    print("ell:", ell)


if __name__ == "__main__":
    main()
