import numpy as np
import pandas as pd

from hppca import fit_hppca


def main():
    rng = np.random.default_rng(1)
    rows = []
    for subject_id in range(10):
        for visit_time in [0.0, 30.0, 90.0]:
            values = rng.normal(size=4)
            if rng.random() < 0.25:
                values[rng.integers(0, values.size)] = np.nan
            rows.append(
                {
                    "subject_id": subject_id,
                    "visit_time": visit_time,
                    "feature_0": values[0],
                    "feature_1": values[1],
                    "feature_2": values[2],
                    "feature_3": values[3],
                }
            )

    df = pd.DataFrame(rows)
    fit = fit_hppca(
        Y_obs=df,
        d1=1,
        d2=1,
        kernel_method="gp_rbf_single_ell",
        init_method="random",
        max_iter=3,
        seed=1,
        subject_id_col="subject_id",
        visit_time_col="visit_time",
        return_latent_dataframe=True,
    )

    W1, W2, sigma2, ell = fit[:4]
    latent_df = fit[-1]
    print("W1 shape:", W1.shape)
    print("W2 shape:", W2.shape)
    print("sigma2:", sigma2)
    print("ell:", ell)
    print(latent_df.head())


if __name__ == "__main__":
    main()
