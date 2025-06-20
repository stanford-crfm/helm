import io
import pickle
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from pyrelimri.tetrachoric_correlation import tetrachoric_corr
from tqdm import tqdm
from huggingface_hub import login, HfApi
import torch
from torch.distributions import Bernoulli
from typing import Any, Tuple, Optional, Callable, List, Union, Dict


def trainer(parameters: List[torch.Tensor],
            optim: torch.optim.Optimizer,
            closure: Callable[[],
                              torch.Tensor]) -> List[torch.Tensor]:
    pbar = tqdm(range(100))
    loss: torch.Tensor
    for iteration in pbar:
        if iteration > 0:
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        loss = optim.step(closure)
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(
                    prev - curr,
                    p=2).item() for prev,
                curr in zip(
                    previous_parameters,
                    parameters))
            grad_norm = sum(torch.norm(p.grad, p=2).item()
                            for p in parameters if p.grad is not None)
            pbar.set_postfix(
                {"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss})
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                break
    return parameters


# The following function is copied verbatim from:
# https://github.com/hardy-education/pymokken/blob/main/scalability_coefs.py
# under the MIT license:

# Copyright (c) 2025, Michael Hardy.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# “Software”), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


def scalability_coefs(X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """
    Compute item-level scalability coefficients (Hi and Zi) using simplified approach,
    which does not include standard errors or confidence intervals.
    (Loevinger, 1948; Mokken, 1971; Molenaar and Sijtsma, 2000; Sijtsma and Molenaar, 2002)

    This function computes:
    - Hi: Item-level H coefficients (scalability of each item with rest of scale)
    - Zi: Item-level Z-scores (standardized Hi coefficients)
    - H: Overall scale H coefficient (scalar)
    - Z: Overall scale Z-score (scalar)
    - Hij: Item-pair H coefficients (matrix of shape (n_items, n_items))
    - Zij: Item-pair Z-scores (matrix of shape (n_items, n_items))

    Parameters
    ----------
    X : array-like of shape (n_subjects, n_items)
        Data matrix containing item responses. Should be integer-valued.
        Missing values are handled by listwise deletion.

    Returns
    -------
    dict
        Dictionary containing:
        - 'Hi': Item-level H coefficients (array of length n_items)
        - 'Zi': Item-level Z-scores (array of length n_items)
        - 'H': Overall scale H coefficient (scalar)
        - 'Z': Overall scale Z-score (scalar)
        - 'Hij': Item-pair H coefficients (matrix of shape (n_items, n_items))
        - 'Zij': Item-pair Z-scores (matrix of shape (n_items, n_items))

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(0, 4, (100, 5))
    >>> result = scalability_simple(X)
    >>> print(f"Item coefficients: {result['Hi']}")
    >>> print(f"Overall coefficient: {result['H']:.3f}")
    """
    # Convert input to numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=float)

    # Handle missing data with listwise deletion
    if np.any(np.isnan(X)):
        complete_cases = ~np.any(np.isnan(X), axis=1)
        X = X[complete_cases]
        if X.shape[0] < 5:
            raise ValueError(
                "Insufficient complete cases after removing missing data")

    # Convert to integers
    X = X.astype(int)

    # Validate input
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 items")
    if X.shape[0] < 5:
        raise ValueError("X must have at least 5 subjects")

    n_subjects, n_items = X.shape

    # Check for zero variance, handle with listwise deletion
    if np.any(np.var(X, axis=0) == 0):
        complete_cases = ~np.any(np.var(X, axis=0) == 0, axis=1)
        X = X[complete_cases]
        if X.shape[0] < 5:
            raise ValueError(
                "Insufficient complete cases after removing zero variance items")

    # Compute H scaling (Loevinger, 1948; Mokken, 1971) using simple method
    # Compute covariance matrices
    S = np.cov(X, rowvar=False)  # Item covariance matrix
    X_sorted = np.sort(X, axis=0)  # Sort each item independently
    Smax = np.cov(X_sorted, rowvar=False)  # Maximum possible covariance

    # Compute Hij matrix (item-pair coefficients)
    Hij = S / Smax
    np.fill_diagonal(Hij, 0)  # Zero out diagonal

    # Compute Hi coefficients (item-level)
    S_offdiag = S.copy()
    Smax_offdiag = Smax.copy()
    np.fill_diagonal(S_offdiag, 0)
    np.fill_diagonal(Smax_offdiag, 0)

    # for future reference:
    Hij = np.divide(
        S_offdiag,
        Smax_offdiag,
        out=np.zeros_like(S_offdiag),
        where=Smax_offdiag != 0)
    Hi = np.sum(Hij, axis=1)

    # Compute overall H coefficient
    H = np.sum(S_offdiag) / np.sum(Smax_offdiag)

    # Compute Z-standardized scaling using simple method
    # (Mokken, 1971; Molenaar and Sijtsma, 2000; Sijtsma and Molenaar, 2002)
    # Only appropriate for testing lowerbound = 0.
    # Item variances, unweighted and unbiased
    var_vec = np.var(X, axis=0, ddof=1)
    Sij = np.outer(var_vec, var_vec)  # Outer product of variances

    # Item-pair Z-standardized scaling coefficients
    Zij = np.divide(S * np.sqrt(n_subjects - 1), np.sqrt(Sij),
                    out=np.zeros_like(S_offdiag), where=Sij != 0)
    np.fill_diagonal(Zij, 0)  # Zero diagonal

    # Item-level Z-standardized scaling
    Sij_for_z = Sij.copy()
    np.fill_diagonal(Sij_for_z, 0)

    Zi = np.divide(
        np.sum(S_offdiag, axis=1) * np.sqrt(n_subjects - 1),
        np.sqrt(np.sum(Sij_for_z, axis=1)),
        out=np.zeros(n_items),
        where=np.sum(Sij_for_z, axis=1) != 0,
    )

    # Overall Z-standardized scaling (divided by 2 because the matrix is
    # symmetric, I think)
    sum_S = np.sum(S_offdiag) / 2.0
    sum_Sij = np.sum(Sij_for_z) / 2.0
    Z = (sum_S * np.sqrt(n_subjects - 1)) / \
        np.sqrt(sum_Sij) if sum_Sij != 0 else 0.0

    return {"Hi": Hi, "Zi": Zi, "H": H, "Z": Z, "Hij": Hij, "Zij": Zij}


def raw_item_total_correlations(X: np.ndarray) -> List[float]:
    total = X.sum(axis=1)
    Xc = X - X.mean(axis=0)
    Tc = total - total.mean()
    numer = (Xc * Tc[:, None]).sum(axis=0)
    denom = np.sqrt((Xc**2).sum(axis=0) * (Tc**2).sum())
    raw_r = numer / denom
    return raw_r.tolist()


if __name__ == "__main__":
    benchmark: str = "lite"
    scenario: str = "gsm"

    # load information from long table
    long_path: str = hf_hub_download(
        repo_id="stair-lab/reeval_data_public",
        repo_type="dataset",
        filename="long.pkl")
    with open(long_path, "rb") as f:
        long: Any = pickle.load(f)
    sub_long: pd.DataFrame = long[(long["benchmark"] == benchmark) & (
        long["scenario"] == scenario)].copy()
    sub_long = sub_long.drop_duplicates(
        subset=[
            "instance_id",
            "train_trial_index",
            "perturbation.name"]).reset_index(
        drop=True)
    sub_long = sub_long[["instance_id",
                         "train_trial_index",
                         "perturbation.name",
                         "input.text"]]

    # load resmat
    resmat_path: str = hf_hub_download(
        repo_id="stair-lab/reeval_data_public",
        repo_type="dataset",
        filename="resmat.pkl")
    with open(resmat_path, "rb") as f:
        resmat: pd.DataFrame = pickle.load(f)
    sub_mask: pd.Series = (
        resmat.columns.get_level_values("benchmark") == benchmark) & (
        resmat.columns.get_level_values("scenario") == scenario)
    sub_resmat: pd.DataFrame = resmat.loc[:, sub_mask]
    sub_resmat = sub_resmat.dropna(axis=0, how="all")
    questions: pd.Index = sub_resmat.columns.get_level_values("input.text")
    data: np.ndarray = sub_resmat.values
    n_test_takers: int
    n_questions: int
    n_test_takers, n_questions = data.shape

    # 1. tetrachoric correlation
    print("1. tetrachoric correlation")
    corr_matrix = np.zeros((n_questions, n_questions))
    for i in tqdm(range(n_questions)):
        for j in range(i, n_questions):
            r = tetrachoric_corr(data[:, i], data[:, j])
            corr_matrix[i, j] = corr_matrix[j, i] = r
    tetrachoric = np.nanmean(corr_matrix, axis=1)

    # 2. 2PL IRT discriminant
    print("2. 2PL IRT discriminant")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_tensor: torch.Tensor = torch.tensor(data, device=device)
    z: torch.Tensor = torch.zeros(
        n_questions,
        requires_grad=True,
        device=device)
    a: torch.Tensor = torch.ones(n_questions, requires_grad=True, device=device)
    optim: torch.optim.Optimizer = torch.optim.LBFGS(
        [z, a], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
    )
    thetas: torch.Tensor = torch.randn(150, n_test_takers, device=device)

    def closure():
        optim.zero_grad()
        probs = torch.sigmoid(
            (thetas[:, :, None] + z[None, None, :]) * a[None, None, :])
        loss = -(Bernoulli(probs=probs).log_prob(data_tensor)
                 ).mean() + 0.01 * (a - 1).pow(2).mean()
        loss.backward()
        return loss

    z, a = trainer([z, a], optim, closure)
    a = a.detach().cpu().numpy()

    # 3. scalability coefficients
    print("3. scalability coefficients")
    scalability_coeff_results: Dict[str, Any] = scalability_coefs(data)
    scalability_coeff: np.ndarray = scalability_coeff_results["Zij"].mean(0)

    # 4. item-total correlation
    print("4. item-total correlation")
    item_total_corr: List[float] = raw_item_total_correlations(data)

    # merge the two data
    validity_metrics: pd.DataFrame = pd.DataFrame(
        {
            "input.text": questions,
            "tetrachoric": tetrachoric,
            "2pl_irt_discriminant": a,
            "scalability_coeff": scalability_coeff,
            "item_total_corr": item_total_corr,
        }
    )
    merged: pd.DataFrame = validity_metrics.merge(
        sub_long, on="input.text", how="inner")
    merged = merged.where(merged.notna(), None)

    # create dict, upload to HF
    validity_dict: Dict[Tuple[str, Optional[str], int], Dict[str, float]] = {
        (row["instance_id"], row["perturbation.name"], row["train_trial_index"]): {
            "tetrachoric": row["tetrachoric"],
            "2pl_irt_discriminant": row["2pl_irt_discriminant"],
            "scalability_coeff": row["scalability_coeff"],
            "item_total_corr": row["item_total_corr"],
        }
        for _, row in merged.iterrows()
    }
    cleaned_validity_dict: Dict[Tuple[str, Optional[str], int], Dict[str, float]] = {
        (inst_id, None if pd.isna(perturb) else perturb, trial_idx): valid
        for (inst_id, perturb, trial_idx), valid in validity_dict.items()
    }
    validity_df = (
        pd.DataFrame.from_dict(cleaned_validity_dict, orient="index")
        .rename_axis(index=["instance_id", "perturbation", "train_trial_index"])
        .reset_index()
    )
    buffer: io.BytesIO = io.BytesIO()
    validity_df.to_parquet(buffer, index=False)
    buffer.seek(0)

    login()
    api: HfApi = HfApi()
    api.upload_file(
        path_or_fileobj=buffer,
        path_in_repo="validity.parquet",
        repo_id="stair-lab/helm_display_validity",
        repo_type="dataset",
    )