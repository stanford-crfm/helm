import io
import pickle
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from pyrelimri.tetrachoric_correlation import tetrachoric_corr
from tqdm import tqdm
from huggingface_hub import login, HfApi

if __name__ == "__main__":
    benchmark = "lite"
    scenario = "gsm"
    
    # load information from long table
    long_path = hf_hub_download(
        repo_id="stair-lab/reeval_data_public",
        repo_type="dataset",
        filename="long.pkl"
    )
    with open(long_path, "rb") as f:
        long = pickle.load(f)
    sub_long = long[(long['benchmark'] == benchmark) & (long['scenario']  == scenario)].copy()
    sub_long = sub_long.drop_duplicates(subset=['instance_id', 'train_trial_index', 'perturbation.name']).reset_index(drop=True)
    sub_long = sub_long[['instance_id', 'train_trial_index', 'perturbation.name', 'input.text']]
    
    # calculate tetrchoric from resmat
    resmat_path = hf_hub_download(
        repo_id="stair-lab/reeval_data_public",
        repo_type="dataset",
        filename="resmat.pkl"
    )
    with open(resmat_path, "rb") as f:
        resmat = pickle.load(f)
    sub_mask = (
        (resmat.columns.get_level_values("benchmark") == benchmark) &
        (resmat.columns.get_level_values("scenario")  == scenario)
    )
    sub_resmat = resmat.loc[:, sub_mask]
    questions = sub_resmat.columns.get_level_values("input.text")
    data = sub_resmat.values
    n_questions = data.shape[1]
    corr_matrix = np.zeros((n_questions, n_questions))
    for i in tqdm(range(n_questions)):
        for j in range(i, n_questions):
            r = tetrachoric_corr(data[:, i], data[:, j])
            corr_matrix[i, j] = corr_matrix[j, i] = r
    logits = np.nanmean(corr_matrix, axis=1)
    
    # merge the two data
    tetra = pd.DataFrame({"input.text": questions, "tetrachoric": logits})
    merged = tetra.merge(
        sub_long,
        on="input.text",
        how="inner"
    )
    merged = merged.where(merged.notna(), None)

    # create dict, upload to HF
    validity_dict = {
        (
            row["instance_id"],
            row["perturbation.name"],
            row["train_trial_index"],
        ): row["tetrachoric"]
        for _, row in merged.iterrows()
    }
    cleaned_validity_dict = {
        (
            inst_id,
            None if pd.isna(perturb) else perturb,
            trial_idx
        ): tetra
        for (inst_id, perturb, trial_idx), tetra in validity_dict.items()
    }
    breakpoint()
    buffer = io.BytesIO()
    pickle.dump(cleaned_validity_dict, buffer)
    buffer.seek(0)

    login()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=buffer,
        path_in_repo="validity.pkl",
        repo_id="stair-lab/helm_display_validity",
        repo_type="dataset",
    )