"""Add Batch_AgeGroup combined categorical column to the round1 preprocessed
h5ad in place (h5py — no X rewrite). Used by round5 sample embedding to
regress out Batch + Age_group via Harmony's single batch_key path."""

import h5py
import numpy as np
import pandas as pd

H5 = "/dcs07/hongkai/data/harry/result/health_aging_PBMC/round1_batch/preprocess/adata_preprocessed.h5ad"


def _read_cat(grp):
    cats = np.array([x.decode() if isinstance(x, bytes) else x for x in grp["categories"][:]])
    codes = grp["codes"][:]
    return pd.Categorical.from_codes(codes, categories=cats)


def main():
    with h5py.File(H5, "r+") as f:
        obs = f["obs"]
        batch = _read_cat(obs["Batch"])
        age   = _read_cat(obs["Age_group"])
        joint = pd.Series(batch.astype(str)) + ":" + pd.Series(age.astype(str))
        cats, codes_arr = np.unique(joint.values, return_inverse=True)
        codes = codes_arr.astype(np.int16 if len(cats) >= 127 else np.int8)
        print(f"n_strata = {len(cats)}")
        print("first 8 strata:", list(cats[:8]))
        # count per stratum
        cnt = pd.Series(joint).value_counts()
        print(f"min strata cells = {cnt.min()}; max = {cnt.max()}; n_strata_with_<100_cells = {(cnt < 100).sum()}")

        # delete if present
        if "Batch_AgeGroup" in obs:
            del obs["Batch_AgeGroup"]
        g = obs.create_group("Batch_AgeGroup")
        g.attrs["encoding-type"] = "categorical"
        g.attrs["encoding-version"] = "0.2.0"
        g.attrs["ordered"] = False
        dt = h5py.string_dtype("utf-8")
        cds = g.create_dataset("categories", data=cats.astype(object), dtype=dt)
        cds.attrs["encoding-type"] = "string-array"
        cds.attrs["encoding-version"] = "0.2.0"
        ccs = g.create_dataset("codes", data=codes)
        ccs.attrs["encoding-type"] = "array"
        ccs.attrs["encoding-version"] = "0.2.0"

        order = list(obs.attrs.get("column-order", []))
        if "Batch_AgeGroup" not in order:
            order.append("Batch_AgeGroup")
            obs.attrs["column-order"] = np.array(order, dtype=object)
    print("done; column Batch_AgeGroup added.")


if __name__ == "__main__":
    main()
