"""Split the `timepoint` column of the 1M-scBloodNL AnnData into
`stimulation_conditions` and `timepoint` (snake_case), then overwrite
the original .h5ad with gzip compression.

Mapping:
    UT      -> stimulation_conditions='UT',  timepoint=0
    3hCA    -> stimulation_conditions='CA',  timepoint=3
    24hCA   -> stimulation_conditions='CA',  timepoint=24
    3hMTB   -> stimulation_conditions='MTB', timepoint=3
    24hMTB  -> stimulation_conditions='MTB', timepoint=24
    3hPA    -> stimulation_conditions='PA',  timepoint=3
    24hPA   -> stimulation_conditions='PA',  timepoint=24
    nan     -> both NA
"""

import re
import anndata as ad
import pandas as pd

H5AD_PATH = "/dcs07/hongkai/data/harry/result/1M-scBloodNL/data/1M-scBloodNL.h5ad"

STIM_CATEGORIES = ["UT", "CA", "MTB", "PA"]

_PATTERN = re.compile(r"^(\d+)h(CA|MTB|PA)$")


def split_label(label):
    if pd.isna(label):
        return pd.NA, pd.NA
    label = str(label)
    if label == "UT":
        return "UT", 0
    if label.lower() == "nan":
        return pd.NA, pd.NA
    m = _PATTERN.match(label)
    if m is None:
        raise ValueError(f"Unrecognized timepoint label: {label!r}")
    time_int, stim_str = int(m.group(1)), m.group(2)
    return stim_str, time_int


def main():
    print(f"Loading: {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"  shape: {adata.shape}")

    original = adata.obs["timepoint"].astype("string")
    print("Original timepoint value counts:")
    print(original.value_counts(dropna=False))

    split = original.map(split_label)
    stim = split.map(lambda x: x[0])
    time = split.map(lambda x: x[1])

    stim_cat = pd.Categorical(stim, categories=STIM_CATEGORIES)
    time_int = pd.array(time.tolist(), dtype="Int32")

    adata.obs["stimulation_conditions"] = pd.Series(
        stim_cat, index=adata.obs.index
    )
    adata.obs["timepoint"] = pd.Series(time_int, index=adata.obs.index)

    print("\nNew stimulation_conditions value counts:")
    print(adata.obs["stimulation_conditions"].value_counts(dropna=False))
    print("\nNew timepoint value counts:")
    print(adata.obs["timepoint"].value_counts(dropna=False))

    print(f"\nWriting (gzip compressed) back to: {H5AD_PATH}")
    adata.write_h5ad(H5AD_PATH, compression="gzip")
    print("Done.")


if __name__ == "__main__":
    main()
