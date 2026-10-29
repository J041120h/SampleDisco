"""predict_phenotype and transfer_labels on synthetic sample embeddings with a known answer.
Run with `pytest tests/` or `python tests/test_sample_prediction.py`."""
import anndata as ad
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, roc_auc_score

from sampledisco import predict_phenotype, transfer_labels


def make(X, **obs):
    a = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(obs, index=[f"s{i}" for i in range(len(X))]))
    a.uns["X_DR_sample"] = pd.DataFrame(X, index=a.obs_names, columns=[f"PC{j + 1}" for j in range(X.shape[1])])
    return a


def separated(n=60, pc=0, seed=0):
    """Two classes 12 SD apart on one PC (``pc``), pure noise on the other four."""
    rng = np.random.default_rng(seed)
    y = np.repeat(["healthy", "severe"], n // 2)
    X = rng.normal(size=(n, 5))
    X[:, pc] += np.where(y == "severe", 6.0, -6.0)
    return make(X, status=y)


def test_binary_perfect():
    pred, m = predict_phenotype(separated(), "status")
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0 and m["n_pcs"] == 5
    assert m["C"] == [10] * 5                                  # every C ties on separable data: weakest penalty
    assert (pred["predicted"] == pred["label"]).all()
    assert np.allclose(pred[["prob_healthy", "prob_severe"]].sum(1), 1)


def test_only_first_n_pcs_used():
    a = separated(pc=2)                                       # signal on PC3 only
    assert predict_phenotype(a, "status", n_pcs=3)[1]["auc"] == 1.0
    assert predict_phenotype(a, "status", n_pcs=2)[1]["auc"] < 0.75
    assert transfer_labels(a, "status", n_pcs=3)[1]["loo_accuracy"] == 1.0
    assert transfer_labels(a, "status", n_pcs=2)[1]["loo_accuracy"] < 0.75


def test_random_labels_near_chance():
    rng = np.random.default_rng(1)
    a = make(rng.normal(size=(200, 10)), status=rng.choice(["a", "b"], 200))
    _, m = predict_phenotype(a, "status")
    assert 0.3 < m["auc"] < 0.7


def test_multiclass_and_regression():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1, 2], 30)
    X = rng.normal(size=(90, 4))
    X[:, 0] += 8 * y
    a = make(X, grade=y, score=3 * X[:, 0] + rng.normal(size=90))
    pred, m = predict_phenotype(a, "grade")
    assert "auc" not in m and m["accuracy"] > 0.95 and {"prob_0", "prob_1", "prob_2"} <= set(pred.columns)
    pred, m = predict_phenotype(a, "score", task="regression")
    assert m["r2"] > 0.95 and len(m["alpha"]) == 5


def test_groups_never_split_and_seeded():
    """Each donor's draws share a fold. The label is a donor trait unrelated to X, and donors sit
    apart in 30 dimensions: when draws of a donor are split across folds the model memorizes donors."""
    rng = np.random.default_rng(3)
    donors = np.repeat([f"d{i}" for i in range(20)], 4)
    status = np.repeat(rng.permutation(np.repeat(["neg", "pos"], 10)), 4)
    X = np.repeat(rng.normal(size=(20, 30)), 4, axis=0) + rng.normal(size=(80, 30)) * 0.05
    a = make(X, status=status, donor=donors)
    pred, m = predict_phenotype(a, "status", group_col="donor")
    assert (pred.groupby(a.obs["donor"])["fold"].nunique() == 1).all() and m["n_groups"] == 20
    assert m["auc"] < 0.8                                    # no donor leakage: near chance
    _, leaky = predict_phenotype(a, "status")                 # draws split across folds: inflated
    assert leaky["auc"] > 0.95
    again, _ = predict_phenotype(a, "status", group_col="donor")
    assert again.equals(pred)


def test_score_by_group():
    """Group scores = scores of the per-group mean of the out-of-fold predictions."""
    rng = np.random.default_rng(5)
    donors = np.repeat([f"d{i}" for i in range(30)], 3)
    trait = np.repeat(rng.choice(["neg", "pos"], 30), 3)
    X = rng.normal(size=(90, 4))
    X[:, 0] += np.where(trait == "pos", 1.0, -1.0)
    a = make(X, trait=trait, donor=donors, bmi=X[:, 0] + rng.normal(size=90))
    pred, m = predict_phenotype(a, "trait", group_col="donor", score_by="group")
    d = pred.groupby(a.obs["donor"]).agg(label=("label", "first"), p=("prob_pos", "mean"))
    assert np.isclose(m["auc"], roc_auc_score(d["label"] == "pos", d["p"]))
    assert np.isclose(m["accuracy"], ((d["p"] > 0.5) == (d["label"] == "pos")).mean())
    assert pred.equals(predict_phenotype(a, "trait", group_col="donor")[0])   # only the scoring changes
    pred, m = predict_phenotype(a, "bmi", group_col="donor", task="regression", score_by="group")
    d = pred.groupby(a.obs["donor"])[["label", "predicted"]].mean()
    assert np.isclose(m["r2"], r2_score(d["label"], d["predicted"]))


def test_missing_labels_skipped_and_obsm_fallback():
    a = separated()
    a.obs.loc[["s0", "s1"], "status"] = np.nan
    a.obsm["X_DR_sample"] = a.uns.pop("X_DR_sample").to_numpy()
    pred, m = predict_phenotype(a, "status")
    assert m["n_samples"] == 58 and "s0" not in pred.index and m["auc"] > 0.99


def test_transfer_recovers_held_out_labels():
    a = separated()
    hidden = a.obs["status"].copy()
    a.obs.loc[a.obs_names[::6], "status"] = np.nan
    table, m = transfer_labels(a, "status")
    q = table[table["query"]]
    assert len(q) == 10 and (q["predicted"] == hidden[q.index]).all() and (q["confidence"] == 1).all()
    assert m["loo_accuracy"] == 1.0 and m["n_reference"] == 50 and m["n_query"] == 10


def test_transfer_mask_vote_tie_and_weights():
    """Query at 0; references at 1 (A), 2 (B), 3 (B), 4 (A). k=4 is a 2:2 tie -> nearest (A).
    k=3: majority B (2/3); distance-weighted A wins (1 vs 1/2 + 1/3)."""
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    a = make(X, cls=["A", "A", "B", "B", "A"])
    query = np.array([True, False, False, False, False])        # labelled but masked
    table, _ = transfer_labels(a, "cls", query=query, n_neighbors=4)
    assert table.loc["s0", "predicted"] == "A" and table.loc["s0", "confidence"] == 0.5
    table, _ = transfer_labels(a, "cls", query=query, n_neighbors=3)
    assert table.loc["s0", "predicted"] == "B" and np.isclose(table.loc["s0", "confidence"], 2 / 3)
    table, _ = transfer_labels(a, "cls", query=query, n_neighbors=3, weighted=True)
    assert table.loc["s0", "predicted"] == "A" and np.isclose(table.loc["s0", "confidence"], 1 / (1 + 1 / 2 + 1 / 3))


def test_transfer_loo_excludes_group():
    """Paired units (one sample in two modalities) sit on top of each other: without group_col each
    unit's nearest reference is its partner, so leave-one-out accuracy is inflated."""
    rng = np.random.default_rng(4)
    centres = rng.normal(size=(40, 3)) * 5
    lab = rng.choice(["x", "y"], 40)                         # unrelated to position
    X = np.repeat(centres, 2, axis=0) + rng.normal(size=(80, 3)) * 0.01
    a = make(X, sample=np.repeat([f"p{i}" for i in range(40)], 2), cls=np.repeat(lab, 2))
    _, leaky = transfer_labels(a, "cls", n_neighbors=1)
    _, fair = transfer_labels(a, "cls", n_neighbors=1, group_col="sample")
    assert leaky["loo_accuracy"] == 1.0 and fair["loo_accuracy"] < 0.75


def test_numeric_labels_keep_dtype():
    a = separated()
    a.obs["sev"] = np.where(a.obs["status"] == "severe", 4.0, 1.0)
    a.obs.loc["s3", "sev"] = np.nan
    table, m = transfer_labels(a, "sev")
    assert table["predicted"].dtype == float and table.loc["s3", "predicted"] == 1.0 and m["loo_accuracy"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
