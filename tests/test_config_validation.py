"""Config validation: the post-0.2.0 keys are optional, every other key is required.
Run with `pytest tests/` or `python tests/test_config_validation.py`."""
import inspect
from importlib.resources import files

import yaml

from sampledisco.cli import OPTIONAL_KEYS, validate_config
from sampledisco.wrapper.wrapper import wrapper


def raises(msg, fn, *args):
    try:
        fn(*args)
    except ValueError as e:
        assert msg in str(e), e
        return
    raise AssertionError(f"expected ValueError: {msg}")


def demo_config():
    return yaml.safe_load((files("sampledisco") / "config" / "config_demo.yaml").read_text())


def test_full_config_validates():
    cfg = demo_config()
    assert len(cfg) == len(inspect.signature(wrapper).parameters) == 311
    validate_config(cfg, wrapper)


def test_pre_0_3_config_validates_and_defaults_keep_old_behaviour():
    cfg = {k: v for k, v in demo_config().items() if k not in OPTIONAL_KEYS}
    assert len(cfg) == 301
    validate_config(cfg, wrapper)
    params = inspect.signature(wrapper).parameters
    assert all(params[k].default == ("gam" if k.endswith("_trajectory_test")
                                     else True if k.endswith("_save_cell_adata_after_embedding") else None)
               for k in OPTIONAL_KEYS)


def test_trajectory_test_keys_reach_the_dge_call(tmp_path, monkeypatch):
    import sampledisco.wrapper.wrapper as W
    import sampledisco.wrapper.multiomics_wrapper as MW
    calls = {}

    def fake_downstream(**kw):
        calls[kw["modality"]] = kw
        return {"status_flags": kw.get("status_flags", {})}

    fake_pre = lambda **kw: {"adata": None, "status_flags": kw.get("status_flags", {})}
    monkeypatch.setattr(W, "downstream_analysis", fake_downstream)
    monkeypatch.setattr(W, "rna_wrapper", fake_pre)
    monkeypatch.setattr(W, "atac_wrapper", fake_pre)
    monkeypatch.setattr(MW, "multiomics_wrapper", fake_pre)
    monkeypatch.setattr(W, "_sample_adata_or_none", lambda *a, **k: None)
    cfg = demo_config()
    cfg.update(output_dir=str(tmp_path), run_multiomics_pipeline=True, multiomics_integration=False,
               rna_trajectory_test="linear", atac_trajectory_test="gam", multiomics_trajectory_test="linear")
    W.wrapper(**cfg)
    assert [calls[m]["trajectory_test"] for m in ("rna", "atac", "multiomics")] == ["linear", "gam", "linear"]
    assert calls["atac"]["dge_pseudobulk_min_variance_ratio"] == 0
    assert "dge_pseudobulk_min_variance_ratio" not in calls["rna"]
    assert "dge_pseudobulk_min_variance_ratio" not in calls["multiomics"]


def test_invalid_trajectory_test_value_rejected():
    for key in ("rna_trajectory_test", "atac_trajectory_test", "multiomics_trajectory_test"):
        raises(f"Invalid value for '{key}'", validate_config, dict(demo_config(), **{key: "quadratic"}), wrapper)


def test_other_missing_key_still_rejected():
    cfg = demo_config()
    del cfg["rna_min_cells"]
    raises("Missing required parameter", validate_config, cfg, wrapper)


def test_unknown_key_still_rejected():
    cfg = dict(demo_config(), not_a_parameter=1)
    raises("Unexpected parameter", validate_config, cfg, wrapper)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
