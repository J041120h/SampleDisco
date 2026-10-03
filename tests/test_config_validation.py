"""Config validation: the three post-0.2.0 keys are optional, every other key is required.
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
    assert len(cfg) == len(inspect.signature(wrapper).parameters) == 304
    validate_config(cfg, wrapper)


def test_pre_0_3_config_validates_and_defaults_to_true():
    cfg = {k: v for k, v in demo_config().items() if k not in OPTIONAL_KEYS}
    assert len(cfg) == 301
    validate_config(cfg, wrapper)
    params = inspect.signature(wrapper).parameters
    assert all(params[k].default is True for k in OPTIONAL_KEYS)


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
