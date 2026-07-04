import argparse
import inspect
import os
import sys
import traceback
from importlib.resources import files

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="SampleDisco — cross-omics sample embedding for single-cell data.",
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="complex", choices=["complex"],
        help="Run mode (only 'complex' is supported; kept for backward compatibility).",
    )
    parser.add_argument(
        "--config", type=str, help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--init-config", nargs="?", const="config.yaml", metavar="PATH", dest="init_config",
        help="Write a ready-to-edit config template to PATH (default: ./config.yaml) and exit.",
    )
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config, func):
    valid_params = inspect.signature(func).parameters
    for key in config:
        if key not in valid_params:
            raise ValueError(f"Unexpected parameter in config: '{key}'")
    for key in valid_params:
        if key not in config:
            raise ValueError(f"Missing required parameter in config: '{key}'")


def write_template(dest):
    if os.path.exists(dest):
        print(f"Error: '{dest}' already exists; refusing to overwrite.", file=sys.stderr)
        sys.exit(1)
    template = (files("sampledisco") / "config" / "config_demo.yaml").read_text()
    with open(dest, "w") as f:
        f.write(template)
    print(
        f"Wrote a starter config to '{dest}'.\n"
        f"Edit the data paths / options, then run:\n"
        f"  sampledisco -m complex --config {dest}"
    )


def main():
    args = parse_args()

    if args.init_config is not None:
        write_template(args.init_config)
        return

    if not args.config:
        print(
            "Error: --config is required. Use `sampledisco --init-config` to create a "
            "starter config first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Imported here (not at module top) so `--init-config` / `--help` stay instant
    # and don't pull in the heavy scanpy / torch stack.
    from sampledisco.wrapper.wrapper import wrapper

    config = load_config(args.config)
    try:
        validate_config(config, wrapper)
        wrapper(**config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
