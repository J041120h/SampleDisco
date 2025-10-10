import argparse
import sys
import yaml
import os
import inspect
from wrapper.wrapper import wrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Run the data processing wrapper.")

    parser.add_argument("-m", "--mode", type=str, required=True, choices=["simple", "complex"],
                        help="Run mode. Choose 'simple' or 'complex'.")

    # Simple mode args
    parser.add_argument("-c", "--count_data", type=str, help="Path to count data file")
    parser.add_argument("-s", "--sample_meta_data", type=str, help="(Optional) Path to sample metadata file")
    parser.add_argument("-o", "--output_directory", type=str, help="Path to output directory")

    # Complex mode args
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    return parser.parse_args()

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_config(config, func):
    valid_params = inspect.signature(func).parameters
    for key in config:
        if key not in valid_params:
            raise ValueError(f"Unexpected parameter in config: '{key}'")
    for key in valid_params:
        if key not in config:
            raise ValueError(f"Missing required parameter in config: '{key}'")

def main():
    args = parse_args()

    if args.mode == "simple":
        if not args.count_data or not args.output_directory:
            print("Error: In 'simple' mode, -c and -o must be provided.", file=sys.stderr)
            sys.exit(1)

        # Call wrapper with or without sample_meta_data
        if args.sample_meta_data:
            wrapper(args.count_data, args.sample_meta_data, args.output_directory)
        else:
            wrapper(args.count_data, output_directory=args.output_directory)

    elif args.mode == "complex":
        if not args.config:
            print("Error: In 'complex' mode, --config must be provided.", file=sys.stderr)
            sys.exit(1)

        config = load_config(args.config)

        try:
            validate_config(config, wrapper)
            wrapper(**config)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
