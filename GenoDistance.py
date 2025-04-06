from wrapper import wrapper
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run the data processing wrapper with mode control.")

    parser.add_argument("-m", "--mode", type=str, required=True,
                        help="Run mode. Only 'simple' is supported currently.")
    
    parser.add_argument("-c", "--count_data", type=str,
                        help="Path to the count data file (e.g., counts.csv)")
    parser.add_argument("-s", "--sample_meta_data", type=str,
                        help="Path to the sample metadata file (e.g., metadata.csv)")
    parser.add_argument("-o", "--output_directory", type=str,
                        help="Path to the output directory")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == "simple":
        # Ensure required arguments are provided in simple mode
        if not args.count_data or not args.sample_meta_data or not args.output_directory:
            print("Error: In 'simple' mode, -c, -s, and -o must all be provided.", file=sys.stderr)
            sys.exit(1)

        wrapper(h5ad_path = args.count_data, sample_meta_path = args.sample_meta_data, output_dir = args.output_directory)

    else:
        print(f"Mode '{args.mode}' is not implemented yet. Only 'simple' mode is supported for now.")
        # You can later call other functions here for different modes

if __name__ == "__main__":
    main()