import pandas as pd
import numpy as np
import os

def distanceCheck(df_path, row, method, summary_csv_path):
    df = pd.read_csv(df_path, index_col=0)

    # Get the directory of the CSV file
    directory = os.path.dirname(os.path.abspath(df_path))

    # Get the sample names
    samples = df.index.tolist()

    # Determine the group for each sample (first two letters)
    groups = {sample: sample[:2] for sample in samples}

    # Initialize lists to store in-group and between-group distances
    in_group_distances = []
    between_group_distances = []

    # Loop over all pairs of samples to compute distances
    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i >= j:
                continue  # Avoid redundant pairs and self-distances
            distance = df.iloc[i, j]
            if groups[sample_i] == groups[sample_j]:
                in_group_distances.append(distance)
            else:
                between_group_distances.append(distance)

    # Calculate average distances
    average_in_group_distance = np.mean(in_group_distances)
    average_between_group_distance = np.mean(between_group_distances)
    score = average_between_group_distance / average_in_group_distance

    # Prepare the result string
    result = (
        f"Average in-group distance: {average_in_group_distance}\n"
        f"Average between-group distance: {average_between_group_distance}\n"
        f"Score: {score}"
    )

    # Define the output file path
    output_file = os.path.join(directory, 'distance_results.txt')

    # Write the results to a text file
    with open(output_file, 'w') as f:
        f.write(result)

    print(f"Results saved to {output_file}")

    # Update the summary CSV
    summary_file = summary_csv_path
    # Check if the file exists
    if os.path.isfile(summary_file):
        # Read the existing summary CSV
        summary_df = pd.read_csv(summary_file, index_col=0)
    else:
        # Create a new DataFrame with the required rows and columns
        rows = ['cell_proportion', 'gene_expression']
        summary_df = pd.DataFrame(index=rows)
    
    # Ensure the method is in the columns
    if method not in summary_df.columns:
        summary_df[method] = np.nan

    # Update the cell with the computed score
    summary_df.loc[row, method] = score

    # Save the updated summary DataFrame back to CSV
    summary_df.to_csv(summary_file)
    print(f"Summary updated in {summary_file}")


def distanceCheckSimple(df_path):
    df = pd.read_csv(df_path, index_col=0)

    # Get the directory of the CSV file
    directory = os.path.dirname(os.path.abspath(df_path))

    # Get the sample names
    samples = df.index.tolist()

    # Determine the group for each sample (first two letters)
    groups = {sample: sample[:2] for sample in samples}

    # Initialize lists to store in-group and between-group distances
    in_group_distances = []
    between_group_distances = []

    # Loop over all pairs of samples to compute distances
    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i >= j:
                continue  # Avoid redundant pairs and self-distances
            distance = df.iloc[i, j]
            if groups[sample_i] == groups[sample_j]:
                in_group_distances.append(distance)
            else:
                between_group_distances.append(distance)

    # Calculate average distances
    average_in_group_distance = np.mean(in_group_distances)
    average_between_group_distance = np.mean(between_group_distances)
    score = average_between_group_distance / average_in_group_distance

    # Prepare the result string
    result = (
        f"Average in-group distance: {average_in_group_distance}\n"
        f"Average between-group distance: {average_between_group_distance}\n"
        f"Score: {score}"
    )

    # Define the output file path
    output_file = os.path.join(directory, 'distance_results.txt')

    # Write the results to a text file
    with open(output_file, 'w') as f:
        f.write(result)

    print(f"Results saved to {output_file}")