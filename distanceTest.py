import pandas as pd
import numpy as np
import os

# Read the distance matrix CSV file into a DataFrame
def distanceCheck(df_path):
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