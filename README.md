# GenoDistance

```markdown
# Single Cell Data Downstream Analysis

This project provides tools for downstream analysis of single-cell RNA sequencing (scRNA-seq) data. It offers a streamlined pipeline for preprocessing, integrating, and assessing sample similarity, enabling researchers to explore cell populations and their relationships across different samples effectively.

## Table of Contents

1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Harmonization](#harmonization)
4. [Sample Similarity](#sample-similarity)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

To get started, clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
```

## Preparation

1. **Download Data**: Obtain the necessary data files and store them locally in a directory of your choice.

2. **Configure Paths**: Update the data and output paths in the program to correctly access the data and store the results.

## Harmonization

The `treecor_harmony` function is designed for preprocessing and integrating scRNA-seq data using Harmony, a tool that reduces batch effects. This function loads a raw gene expression matrix and associated metadata, performs data filtering, normalization, principal component analysis (PCA), and integrates the data to mitigate batch effects.

### Key Steps

1. **Data Loading and Filtering**
   - **Load Data**: Reads count data and metadata.
   - **Filter Genes**: Removes genes with low expression.
   - **Quality Control**: Eliminates cells with high mitochondrial gene content, which often indicates low-quality data.

2. **Harmony Integration**
   - **Batch Effect Reduction**: Applies Harmony to minimize batch effects across samples, ensuring comparability of cells from different batches.

3. **Dimension Reduction and Clustering**
   - **PCA**: Reduces data dimensionality using Principal Component Analysis.
   - **Clustering**: Performs Leiden clustering on Harmony embeddings to identify cell subpopulations.

4. **Visualization**
   - **UMAP**: Visualizes cell clusters in 2D space.
   - **Dendrogram**: Generates a phylogenetic tree to illustrate relationships among clusters.

5. **Output**
   - **Save Results**: Stores the processed data as an AnnData object.
   - **Plots**: Saves cluster plots and the dendrogram to the specified output directory.

### Usage Example

```python
from your_module import treecor_harmony

# Define input and output paths
input_data = 'path/to/raw_expression_matrix.h5ad'
metadata = 'path/to/metadata.csv'
output_dir = 'path/to/output/'

# Run the harmonization pipeline
treecor_harmony(input_data, metadata, output_dir)
```

## Sample Similarity

The sample similarity module calculates similarity based on different cell types within each sample. Users can choose to focus on the average expression of each cell type or the proportion of each type by adjusting the weights of the distance matrix. Additionally, an integrated method multiplies the weight by its expression to obtain an internally weighted distance matrix. Both methods utilize the Earth Mover's Distance (EMD) and summarize the results in CSV files and heatmaps.

### Key Features

- **Customization**: 
  - **Average Expression**: Focus on the average expression levels of each cell type.
  - **Cell Type Proportion**: Emphasize the proportion of each cell type.
  - **Integrated Method**: Combine both approaches by weighting expression with proportions.

- **Visualization**: Generates heatmaps to visualize sample similarities.

- **Output**: Saves similarity matrices and corresponding heatmaps in the output directory.

### Usage Example

```python
from your_module import calculate_sample_similarity

# Define input and output paths
processed_data = 'path/to/processed_data.h5ad'
output_dir = 'path/to/output/'

# Calculate sample similarity based on average expression
calculate_sample_similarity(processed_data, output_dir, method='average_expression')

# Calculate sample similarity based on cell type proportion
calculate_sample_similarity(processed_data, output_dir, method='cell_type_proportion')

# Calculate sample similarity using the integrated method
calculate_sample_similarity(processed_data, output_dir, method='integrated')
```

## Usage

Provide detailed instructions or examples on how to use the different parts of the project here. This can include command-line instructions, scripts, or additional code snippets to help users get started quickly.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Submit a pull request describing your changes.

Please ensure that your contributions adhere to the project's coding standards and include appropriate tests.

---
```
