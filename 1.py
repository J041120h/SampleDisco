from pyensembl import EnsemblRelease

# Load Ensembl reference (GRCh38 default with release 110 unless you set differently)
ensembl_release = 110  # Change if needed
ensembl = EnsemblRelease(release=ensembl_release, species="homo_sapiens")

# Download and index if not already available
try:
    genes = ensembl.genes()
except Exception:
    print("Downloading Ensembl data (first time only)…")
    ensembl.download()
    ensembl.index()
    genes = ensembl.genes()

# Query for UTP4
gene_name = "UTP4"
utp4_genes = ensembl.genes_by_name(gene_name)

# Print all available information
for gene in utp4_genes:
    print(f"Gene ID       : {gene.gene_id}")
    print(f"Gene Name     : {gene.gene_name}")
    print(f"Chromosome    : {gene.contig}")
    print(f"Start         : {gene.start}")
    print(f"End           : {gene.end}")
    print(f"Strand        : {'+' if gene.strand == 1 else '-'}")
    print("-" * 40)

