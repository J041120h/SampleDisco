"""Supervised downstream use of the sample embedding: phenotype prediction and label transfer."""
from sampledisco.sample_prediction.prediction import predict_phenotype, transfer_labels

__all__ = ["predict_phenotype", "transfer_labels"]
