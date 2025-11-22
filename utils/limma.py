import pandas as pd
import patsy
import numpy as np
import sys

# Limma function for batch correction
def limma(pheno, exprs, covariate_formula, design_formula='1', rcond=1e-8):
    design_matrix = patsy.dmatrix(design_formula, pheno)
    
    design_matrix = design_matrix[:,1:]
    rowsum = design_matrix.sum(axis=1) -1
    design_matrix=(design_matrix.T+rowsum).T
    
    covariate_matrix = patsy.dmatrix(covariate_formula, pheno)
    design_batch = np.hstack((covariate_matrix,design_matrix))
    coefficients, res, rank, s = np.linalg.lstsq(design_batch, exprs.T, rcond=rcond)
    beta = coefficients[-design_matrix.shape[1]:]
    return exprs - design_matrix.dot(beta).T
