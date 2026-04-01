from datetime import datetime
from tests import (crossMMD2sampleUnpaired, safe_crossSSMMD2sample, 
                   TwoSampleMMDSquared)
from utils import (RBFkernel, RBFkernel1, get_bootstrap_threshold, 
                   get_normal_threshold, get_spectral_threshold, 
                   get_bootstrap_std,GaussianVector, get_median_bw)
from functools import partial
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle

# Set device for computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
d1, d2 = 10, 10  # Dimensions
rho1, rho2 = 0.95, 0.95  # Covariance
eps = 0.3  # Magnitude of perturbation
num_pert = 1  # Number of coordinates to perturb


n1, m1, n2, m2 = 200, 1000, 200, 1000  # Sample sizes
alpha = 0.05  # Level of test
num_bootstrap = 200  # Number of bootstrap samples
num_points = 10  # Number of sample sizes
initial_sample_size_n1 = n1 / num_points
initial_sample_size_m1 = m1 / num_points
initial_sample_size_n2 = n2 / num_points
initial_sample_size_m2 = m2 / num_points
num_perms = 200  # Number of permutations
num_trials = 1000  # Number of trials
kernel_type= RBFkernel # Kernel type: Linearkernel, RBFkernel

# Function handles for threshold computing methods
thresh_permutation = partial(get_bootstrap_threshold, num_perms=num_perms)
thresh_normal = get_normal_threshold
thresh_spectral = partial(get_spectral_threshold,  alpha=alpha, numNullSamp=200)

poly_degree = 2
split = 2

# Create the data sources
meanX, meanY = np.zeros((d1,)),  np.zeros((d1,))
meanX[:num_pert] = (eps)
meanY[:num_pert] = (eps)
meanV, meanW = np.zeros((d2,)), np.zeros((d2,))
meanW[:num_pert] = (eps)
covX =  np.eye(d1)
covY =  np.eye(d1)
covV =  np.eye(d2) + rho1 * (np.ones((d2, d2)) - np.eye(d2))
covW =  np.eye(d2) + rho2 * (np.ones((d2, d2)) - np.eye(d2))

def SourceX(n):
    return GaussianVector(mean=meanX, cov=covX, n=n)
def SourceV(n):
    return GaussianVector(mean = meanV, cov=covV, n=n)
def SourceY(n):
    return GaussianVector(mean = meanY, cov=covY, n=n)
def SourceW(n):
    return GaussianVector(mean=meanW, cov=covW, n=n)

methods = ['MMD-perm', 'xMMD', 'xssMMD(knn)','xssMMD(ker)', 'xssMMD(rf)']

# Initialize sample sizes
NN1 = np.linspace(initial_sample_size_n1, n1, num_points, dtype=int)
NN2 = np.linspace(initial_sample_size_n2, n2, num_points, dtype=int)
MM1 = np.linspace(initial_sample_size_m1, m1, num_points, dtype=int)
MM2 = np.linspace(initial_sample_size_m2, m2, num_points, dtype=int)


# Initialize the dictionaries to store the power results
TypeIErrorDict = {method: np.zeros((num_trials, len(NN1))) for method in methods}
TypeIErrorStdDict = {method: np.zeros(NN1.shape) for method in methods}

TimeDict = {method: np.zeros((num_trials, len(NN1))) for method in methods}
AvgTimeDict = {} 
# ---------------------------------------------
    
start_time = time.time()

# Main loop for trials
for i in tqdm(range(num_trials)):
    for j, (n1i, n2i, m1i, m2i) in enumerate(zip(NN1, NN2, MM1, MM2)):
        X = SourceX(n1i)
        Y = SourceY(n2i)
        
        X_new = SourceX(m1i)
        V = np.vstack([X[:, :2], X_new[:, :2]])
        
        Y_new = SourceY(m2i)
        W = np.vstack([Y[:, [-2, -1]], Y_new[:, [-2, -1]]])
        
        XV = np.hstack([X, V[:n1i]]) # X와 V를 합침
        YW = np.hstack([Y, W[:n2i]]) # Y와 W를 합침


        # Obtain the bandwidth of the kernel
        bw_marginal = get_median_bw(X, Y)
        bw_joint = get_median_bw(XV, YW)
        kernel_func_joint = partial(RBFkernel1, bw=bw_joint) if kernel_type == RBFkernel else partial(Linearkernel)
        kernel_func_marginal = partial(RBFkernel1, bw=bw_marginal) if kernel_type == RBFkernel else partial(Linearkernel)

        # Set up function handles for the different statistics
        unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True)
        biased_mmd2 = partial(TwoSampleMMDSquared, unbiased=False)
        cross_mmd2 = crossMMD2sampleUnpaired
        crossSSMMD_ker = partial(safe_crossSSMMD2sample, method = "KernelRegression")
        
        # Run tests and record outcomes
        X, V, Y, W, XV, YW = torch.tensor(X), torch.tensor(V), torch.tensor(Y), torch.tensor(W), torch.tensor(XV), torch.tensor(YW)
        for method in methods:
            
            t_method_start = time.time()
            
            if method=='MMD-perm':
                stat = unbiased_mmd2(XV, YW, kernel_func_joint)
                th = thresh_permutation(XV, YW, kernel_func_joint, unbiased_mmd2, alpha=alpha)
            elif method=='xMMD':
                stat = cross_mmd2(XV, YW, kernel_func_joint)
                th = thresh_normal(alpha)
            elif method == 'xssMMD(lin)':
                stat = safe_crossSSMMD2sample(X,V,Y,W,kernel_func_marginal, "linearRegression")
                th = thresh_normal(alpha)
            elif method == 'xssMMD(knn)':                
                stat = safe_crossSSMMD2sample(X,V,Y,W,kernel_func_marginal, "KNN")
                th = thresh_normal(alpha)
            elif method == 'xssMMD(ker)':
                stat = safe_crossSSMMD2sample(X,V,Y,W,kernel_func_marginal, "KernelRegression")
                th = thresh_normal(alpha)
            elif method == 'xssMMD(rf)':
                stat = safe_crossSSMMD2sample(X,V,Y,W,kernel_func_marginal, "RandomForest")
                th = thresh_normal(alpha)
            
            t_method_end = time.time()
            TimeDict[method][i][j] = t_method_end - t_method_start
            
            TypeIErrorDict[method][i][j] = 1.0*(stat>th)
    
# Compute mean and std of power
for method in methods:
    TypeIErrorStdDict[method] = np.array([
        get_bootstrap_std(TypeIErrorDict[method][:, i], num_bootstrap=num_bootstrap)
        for i in range(len(NN1))
    ])
    TypeIErrorDict[method] = TypeIErrorDict[method].mean(axis=0)
    

# Save results to pickle files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pickle.dump(TypeIErrorDict, open(f'./TypeIErrorDict_{rho1}_{timestamp}.pkl', 'wb'))
pickle.dump(TypeIErrorStdDict, open(f'./TypeIErrorStdDict_{rho1}_{timestamp}.pkl', 'wb'))



print("\n" + "="*85)
print(f"Type I Error Rate Summary (Alpha={alpha})")
print("="*85)

header = f"{'Sample (n, m)':<22}" + "".join([f"{m:>13}" for m in methods])
print(header)
print("-" * len(header))

# Row of the data
for j in range(len(NN1)):
    n_val = NN1[j]
    m_val = MM1[j]
    row_label = f"n={n_val}, m={m_val}"
    
    row_str = f"{row_label:<22}"
    
    for method in methods:
        # TypeIErrorDict는 이미 mean(axis=0)이 수행되어 [num_points] 크기의 배열임
        val = TypeIErrorDict[method][j]
        row_str += f"{val:>13.3f}" 
    
    print(row_str)

print("="*85 + "\n")


end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")
