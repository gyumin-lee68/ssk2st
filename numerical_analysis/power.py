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
num_pert = 3  # Number of coordinates to perturb
b = [0, -2, -1] # how to construct X or Y
# b = slice(None)  # Alternative scenario

n1, m1, n2, m2 = 300, 3000, 300, 3000  # Sample sizes
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
meanY[:num_pert] = (eps)
meanV, meanW = np.zeros((d2,)), np.zeros((d2,))
meanW[:num_pert] = (eps)
covX =  np.eye(d1) + rho1 * (np.ones((d1, d1)) - np.eye(d1))
covY =  np.eye(d1) + rho2 * (np.ones((d1, d1)) - np.eye(d1))
covV =  np.eye(d2) + rho1 * (np.ones((d2, d2)) - np.eye(d2))
covW =  np.eye(d2) + rho2 * (np.ones((d2, d2)) - np.eye(d2))

def SourceX(n):
    return GaussianVector(mean=meanX, cov=covX, n=n)
def SourceV(n):
    return GaussianVector(mean = meanV, cov=covV, n=n)
def SourceY(n):
    return GaussianVector(mean=meanY, cov=covY, n=n)
def SourceW(n):
    return GaussianVector(mean=meanW, cov=covW, n=n)

methods = ['MMD-perm', 'xMMD', 'xssMMD(knn)','xssMMD(ker)', 'xssMMD(rf)']

# Initialize sample sizes
NN1 = np.linspace(initial_sample_size_n1, n1, num_points, dtype=int)
NN2 = np.linspace(initial_sample_size_n2, n2, num_points, dtype=int)
MM1 = np.linspace(initial_sample_size_m1, m1, num_points, dtype=int)
MM2 = np.linspace(initial_sample_size_m2, m2, num_points, dtype=int)


# Initialize the dictionaries to store the power results
PowerDict = {method: np.zeros((num_trials, len(NN1))) for method in methods}
PowerStdDevDict = {method: np.zeros(NN1.shape) for method in methods}
    
start_time = time.time()

# Main loop for trials
for i in tqdm(range(num_trials)):
    for j, (n1i, n2i, m1i, m2i) in enumerate(zip(NN1, NN2, MM1, MM2)):
        V, W = SourceV(n1i+m1i), SourceW(n2i+m2i)
        
        # Choose labeled portion for X to avoid empty unlabeled split later
        X = V[:n1i, b].sum(axis=1).reshape(-1, 1)
        # X = V[:n1i, :].sum(axis=1).reshape(-1, 1)     # Alternative scenario
        # X = SourceX(n1i)                              # Alternative: independent source
        
        Y = W[:n2i, b].sum(axis=1).reshape(-1, 1)
        # Y = W[:n2i, :].sum(axis=1).reshape(-1, 1)     # Alternative scenario
        # Y = SourceY(n2i)                              # Alternative: independent source

        # Obtain the bandwidth of the kernel
        bw = get_median_bw(X, Y)
        kernel_func = partial(RBFkernel1, bw=bw) if kernel_type == RBFkernel else partial(Linearkernel)

        # Set up function handles for the different statistics
        unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True)
        biased_mmd2 = partial(TwoSampleMMDSquared, unbiased=False)
        cross_mmd2 = crossMMD2sampleUnpaired
        crossSSMMD_ker = partial(safe_crossSSMMD2sample, method = "KernelRegression")
        
        # Run tests and record outcomes
        X, V, Y, W = torch.tensor(X), torch.tensor(V), torch.tensor(Y), torch.tensor(W)
        for method in methods:
            if method=='MMD-perm':
                stat = unbiased_mmd2(X, Y, kernel_func)
                th = thresh_permutation(X, Y, kernel_func, unbiased_mmd2, alpha=alpha)
            elif method=='xMMD':
                stat = cross_mmd2(X, Y, kernel_func)
                th = thresh_normal(alpha)
            elif method == 'xssMMD(lin)':
                stat = safe_crossSSMMD2sample(X,V,Y,W,kernel_func, "linearRegression")
                th = thresh_normal(alpha)
            elif method == 'xssMMD(knn)':                
                stat = safe_crossSSMMD2sample(X,V,Y,W,kernel_func, "KNN")
                th = thresh_normal(alpha)
            elif method == 'xssMMD(ker)':
                stat = safe_crossSSMMD2sample(X,V,Y,W,kernel_func, "KernelRegression")
                th = thresh_normal(alpha)
            elif method == 'xssMMD(rf)':
                stat = safe_crossSSMMD2sample(X,V,Y,W,kernel_func, "RandomForest")
                th = thresh_normal(alpha)
                
            PowerDict[method][i][j] = 1.0*(stat>th)
    
# Compute mean and std of power
for method in methods:
    PowerStdDevDict[method] = np.array([
        get_bootstrap_std(PowerDict[method][:, i], num_bootstrap=num_bootstrap)
        for i in range(len(NN1))
    ])
    PowerDict[method] = PowerDict[method].mean(axis=0)

# Save results to pickle files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pickle.dump(PowerDict, open(f'./PowerDict_{rho1}_{b}_{timestamp}.pkl', 'wb'))
pickle.dump(PowerStdDevDict, open(f'./PowerStdDevDict_{rho1}_{b}_{timestamp}.pkl', 'wb'))


fig, ax = plt.subplots(figsize=(6, 5))

ax.grid(True)

color_map = {
    'MMD-perm': 'blue',
    'xMMD': 'green',
    'xssMMD(knn)': 'red',
    'xssMMD(ker)': 'orange', # 또는 'darkorange'
    'xssMMD(rf)': 'purple'
}
default_colors = sns.color_palette("deep")

# Plot results
fig, ax = plt.subplots(facecolor='white')
for method in methods:
    pm, ps = PowerDict[method], PowerStdDevDict[method]
    ax.plot(NN1, pm, label=method)
    ax.fill_between(NN1, pm - ps, pm + ps, alpha=0.15)

ax.set_title("Power vs Sample-Size")
ax.set_ylabel('Power')
ax.set_xlabel('Sample-Size n1')

ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=10)

plt.tight_layout()
plt.savefig(f'./figure/PowerCurve_{rho1}_{b}_{timestamp}_styled.pdf')
plt.show()

end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")