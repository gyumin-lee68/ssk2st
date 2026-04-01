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
meanY[:num_pert] = (eps)
meanV, meanW = np.zeros((d2,)), np.zeros((d2,))
covX =  np.eye(d1) + rho1 * (np.ones((d1, d1)) - np.eye(d1))
covY =  np.eye(d1) + rho2 * (np.ones((d1, d1)) - np.eye(d1))
covV =  np.eye(d2)
covW =  np.eye(d2)

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
PowerDict = {method: np.zeros((num_trials, len(NN1))) for method in methods}
PowerStdDevDict = {method: np.zeros(NN1.shape) for method in methods}
TimeDict = {method: np.zeros((num_trials, len(NN1))) for method in methods}
AvgTimeDict = {} 
# ---------------------------------------------
    
start_time = time.time()

# Main loop for trials
for i in tqdm(range(num_trials)):
    for j, (n1i, n2i, m1i, m2i) in enumerate(zip(NN1, NN2, MM1, MM2)):
        X = SourceX(n1i)
        Y = SourceY(n2i)
        
        V_new = SourceV(m1i)
        V = np.vstack([X[:, [-2, -1]], V_new[:, [-2, -1]]])
        
        W_new = SourceW(m2i)
        W = np.vstack([Y[:, [-2, -1]], W_new[:, [-2, -1]]])
        
        XV = np.hstack([X, V[:n1i]]) 
        YW = np.hstack([Y, W[:n2i]]) 


        # Obtain the bandwidth of the kernel
        bw_marginal = get_median_bw(XV, YW)
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
pickle.dump(PowerDict, open(f'./PowerDict_{rho1}_{timestamp}.pkl', 'wb'))
pickle.dump(PowerStdDevDict, open(f'./PowerStdDevDict_{rho1}_{timestamp}.pkl', 'wb'))


print("\n" + "="*85)
print(f"Power Summary (Alpha={alpha})")
print("="*85)

header = f"{'Sample (n, m)':<22}" + "".join([f"{m:>13}" for m in methods])
print(header)
print("-" * len(header))

# printing the row
for j in range(len(NN1)):
    n_val = NN1[j]
    m_val = MM1[j]
    row_label = f"n={n_val}, m={m_val}"
    
    row_str = f"{row_label:<22}"
    
    for method in methods:
        val = PowerDict[method][j]
        row_str += f"{val:>13.3f}" 
    
    print(row_str)

print("="*85 + "\n")
# ========================================


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

for i, method in enumerate(methods):
    pm, ps = PowerDict[method], PowerStdDevDict[method]
    
    this_color = color_map.get(method, default_colors[i % len(default_colors)])
    ax.plot(NN1, pm, label=method, color=this_color, linewidth=1.5)
    ax.fill_between(NN1, pm - ps, pm + ps, color=this_color, alpha=0.2, edgecolor=None)


ax.set_title(f"Scenario 1 (Alt)", fontsize=14)
ax.set_ylabel('Power', fontsize=12)
ax.set_xlabel('Sample-Size n1', fontsize=12)

ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=10)

plt.tight_layout()
plt.savefig(f'./figure/PowerCurve_{rho1}_{timestamp}_styled.pdf')
plt.show()

end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")
