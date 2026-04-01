from datetime import datetime
from tests import (crossMMD2sampleUnpaired, safe_crossSSMMD2sample, 
                   TwoSampleMMDSquared)
from utils import (RBFkernel, RBFkernel1, get_bootstrap_threshold, 
                   get_normal_threshold, get_spectral_threshold, 
                   get_bootstrap_std, GaussianVector, get_median_bw)
from functools import partial
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
d1, d2 = 10, 10  
eps = 0.3  # Magnitude of perturbation for X (Alternative)
num_pert = 1  

n_fixed, m_fixed = 100, 1000 
alpha = 0.05  
num_bootstrap = 200  
num_perms = 200  
num_trials = 1000  
kernel_type = RBFkernel 

# List of Correlations ---
rho_list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]

thresh_permutation = partial(get_bootstrap_threshold, num_perms=num_perms)
thresh_normal = get_normal_threshold

methods = ['MMD-perm', 'xMMD', 'xssMMD(knn)','xssMMD(ker)', 'xssMMD(rf)']

PowerDict = {method: np.zeros((num_trials, len(rho_list))) for method in methods}
PowerStdDevDict = {method: np.zeros(len(rho_list)) for method in methods}
TimeDict = {method: np.zeros((num_trials, len(rho_list))) for method in methods}

start_time = time.time()

# Main loop for trials
for i in tqdm(range(num_trials)):
    for j, rho in enumerate(rho_list):
      
        meanX, meanY = np.zeros((d1,)), np.zeros((d1,))
        meanX[:num_pert] = eps
        
        cov = np.eye(d1) + rho * (np.ones((d1, d1)) - np.eye(d1))
        
        X = GaussianVector(mean=meanX, cov=cov, n=n_fixed)
        Y = GaussianVector(mean=meanY, cov=cov, n=n_fixed)
        X_new = GaussianVector(mean=meanX, cov=cov, n=m_fixed)
        Y_new = GaussianVector(mean=meanY, cov=cov, n=m_fixed)
        
        # V and W are independent
        V = np.vstack([X[:, [-2, -1]], X_new[:, [-2, -1]]])
        W = np.vstack([Y[:, [-2, -1]], Y_new[:, [-2, -1]]])
        
        XV = np.hstack([X, V[:n_fixed]]) 
        YW = np.hstack([Y, W[:n_fixed]])

        bw_marginal = get_median_bw(X, Y) 
        bw_joint = get_median_bw(XV, YW)
        kernel_func_joint = partial(RBFkernel1, bw=bw_joint) if kernel_type == RBFkernel else partial(Linearkernel)
        kernel_func_marginal = partial(RBFkernel1, bw=bw_marginal) if kernel_type == RBFkernel else partial(Linearkernel)

        unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True)
        cross_mmd2 = crossMMD2sampleUnpaired
        
        X_t, V_t, Y_t, W_t = torch.tensor(X), torch.tensor(V), torch.tensor(Y), torch.tensor(W)
        XV_t, YW_t = torch.tensor(XV), torch.tensor(YW)
        
        for method in methods:
            t_method_start = time.time()
            
            if method=='MMD-perm':
                stat = unbiased_mmd2(XV_t, YW_t, kernel_func_joint)
                th = thresh_permutation(XV_t, YW_t, kernel_func_joint, unbiased_mmd2, alpha=alpha)
            elif method=='xMMD':
                stat = cross_mmd2(XV_t, YW_t, kernel_func_joint)
                th = thresh_normal(alpha)
            elif method == 'xssMMD(knn)':                
                stat = safe_crossSSMMD2sample(X_t,V_t,Y_t,W_t, kernel_func_marginal, "KNN")
                th = thresh_normal(alpha)
            elif method == 'xssMMD(ker)':
                stat = safe_crossSSMMD2sample(X_t,V_t,Y_t,W_t, kernel_func_marginal, "KernelRegression")
                th = thresh_normal(alpha)
            elif method == 'xssMMD(rf)':
                stat = safe_crossSSMMD2sample(X_t,V_t,Y_t,W_t, kernel_func_marginal, "RandomForest")
                th = thresh_normal(alpha)
            
            TimeDict[method][i][j] = time.time() - t_method_start
            PowerDict[method][i][j] = 1.0*(stat>th)
    
# Compute mean and std 
for method in methods:
    PowerStdDevDict[method] = np.array([
        get_bootstrap_std(PowerDict[method][:, i], num_bootstrap=num_bootstrap)
        for i in range(len(rho_list))
    ])
    PowerDict[method] = PowerDict[method].mean(axis=0)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pickle.dump(PowerDict, open(f'./PowerDict_rho_{timestamp}.pkl', 'wb'))

print(f"Elapsed time: {time.time() - start_time} seconds")
