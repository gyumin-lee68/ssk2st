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
        # Alternative 세팅: X와 Y가 다름
        meanX, meanY = np.zeros((d1,)), np.zeros((d1,))
        meanX[:num_pert] = eps
        # rho에 따라 매번 공분산 행렬 생성
        cov = np.eye(d1) + rho * (np.ones((d1, d1)) - np.eye(d1))
        
        X = GaussianVector(mean=meanX, cov=cov, n=n_fixed)
        Y = GaussianVector(mean=meanY, cov=cov, n=n_fixed)
        X_new = GaussianVector(mean=meanX, cov=cov, n=m_fixed)
        Y_new = GaussianVector(mean=meanY, cov=cov, n=m_fixed)
        
        # V와 W는 완전히 동일한 분포 (Shift 없음)
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

# === 결과 요약 및 LaTeX 자동 생성 ===
print("\n" + "="*85)
print(f"Power Summary (Varying rho, Alpha={alpha})")
print("="*85)
header = f"{'rho':<15}" + "".join([f"{m:>13}" for m in methods])
print(header)
print("-" * len(header))
for j, rho in enumerate(rho_list):
    row_str = f"{rho:<15.2f}"
    for method in methods:
        row_str += f"{PowerDict[method][j]:>13.3f}"
    print(row_str)
print("="*85 + "\n")

print("LaTeX Table Code for Overleaf:\n")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\footnotesize")
print(r"\setlength{\tabcolsep}{4pt}")
print(r"\caption{Estimated power under the alternative ($P_X \neq P_Y, P_V = P_W$) across varying correlation strengths $\rho$ between target and auxiliary covariates.}")
print(r"\label{tab:power_varying_rho}")
print(r"\begin{tabular}{l" + "c" * len(rho_list) + "}")
print(r"\toprule")
print(r"Method \textbackslash \ $\rho$ & " + " & ".join([f"${r:.2f}$" for r in rho_list]) + r" \\")
print(r"\midrule")
for method in methods:
    print(f"{method:15s} & " + " & ".join([f"{PowerDict[method][j]:.3f}" for j in range(len(rho_list))]) + r" \\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
print("\n" + "="*85 + "\n")

# Plotting
fig, ax = plt.subplots(figsize=(6, 5))
ax.grid(True)
color_map = {'MMD-perm': 'blue', 'xMMD': 'green', 'xssMMD(knn)': 'red', 'xssMMD(ker)': 'orange', 'xssMMD(rf)': 'purple'}
for method in methods:
    pm, ps = PowerDict[method], PowerStdDevDict[method]
    ax.plot(rho_list, pm, label=method, color=color_map[method], linewidth=1.5, marker='s')
    ax.fill_between(rho_list, pm - ps, pm + ps, color=color_map[method], alpha=0.2)

ax.set_title(r"Power under Alternative ($P_X \neq P_Y, P_V = P_W$)", fontsize=13)
ax.set_ylabel('Power', fontsize=12)
ax.set_xlabel(r'Correlation between Target and Auxiliary ($\rho$)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(f'./figure/Power_VaryingRho_{timestamp}_styled.pdf')
plt.show()

print(f"Elapsed time: {time.time() - start_time} seconds")
