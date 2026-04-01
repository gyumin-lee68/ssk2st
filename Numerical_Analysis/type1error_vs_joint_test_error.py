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

# Set device for computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
d1, d2 = 10, 10  
rho = 0.95  
eps = 0.3  # Magnitude of perturbation for X and Y (Null)
num_pert = 1  

# 고정된 샘플 사이즈 설정
n_fixed, m_fixed = 100, 1000 
alpha = 0.05  
num_bootstrap = 200  
num_perms = 200  
num_trials = 1000  
kernel_type = RBFkernel 

# --- 우리가 변화시킬 Nuisance Shift 리스트 ---
eps_V_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Function handles for threshold computing methods
thresh_permutation = partial(get_bootstrap_threshold, num_perms=num_perms)
thresh_normal = get_normal_threshold

poly_degree = 2
split = 2

methods = ['MMD-perm', 'xMMD', 'xssMMD(knn)','xssMMD(ker)', 'xssMMD(rf)']

# Initialize the dictionaries
TypeIErrorDict = {method: np.zeros((num_trials, len(eps_V_list))) for method in methods}
TypeIErrorStdDict = {method: np.zeros(len(eps_V_list)) for method in methods}
TimeDict = {method: np.zeros((num_trials, len(eps_V_list))) for method in methods}

start_time = time.time()

# Main loop for trials
for i in tqdm(range(num_trials)):
    for j, eps_V in enumerate(eps_V_list):
        # Null 세팅: X와 Y가 완벽히 동일
        meanX, meanY = np.zeros((d1,)), np.zeros((d1,))
        meanX[:num_pert] = eps
        meanY[:num_pert] = eps
        cov = np.eye(d1) + rho * (np.ones((d1, d1)) - np.eye(d1))
        
        X = GaussianVector(mean=meanX, cov=cov, n=n_fixed)
        Y = GaussianVector(mean=meanY, cov=cov, n=n_fixed)
        X_new = GaussianVector(mean=meanX, cov=cov, n=m_fixed)
        Y_new = GaussianVector(mean=meanY, cov=cov, n=m_fixed)
        
        # V에만 eps_V 만큼의 노이즈 추가
        V = np.vstack([X[:, :2], X_new[:, :2]])
        V[:, 0] += eps_V 
        W = np.vstack([Y[:, [-2, -1]], Y_new[:, [-2, -1]]])
        
        XV = np.hstack([X, V[:n_fixed]]) 
        YW = np.hstack([Y, W[:n_fixed]])

        # Obtain the bandwidth of the kernel (버그 수정됨)
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
            TypeIErrorDict[method][i][j] = 1.0*(stat>th)
    
# Compute mean and std 
for method in methods:
    TypeIErrorStdDict[method] = np.array([
        get_bootstrap_std(TypeIErrorDict[method][:, i], num_bootstrap=num_bootstrap)
        for i in range(len(eps_V_list))
    ])
    TypeIErrorDict[method] = TypeIErrorDict[method].mean(axis=0)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pickle.dump(TypeIErrorDict, open(f'./TypeIErrorDict_epsV_{timestamp}.pkl', 'wb'))

# === 결과 요약 및 LaTeX 자동 생성 ===
print("\n" + "="*85)
print(f"Type-I Error Rate Summary (Varying eps_V, Alpha={alpha})")
print("="*85)
header = f"{'eps_V':<15}" + "".join([f"{m:>13}" for m in methods])
print(header)
print("-" * len(header))
for j, eps_V in enumerate(eps_V_list):
    row_str = f"{eps_V:<15.1f}"
    for method in methods:
        row_str += f"{TypeIErrorDict[method][j]:>13.3f}"
    print(row_str)
print("="*85 + "\n")

print("LaTeX Table Code for Overleaf:\n")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\footnotesize")
print(r"\setlength{\tabcolsep}{4pt}")
print(r"\caption{Type-I error rate of the joint and marginal tests under the null ($P_X=P_Y, P_V \neq P_W$) with varying nuisance shift magnitude $\epsilon_V$.}")
print(r"\label{tab:type1_varying_eps}")
print(r"\begin{tabular}{l" + "c" * len(eps_V_list) + "}")
print(r"\toprule")
print(r"Method \textbackslash \ $\epsilon_V$ & " + " & ".join([f"${e:.1f}$" for e in eps_V_list]) + r" \\")
print(r"\midrule")
for method in methods:
    print(f"{method:15s} & " + " & ".join([f"{TypeIErrorDict[method][j]:.3f}" for j in range(len(eps_V_list))]) + r" \\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
print("\n" + "="*85 + "\n")

# Plotting
fig, ax = plt.subplots(figsize=(6, 5))
ax.grid(True)
color_map = {'MMD-perm': 'blue', 'xMMD': 'green', 'xssMMD(knn)': 'red', 'xssMMD(ker)': 'orange', 'xssMMD(rf)': 'purple'}
for method in methods:
    pm, ps = TypeIErrorDict[method], TypeIErrorStdDict[method]
    ax.plot(eps_V_list, pm, label=method, color=color_map[method], linewidth=1.5, marker='o')
    ax.fill_between(eps_V_list, pm - ps, pm + ps, color=color_map[method], alpha=0.2)

ax.axhline(y=0.05, color='black', linestyle='--', label='Nominal Level (0.05)')
ax.set_title(r"Type-I Error under Joint Null ($P_X=P_Y, P_V \neq P_W$)", fontsize=13)
ax.set_ylabel('Type-I Error Rate', fontsize=12)
ax.set_xlabel(r'Nuisance Shift Magnitude ($\epsilon_V$)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(f'./figure/TypeIError_VaryingEpsV_{timestamp}_styled.pdf')
plt.show()

print(f"Elapsed time: {time.time() - start_time} seconds")