from datetime import datetime
from functools import partial
import numpy as np
import torch
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsRegressor
from tests import (crossMMD2sampleUnpaired, safe_crossSSMMD2sample, 
                   TwoSampleMMDSquared, crossSSMMD2sample)
from utils import (RBFkernel, RBFkernel1, median_bw_selector, median_bw_selector_2, 
                  get_bootstrap_threshold, get_normal_threshold, get_spectral_threshold, 
                  GaussianVector, get_bootstrap_std, Linearkernel, 
                  get_median_bw, TVector, MultiExpVector)
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns
from tqdm import tqdm
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode

# Ensure output directory exists
output_dir = './figure/'
os.makedirs(output_dir, exist_ok=True)

# Set the sample size, dimension, kernel type,and degrees of freedom, if needed.
n1, m1, n2, m2, d, df = 200, 200, 200, 200, 10, 30

# Set the kernel type. Linearkernel for bilinear kernel, RBFkernel for RBF kernel
kernel_type= RBFkernel 

# Set the correlation coefficient and the covariance matrix.
rho = 0.95
M = np.eye(d) + rho * (np.ones((d, d)) - np.eye(d))

# Set the noise level.
eps= [0.0]

result1, result2, result3 = [], [], []

# Generate the null distribution.
for ep in tqdm(eps):
    def perform_iteration(i):
        
        V = MultiExpVector(scale=np.array([0.1 * i for i in range(d)]), n=2*n1 + m1)
        X = MultiExpVector(scale=np.array([0.1 * i for i in range(d)]), n=2*n1)
        
        W = MultiExpVector(scale=np.array([0.1 * i for i in range(d)]), n=2*n2 + m2)
        Y = MultiExpVector(scale=np.array([0.1 * i for i in range(d)]), n=2*n2)
        
        X, V, Y, W = torch.tensor(X, dtype=torch.float32, device='cpu'), torch.tensor(V, dtype=torch.float32, device='cpu'), torch.tensor(Y, dtype=torch.float32, device='cpu'), torch.tensor(W, dtype=torch.float32, device='cpu')
        
        bw = get_median_bw(X, Y)
        
        kernel_func = None
        if kernel_func is None: # default is to use the RBF kernel
            global kernel_type
            if kernel_type==RBFkernel or kernel_type is None:
                kernel_type=RBFkernel # just in case it is None
                kernel_name = 'RBFkernel'
                kernel_func = partial(RBFkernel1, bw=bw) #bw=bw

            elif kernel_type==Linearkernel:
                kernel_name = 'Linearkernel'
                kernel_func = partial(Linearkernel)
        
        T_xss_knn = safe_crossSSMMD2sample(X,V,Y,W,kernel_func, "KNN")
        T_xss_ker = safe_crossSSMMD2sample(X,V,Y,W,kernel_func, "KernelRegression")
        T_xss_rf = safe_crossSSMMD2sample(X,V,Y,W,kernel_func, "RandomForest")
        
        # Print the results every 1000 iterations
        if i % 1000 == 0:  
            print(f"Iteration {i}:")
            print(f"KNN: {T_xss_knn}")
            print(f"Kernel: {T_xss_ker}")
            print(f"RandomForest: {T_xss_rf}")

        return T_xss_knn, T_xss_ker, T_xss_rf 
    
    total_iterations = 10000
    batch_size = 1000
    results = []
    
    for batch in tqdm(range(0, total_iterations, batch_size)):
        batch_results = Parallel(n_jobs=-1)(
            delayed(perform_iteration)(i) 
            for i in range(batch, min(batch + batch_size, total_iterations))
        )
        results.extend(batch_results)
        
    T_xss_knn, T_xss_ker, T_xss_rf = zip(*results) 

    result1.append(T_xss_knn)
    result2.append(T_xss_ker)
    result3.append(T_xss_rf)
    
# Save PowerDict and PowerStdDevDict to pickle files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_to_save = {
    "result1": result1,
    "result2": result2,
    "result3": result3
}
pickle.dump(data_to_save, open(f'./NullDist_{kernel_type.__name__}_n1_{n1/2}_m1_{m1}_n2_{n2/2}_m2_{m2}_d_{d}_{timestamp}.pkl', 'wb'))

# Prepare to plot the results
xx = np.linspace(-10, 10, 10000)
pp = stats.norm.pdf(xx) # the normal pdf

# Plot the null distribution of xMMD statistic
fig = plt.figure()
fig.patch.set_facecolor('white') 
ax = fig.add_subplot(111)
ax.hist(x=[result1[0], result2[0], result3[0]], density=True, alpha=0.8, 
        label=['xssMMD(kNN)', 'xssMMD(kernel)', 'xssMMD(rf)'], 
        color=['dodgerblue', 'tomato', 'limegreen'])
ax.plot(xx, pp, color='k')  
ax.set_ylabel('Probability density', fontsize=16)
ax.set_title(f'SS-xMMD (n1={n1/2}, n2={n2/2}, m1={m1}, m2={m2},dim={d})', fontsize=14)

ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
ax.set_facecolor('white')  
ax.tick_params(axis='both', which='major', labelsize=12)  
ax.set_xlim(-5,5)

for spine in ax.spines.values():
    spine.set_edgecolor('black')  
    spine.set_linewidth(1.5) 
    
figname = None
if figname is None:
    figname = f'NullDist_{kernel_type.__name__}_n1_{n1/2}_m1_{m1}_n2_{n2/2}_m2_{m2}_d_{d}'
    timestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    figname = './figure/' + figname + timestr + '.pdf'
    
# Save the figure
figname = f'NullDist_{kernel_type.__name__}_n1_{n1/2}_m1_{m1}_n2_{n2/2}_m2_{m2}_d_{d}_{timestamp}.pdf'
plt.savefig(os.path.join(output_dir, figname), facecolor=fig.get_facecolor())