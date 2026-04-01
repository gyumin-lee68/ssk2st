from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
from functools import partial
from tests import (crossMMD2sampleUnpaired, safe_crossSSMMD2sample, 
                   TwoSampleMMDSquared, crossSSMMD2sample)
from utils import (RBFkernel, RBFkernel1, median_bw_selector, get_bootstrap_threshold, 
                   get_normal_threshold, get_spectral_threshold, GaussianVector, Linearkernel, 
                   get_median_bw, normalize_vector)
import torch
from tqdm import tqdm
import pandas as pd
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
htru2 = fetch_ucirepo(id=372) 
  
# data (as pandas dataframes) 
data = htru2.data.features 
label = htru2.data.targets 

# Split data into pulsar and nonpulsar
X_pulsar = data[label.values == 1]
X_nonpulsar = data[label.values == 0]
X_pulsar_np = X_pulsar.to_numpy()
X_nonpulsar_np = X_nonpulsar.to_numpy()

# Set covariates used for your labeled data
labeled_covariates_idx = [0,1,4,5]  # IPMean, SD, DM(Mean, SD)

X = X_pulsar_np[:, labeled_covariates_idx] 
Y = X_nonpulsar_np[:, labeled_covariates_idx]

# unlabeled covariates
unlabeled_covariates_idx = list(set(range(data.shape[1])) - set(labeled_covariates_idx))
unlabeled_covariates_idx_V = [0,4]
unlabeled_covariates_idx_W = [0,4]

V = X_pulsar_np[:, unlabeled_covariates_idx_V]  
W = X_nonpulsar_np[:, unlabeled_covariates_idx_W]

# run tests
def run_tests(K, L, X, Y, V, W, N1, N2, alpha=0.05):
    # Clear GPU cache at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Convert numpy arrays to PyTorch tensors
    X_start = torch.tensor(X, dtype=torch.float32)
    Y_start = torch.tensor(Y, dtype=torch.float32)
    V_start = torch.tensor(V, dtype=torch.float32)
    W_start = torch.tensor(W, dtype=torch.float32)
    
    tests = ['mmd-perm', 'xMMD', 'xssMMD(knn)', 'xssMMD(ker)', 'xssMMD(rf)']
    outputs = [[] for _ in range(len(tests))]
    stats = [[] for _ in range(len(tests))]
    total_iterations = K * L
    completed_iterations = 0

    # std of Gaussian noise
    noise_std = 1
    
    # run K times
    for kk in tqdm(range(K)):
        # Clear GPU cache at the start of each K iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        torch.manual_seed(kk * 19)
        torch.cuda.manual_seed(kk * 19)
        np.random.seed(1102 * (kk + 10))

        # Sample N1 random indices
        idx1 = torch.randperm(X_start.shape[0])[:N1]  # Randomly sample N1 indices
        # Calculate the remaining indices
        all_indices1 = torch.arange(0, X_start.shape[0])  
        remaining_indices1 = all_indices1[~torch.isin(all_indices1, idx1)]  

        X = X_start[idx1]
        center1 = X.mean(axis=0, keepdims=True)  
        X_cen = X - center1 
        X_normalize = np.apply_along_axis(normalize_vector, axis=0, arr=X_cen)
          
        gaussian_noise_X = np.random.normal(loc=0, scale=1, size=X_normalize.shape) 
        X_noisy = torch.tensor(X_normalize + noise_std * gaussian_noise_X, dtype=torch.float32)  
        
        V_first_part = V_start[idx1]  
        V_remaining_part = V_start[remaining_indices1] 
        V = torch.cat([V_first_part, V_remaining_part], dim=0) 
        center3 = V.mean(axis=0, keepdims=True)  
        V_cen = V - center3 
        V_normalize = torch.tensor(np.apply_along_axis(normalize_vector, axis=0, arr=V_cen), dtype=torch.float32)
          
        # run L times
        for ll in tqdm(range(L)):
            
            idx2 = torch.randperm(Y_start.shape[0])[:N2]  
            all_indices2 = torch.arange(0, Y_start.shape[0]) 
            remaining_indices2 = all_indices2[~torch.isin(all_indices2, idx2)]  

            Y = Y_start[idx2]
            center2 = Y.mean(axis=0, keepdims=True) 
            Y_cen = Y - center2  
            Y_normalize = np.apply_along_axis(normalize_vector, axis=0, arr=Y_cen)
            
            gaussian_noise_Y = np.random.normal(loc=0, scale=1, size=Y_normalize.shape)  
            Y_noisy = torch.tensor(Y_normalize + noise_std * gaussian_noise_Y, dtype=torch.float32) 
            
            W_first_part = W_start[idx2]  
            W_remaining_part = W_start[remaining_indices2]  
            W = torch.cat([W_first_part, W_remaining_part], dim=0) 
            center4 = W.mean(axis=0, keepdims=True)  
            W_cen = W - center4  
            W_normalize = torch.tensor(np.apply_along_axis(normalize_vector, axis=0, arr=W_cen), dtype=torch.float32)

            # Kernel bandwidth
            bw = get_median_bw(X=X_noisy, Y=Y_noisy)
            bw2 = get_median_bw(X=V_normalize, Y=W_normalize)
            kernel_type = RBFkernel 
            kernel_func = None
            if kernel_func is None:  # default is to use the RBF kernel
                if kernel_type == RBFkernel or kernel_type is None:
                    kernel_type = RBFkernel  # just in case it is None
                    kernel_func = partial(RBFkernel1, bw=bw)
                    kernel_func2 = partial(RBFkernel1, bw=bw2)
                elif kernel_type == Linearkernel:
                    kernel_func = partial(Linearkernel)

            unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True)
            biased_mmd2 = partial(TwoSampleMMDSquared, unbiased=False)
            cross_mmd2 = crossMMD2sampleUnpaired
            
            retry = True  

            while retry:
                retry = False 
                # Perform tests
                for i, test in enumerate(tests):
                    try:
                        if test == 'mmd-perm':
                            stat = unbiased_mmd2(X_noisy, Y_noisy, kernel_func)
                            th = thresh_permutation(X_noisy, Y_noisy, kernel_func, unbiased_mmd2, alpha=alpha)
                        elif test == 'mmd-perm2':
                            stat = unbiased_mmd2(V_normalize, W_normalize, kernel_func2)
                            th = thresh_permutation(V_normalize, W_normalize, kernel_func2, unbiased_mmd2, alpha=alpha)
                        elif test == 'xMMD':
                            stat = cross_mmd2(X_noisy, Y_noisy, kernel_func)
                            th = thresh_normal(alpha)
                        elif test == 'xssMMD(knn)':
                            stat = crossSSMMD2sample(X_noisy, V_normalize, Y_noisy, W_normalize, kernel_func, "KNN")
                            th = thresh_normal(alpha)
                        elif test == 'xssMMD(ker)':
                            stat = crossSSMMD2sample(X_noisy, V_normalize, Y_noisy, W_normalize, kernel_func, "KernelRegression")
                            th = thresh_normal(alpha)
                        elif test == 'xssMMD(rf)':
                            stat = crossSSMMD2sample(X_noisy, V_normalize, Y_noisy, W_normalize, kernel_func, "RandomForest")
                            th = thresh_normal(alpha)
                        else:
                            raise ValueError(f"Unknown test type: {test}")

                        # Append stat and result
                        stats[i].append(stat)
                        outputs[i].append((stat > th).float().item())
                    except Exception as e:
                        print(f"Error in '{test}' test: {e}")
                        retry = True  
                        break  

                if retry:
                    print(f"Retrying iteration {kk + 1}, {ll + 1}...")
                    break  
                else:
                    break  

            # Increment completed iteration counter
            completed_iterations += 1

            # Clear GPU cache after each L iteration to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Ensure that the total number of iterations reaches K * L
            if completed_iterations >= total_iterations:
                print(f"Completed {total_iterations} iterations.")
                break
    
    # Calculate mean results
    mean_outputs = [np.mean(output) for output in outputs]
    
    # Print results
    for i, test in enumerate(tests):
        print(f"Test: {test}, Mean result: {mean_outputs[i]}")
    
    return stats, outputs

# Print shapes
print(f"X shape: {X.shape}, V shape: {V.shape}")
print(f"Y shape: {Y.shape}, W shape: {W.shape}")

#set up function handles for different threshold computing methods
num_perms = 200
thresh_permutation = partial(get_bootstrap_threshold, num_perms=num_perms)
thresh_normal = get_normal_threshold
thresh_spectral = partial(get_spectral_threshold,  alpha=0.05, numNullSamp=200)

# Run two-sample test
run_tests(K=50, L=20, X=X, Y=Y, V=V, W=W, N1=100, N2=100)