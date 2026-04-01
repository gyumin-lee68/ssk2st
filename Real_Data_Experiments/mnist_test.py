import numpy as np
from tensorflow.keras.datasets import mnist

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from functools import partial
from tests import (crossMMD2sampleUnpaired, safe_crossSSMMD2sample, 
                   TwoSampleMMDSquared, crossSSMMD2sample)
from utils import (RBFkernel, RBFkernel1, median_bw_selector, get_bootstrap_threshold, 
                   get_normal_threshold, get_spectral_threshold, GaussianVector, Linearkernel, 
                   get_median_bw, normalize_vector)
import torch
from tqdm import tqdm
import pandas as pd

# Set random seed for reproducibility
RANDOM_SEED = 421
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Load data
(train_x_raw, train_y_labels), _ = mnist.load_data()

# Define the number of samples for each set (per digit)
n_labeled = 1000  # N1: Size of the shared subset (X/Y)
n_unlabeled = 4000  # N1 + M1: Total size of the V/W pool per digit

# Define the standard deviation of the Gaussian noise
# X/Y are clean, V/W are noisy (The covariate data)
NOISE_STD_DEV_COVARIATE = 1.5  # V, W: High Noise (The covariate data quality)

DIGITS_GROUP1 = [0, 1, 2, 3, 5, 8] # Group 1 [0, 1, 2, 3, 4, 5, 9] [0, 1, 2, 3, 9] [0, 1, 2, 3, 5, 8]
DIGITS_GROUP2 = [0, 1, 2, 3, 5, 9] # Group 2 [0, 1, 2, 3, 4, 5, 8] [0, 1, 2, 3, 6] [0, 1, 2, 3, 5, 9]


def prepare_and_merge_data_covariate(digits_list, noise_std_v):
    labeled_data_list = []
    covariate_data_list = []
    
    for digit in digits_list:
        # Filter data for the current digit
        data_indices = np.where(train_y_labels == digit)[0]

        # Ensure enough data for n_unlabeled (4000)
        if len(data_indices) < n_unlabeled:
             print(f"Warning: Not enough samples for digit {digit}. Available: {len(data_indices)}, Needed: {n_unlabeled}")
             continue
        
        # Shuffle the indices once to ensure randomness across different experimental runs
        np.random.shuffle(data_indices)
        
        # N1: Indices for the Labeled subset (X/Y) and the first part of V/W
        n1_indices = data_indices[:n_labeled]
        # M1: Indices for the Unlabeled subset (V/W-only part)
        m1_indices = data_indices[n_labeled:n_unlabeled] 
        
        # Get the N1 shared raw images
        x_images_raw = train_x_raw[n1_indices] 
        
        # Normalize and flatten (784 dimensions). NO NOISE ADDED.
        # This order (based on n1_indices) defines the order for the first 1000 samples of the digit.
        x_clean = x_images_raw.astype('float32') / 255.0
        labeled_data_list.append(x_clean.reshape((x_clean.shape[0], -1)))
        
        # Combine N1 and M1 indices to get the full V/W pool.
        # CRUCIAL: n1_indices MUST come first to ensure the first N1 samples in V/W match the order of X/Y.
        v_w_indices_ordered = np.concatenate([n1_indices, m1_indices])
        v_w_images_raw = train_x_raw[v_w_indices_ordered] 

        # Normalize the full pool
        v_w_norm = v_w_images_raw.astype('float32') / 255.0
        
        # Add HIGH Noise (Covariate information)
        v_w_noise = np.random.normal(0, noise_std_v, size=v_w_norm.shape)
        v_w_noisy = np.clip(v_w_norm + v_w_noise, 0.0, 1.0)

        # Flatten the noisy covariate images (784 dimensions)
        # This V/W data is now ordered such that the first 1000 samples match the X/Y order.
        covariate_data_list.append(v_w_noisy.reshape((v_w_noisy.shape[0], -1)))
        
    # Merge all data into single arrays. Since both lists maintain the sequential digit order (0, 1, 2, 3),
    # the final concatenated arrays (X, V) will have perfect matching for the first 4000 samples.
    merged_labeled = np.concatenate(labeled_data_list, axis=0)
    merged_covariate = np.concatenate(covariate_data_list, axis=0)
    
    return merged_labeled, merged_covariate

# Prepare data for Group 1 and assign to X, V
# X: Labeled (4000 samples total, 784D, Clean)
# V: Covariate (16000 samples total, 784D, Noisy)
X, V = prepare_and_merge_data_covariate(DIGITS_GROUP1, NOISE_STD_DEV_COVARIATE)

# Prepare data for Group 2 and assign to Y, W
# Y: Labeled (4000 samples total, 784D, Clean)
# W: Covariate (16000 samples total, 784D, Noisy)
Y, W = prepare_and_merge_data_covariate(DIGITS_GROUP2, NOISE_STD_DEV_COVARIATE)

# --- 5. Verification and Final Output ---

print("-" * 50)
print("Data Preparation Complete (X/Y are Clean Marginals, V/W are Noisy Covariates)")
print(f"Group 1 Digits: {DIGITS_GROUP1}, Group 2 Digits: {DIGITS_GROUP2}")
print("-" * 50)

print(f"Total samples in X: {X.shape[0]}, Total samples in V: {V.shape[0]}")
print(f"Total samples in Y: {Y.shape[0]}, Total samples in W: {W.shape[0]}")
print("-" * 50)

print("Final Dataset Shapes (All are Full Size 784D):")
print(f"Group 1 Labeled Set (X, Clean Marginal): {X.shape}")
print(f"Group 1 Covariate Set (V, Noisy Covariate): {V.shape}")
print(f"Group 2 Labeled Set (Y, Clean Marginal): {Y.shape}")
print(f"Group 2 Covariate Set (W, Noisy Covariate): {W.shape}")
print("-" * 50)


# run tests
def run_tests(K, L, X, Y, V, W, N1, N2, M1, M2, alpha=0.05):
    # Clear GPU cache at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Convert numpy arrays to PyTorch tensors
    X_start = torch.tensor(X, dtype=torch.float32)
    Y_start = torch.tensor(Y, dtype=torch.float32)
    V_start = torch.tensor(V, dtype=torch.float32)
    W_start = torch.tensor(W, dtype=torch.float32)
    
    tests = ['mmd-perm', 'mmd-perm2','xMMD', 'xssMMD(knn)', 'xssMMD(ker)', 'xssMMD(rf)']
    outputs = [[] for _ in range(len(tests))]
    stats = [[] for _ in range(len(tests))]
    total_iterations = K * L
    completed_iterations = 1
    
    # run K times
    for kk in tqdm(range(K)):
        # Clear GPU cache at the start of each K iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        torch.manual_seed(kk * 19)
        torch.cuda.manual_seed(kk * 19)
        np.random.seed(1102 * (kk + 10))

        # run L times
        for ll in tqdm(range(L)):
            
            # Sample N1 random indices for X and V (with controlled randomness)
            idx_x = torch.randperm(X_start.shape[0])[:N1]  # Randomly sample N1 indices
            # Calculate the remaining indices
            idx_v = torch.randperm(V_start.shape[0]-X_start.shape[0])[:M1]  # Randomly sample N1 indices

            X = X_start[idx_x]
            V_first_part = V_start[idx_x]  
            V_remaining_part = V_start[idx_v+X_start.shape[0]]  
            V = torch.cat([V_first_part, V_remaining_part], dim=0) 
            
            # Sample N2 random indices for Y and W (with controlled randomness)
            idx_y = torch.randperm(Y_start.shape[0])[:N2]     
            # Calculate the remaining indices
            idx_w = torch.randperm(W_start.shape[0]-Y_start.shape[0])[:M2] 

            Y = Y_start[idx_y]
            W_first_part = W_start[idx_y] 
            W_remaining_part = W_start[idx_w] 
            W = torch.cat([W_first_part, W_remaining_part], dim=0) 
            
            # Kernel bandwidth
            bw = get_median_bw(X=X, Y=Y)
            bw2 = get_median_bw(X=V, Y=W)
            kernel_type = RBFkernel  # Choose RBFkernel or Linearkernel
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
            
            retry = True  # Initialize retry fla

            while retry:
                retry = False  
                # Perform tests
                for i, test in enumerate(tests):
                    try:
                        if test == 'mmd-perm':
                            stat = unbiased_mmd2(X, Y, kernel_func)
                            th = thresh_permutation(X, Y, kernel_func, unbiased_mmd2, alpha=alpha)
                        elif test == 'mmd-perm2':
                            stat = unbiased_mmd2(V, W, kernel_func2)
                            th = thresh_permutation(V, W, kernel_func2, unbiased_mmd2, alpha=alpha)
                        elif test == 'xMMD':
                            stat = cross_mmd2(X, Y, kernel_func)
                            th = thresh_normal(alpha)
                        elif test == 'xssMMD(knn)':
                            stat = crossSSMMD2sample(X, V, Y, W, kernel_func, "KNN")
                            th = thresh_normal(alpha)
                        elif test == 'xssMMD(ker)':
                            stat = crossSSMMD2sample(X, V, Y, W, kernel_func, "KernelRegression")
                            th = thresh_normal(alpha)
                        elif test == 'xssMMD(rf)':
                            stat = crossSSMMD2sample(X, V, Y, W, kernel_func, "RandomForest")
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
    
    # Print parameter settings
    print("-" * 50)
    print("EXPERIMENT PARAMETERS:")
    print(f"Digits Group 1: {DIGITS_GROUP1}")
    print(f"Digits Group 2: {DIGITS_GROUP2}")
    print(f"n_labeled (N1): {n_labeled}")
    print(f"n_unlabeled (N1 + M1): {n_unlabeled}")
    print(f"NOISE_STD_DEV_COVARIATE: {NOISE_STD_DEV_COVARIATE}")
    print(f"Test Parameters - K: {K}, L: {L}, N1: {N1}, N2: {N2}, M1: {M1}, M2: {M2}")
    print(f"Alpha (significance level): {alpha}")
    print("-" * 50)
    
    # Print results
    print("TEST RESULTS:")
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
run_tests(K=100, L=10, X=X, Y=Y, V=V, W=W, N1=200, N2=200, M1=200, M2=200)