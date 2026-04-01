import torch
from math import sqrt, log
from functools import partial
import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_t, laplace
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer

from copy import deepcopy
from scipy.linalg import solve as scp_solve
from warnings import warn
plt.style.use('seaborn-v0_8')

###====================Kernel Functions=======================
def RBFkernel(x, y=None, bw=1.0, amp=1.0):
    """
        k(x, y) = amp * exp( - \|x-y\|^2 / (2*bw^2))
    """
    y = x if y is None else y
    dists = cdist(x, y)
    squared_dists = dists * dists
    k = amp * np.exp( -(1/(2*bw*bw)) * squared_dists )
    return k

def ConstantKernel(x, y=None, c=None):
    """
        k(x, y) = c
    """
    y = x if y is None else y
    c = 1 if c is None else c
    k = np.einsum('ji, ki -> jk', x, y)*0 + c
    return k

def LinearKernel(x, y=None, c=None):
    """
        k(x, y) = x^T y + c
    """
    y = x if y is None else y
    c = 0 if c is None else c
    k = np.einsum('ji, ki -> jk', x, y) + c
    return k


def RBFkernel1(x, y=None, bw=1.0, amp=1.0, pairwise=False):
    """
    PyTorch version of the RBF kernel.
    k(x, y) = amp * exp(-||x - y||^2 / (2 * bw^2))
    """
    y = x if y is None else y

    if pairwise:
        assert y.shape == x.shape
        squared_dists = torch.sum((y - x) ** 2, axis=1)
    else:
        # Calculate pairwise squared Euclidean distances
        dists = torch.cdist(x, y)
        squared_dists = dists ** 2

    k = amp * torch.exp(-squared_dists / (2 * bw * bw))

    return k

def Linearkernel(x, y=None, c=None):
    """
    k(x, y) = x^T y + c
    """

    y = x if y is None else y
    
    c = 0 if c is None else c
    
    # Use einsum to compute x^T y
    k = torch.einsum('ji, ki -> jk', x, y) + c
    
    return k

###====================Bandwidth selection utils=======================
def get_median_bw(Z=None, X=None, Y=None):
    """
    Return the median of the pairwise distances (in terms of L2 norm).
    """
    if Z is None:
        assert (X is not None) and (Y is not None)
        Z = np.concatenate([X, Y], axis=0)
    dists_ = pdist(Z)
    sig = np.median(dists_)
    return sig

def median_bw_selector(SourceX, SourceY, X, Y, mode=1, num_ptsX=None, num_ptsY=None):
    """
        SourceX: function handle for generating X
        SourceY: function handle for generating X
        X,Y:     nxd arrays of observations
        mode=1: generate num_pts (X, Y) points from SourceX, SourceY
                for median_calculation
        mode=2: Choose the first n/2 points of X and Y for median
                calculation
    """
    if mode==1:
        # choose a default value of num_points if needed
        num_ptsX = 25 if num_ptsX is None else num_ptsX
        num_ptsY = 25 if num_ptsY is None else num_ptsY
        # generate a new set of observations for bandwidth selection
        X_, Y_ = SourceX(num_ptsX), SourceY(num_ptsY)
    elif mode==2:
        assert X is not None and Y is not None
        n, m = len(X), len(Y)
        # use the first half of the given data for bandwidth selection
        X_, Y_ = X[:n//2], Y[:m//2]
    else:
        raise Exception(f"mode must either be 1 or 2: input = {mode}")
    bw = get_median_bw(X=X_, Y=Y_)
    return bw

def median_bw_selector_2(X, Y):
    """
        X,Y:     nxd arrays of observations
    """
    n, m = len(X), len(Y)
    # use the first half of the given data for bandwidth selection
    X_, Y_ = X[:n//2], Y[:m//2]

    bw = get_median_bw(X=X_, Y=Y_)
    return bw

def get_bootstrap_threshold(X, Y, kernel_func, statfunc, alpha=0.05,
                            num_perms=500, progress_bar=False,
                            return_stats=False):
    """
        Return the level-alpha rejection threshold for the statistic
        computed by the function handle stat_func using num_perms
        permutations.
    """
    # Convert to tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)
    
    assert len(X.shape) == 2
    # concatenate the two samples
    Z = torch.vstack((X, Y))
    # assert len(X)==len(Y)
    n, n_plus_m = len(X), len(Z)
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z)

    # original_statistic = statfunc(X, Y, kernel_func)
    perm_statistics = torch.zeros(num_perms)

    range_ = tqdm(range(num_perms)) if progress_bar else range(num_perms)
    for i in range_:
        perm = torch.randperm(n_plus_m)
        X_, Y_ = Z[perm[:n]], Z[perm[n:]]
        stat = statfunc(X_, Y_, kernel_func)
        perm_statistics[i] = stat

    # obtain the threshold
    perm_statistics = torch.sort(perm_statistics).values

    i_ = int(num_perms*(1-alpha))
    threshold = perm_statistics[i_]
    if return_stats:
        return threshold, perm_statistics
    else:
        return threshold



def get_normal_threshold(alpha):
    return stats.norm.ppf(1-alpha)

def get_spectral_threshold(X, Y, kernel_func, alpha=0.05, numEigs=None,
                            numNullSamp=200):
    n = len(X)
    assert len(Y)==n

    if numEigs is None:
        numEigs = 2*n-2
    numEigs = min(2*n-2, numEigs)

    testStat = n*TwoSampleMMDSquared(X, Y, kernel_func, unbiased=False)

    #Draw samples from null distribution
    Z = np.vstack((X, Y))
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z) #

    H = np.eye(2*n) - 1/(2*n)*np.ones((2*n, 2*n))
    KZ_ = np.matmul(H, np.matmul(KZ, H))


    kEigs = np.linalg.eigvals(KZ_)[:numEigs]
    kEigs = 1/(2*n) * abs(kEigs);
    numEigs = len(kEigs);

    nullSampMMD = np.zeros((numNullSamp,))

    for i in range(numNullSamp):
        samp = 2* np.sum( kEigs * (np.random.randn(numEigs))**2)
        nullSampMMD[i] = samp

    nullSampMMD  = np.sort(nullSampMMD)
    threshold = nullSampMMD[round((1-alpha)*numNullSamp)]
    return threshold


def get_spectral_threshold_torch(X, Y, kernel_func, alpha=0.05, numEigs=None,
                            numNullSamp=200):
    n = len(X)
    assert len(Y)==n

    if numEigs is None:
        numEigs = 2*n-2
    numEigs = min(2*n-2, numEigs)

    testStat = n*TwoSampleMMDSquared(X, Y, kernel_func, unbiased=False)

    #Draw samples from null distribution
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Z = torch.vstack((X, Y))
    Z = Z.to(device)
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z) #

    H = torch.eye(2*n) - 1/(2*n)*torch.ones((2*n, 2*n))
    H = H.to(device)
    KZ_ = torch.mm(H, torch.mm(KZ, H))


    kEigs = torch.linalg.eigvals(KZ_)[:numEigs]
    kEigs = 1/(2*n) * abs(kEigs);
    numEigs = len(kEigs);

    nullSampMMD = torch.zeros((numNullSamp,))

    for i in range(numNullSamp):
        samp = 2* torch.sum( kEigs * (torch.randn((numEigs,), device=device))**2)
        nullSampMMD[i] = samp

    nullSampMMD, _ = torch.sort(nullSampMMD)
    threshold = nullSampMMD[round((1-alpha)*numNullSamp)]
    return threshold.item()


def get_unifrom_convergence_threshold(n, m, k_max=1.0, alpha=0.05, biased=False):
    assert 0<alpha<1
    if biased:
        #use Mcdiarmid's inequality based bound stated in Corollary 9 of
        # Gretton et al. (2012), JMLR
        threshold = sqrt(k_max/n + k_max/m)*(1+ sqrt(2*log(1/alpha)))
    else:
        # use Hoeffding's inequality based bound stated in Corollary 11 of
        # Gretton et al. (2012), JMLR
        threshold = (sqrt(2)*4*k_max/sqrt(m+n)) * sqrt(log(1/alpha))
    return threshold

###====================Misc uitls=======================
# normalize vector
def normalize_vector(vector):
    """
    Normalize a given vector to have a magnitude of 1.
    
    Parameters:
        vector (numpy.ndarray): Input vector to be normalized.
        
    Returns:
        numpy.ndarray: Normalized vector.
    """
    # norm = np.linalg.norm(vector)  # 벡터의 크기 계산
    # if norm == 0:
    #     return vector
    # return vector / norm
    std = np.std(vector)
    if std ==0:
        return vector
    return vector / std



def GaussianVector(mean, cov, n, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    rv = stats.multivariate_normal(mean, cov)
    X = rv.rvs(size=n)
    
    return X
def MultiExpVector(scale, n, seed=None):
    """
    Sample from a multivariate exponential distribution.

    Parameters:
    - scale: The scale parameter for the exponential distribution (can be a scalar or vector).
    - n: The number of samples to generate.
    - seed: Random seed for reproducibility (optional).
    
    Returns:
    - X: A numpy array of shape (n, len(scale)) containing the samples.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # If scale is a scalar, use it for all dimensions
    if np.ndim(scale) == 0:
        scale = np.full(n, scale)
    
    # Sample from the exponential distribution
    X = np.random.exponential(scale, size=(n, len(scale)))
    
    return X

def UniformVector(lower, upper, n, d, seed=None):
    """
    Simulate vectors from a uniform distribution.

    Parameters:
    lower (array-like): Lower bounds for the uniform distribution.
    upper (array-like): Upper bounds for the uniform distribution.
    n (int): Number of samples to generate.
    d (int): Dimension of each sample.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    ndarray: Array of shape (n, d) with uniformly distributed samples.
    """
    if seed is not None:
        np.random.seed(seed)

    lower = np.array(lower)
    upper = np.array(upper)

    # Ensure the lower and upper bounds are of the correct dimensions
    if lower.shape != (d,) or upper.shape != (d,):
        raise ValueError("Lower and upper bounds must be of shape (d,)")

    X = np.random.uniform(lower, upper, (n, d))

    return X


def TVector(loc, shape, df, n, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    return multivariate_t.rvs(loc = loc,shape = shape,df=df, size=n)

def DirichletVector(d, n, Alpha=None):
    if Alpha is None:
        Alpha = np.ones((d,))
    X = np.random.dirichlet(alpha=Alpha, size=n)
    # X = torch.from_numpy(X_).float()
    return X

# Get the resampled version of empirical variance using a function handle
def get_resampled_std(X, Y, stat_func, kernel_func=None, samples=200):
    if kernel_func is None:
        kernel_func = RBFkernel1

    nx, ny = len(X), len(Y)
    stat_vals = np.zeros((samples, ))
    for i in range(samples):
        idxX = np.random.randint(0, nx, (nx,))
        idxY = np.random.randint(0, ny, (ny,))
        X_ = X[idxX]
        Y_ = Y[idxY]
        stat_vals[i] = stat_func(X_, Y_, kernel_func)
    std = stat_vals.std()
    return std


# Get the resampled version of empirical variance using observation vector
def get_bootstrap_std(obs, num_bootstrap=200):
    vals = np.zeros((num_bootstrap,))
    N = len(obs)
    for i in range(num_bootstrap):
        idx = np.random.choice(a=N, size=(N,))
        # idx = torch.randint(low=0, high=N, size=(N,))
        vals[i] = (obs[idx]).mean()
    return vals.std()

###

class LinearRegressionLSE:
    def __init__(self):
        self.weights = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        # Move tensors to the appropriate device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Add a column of ones to X for the bias term
        X = torch.cat([torch.ones(X.shape[0], 1).to(self.device), X], dim=1)
        
        # Calculate weights using the normal equation
        X_T = torch.transpose(X, 0, 1)
        self.weights = torch.linalg.inv(X_T @ X) @ X_T @ y
        return self 
    def predict(self, X):
        # Move tensor to the appropriate device
        X = X.to(self.device)
        
        # Add a column of ones to X for the bias term
        X = torch.cat([torch.ones(X.shape[0], 1).to(self.device), X], dim=1)
        
        # Compute predictions: y = X * w
        return X @ self.weights
    

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _to_tensor(self, data):
        # Convert NumPy arrays to PyTorch tensors if necessary
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return data.to(self.device)
    
    def fit(self, X, y):
        self.X_train = self._to_tensor(X)
        self.y_train = self._to_tensor(y)
        return self 
    
    def predict(self, X):
        X = self._to_tensor(X)
        predictions = []

        

        # Calculate distances from X to all training points
        for x in X:
            distances = torch.sqrt(torch.sum((self.X_train - x) ** 2, dim=1))
            # Get the k nearest neighbors
            knn_indices = torch.topk(distances, self.k, largest=False).indices
            # Predict by taking the mean of the k-nearest neighbors' labels
            knn_labels = self.y_train[knn_indices]
            predictions.append(torch.mean(knn_labels))


        return torch.stack(predictions)


class KernelRegression:
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        self.X_train = None
        self.y_train = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _to_tensor(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return data.to(self.device)
    
    def gaussian_kernel(self, distances):
        return torch.exp(-0.5 * (distances / self.bandwidth) ** 2)

    def fit(self, X, y):
        self.X_train = self._to_tensor(X)
        self.y_train = self._to_tensor(y)
        
        if self.bandwidth is None:
            with torch.no_grad():
                dist_mat = torch.cdist(self.X_train, self.X_train)
                self.bandwidth = torch.median(dist_mat[dist_mat > 0])
                self.bandwidth = torch.max(self.bandwidth, torch.tensor(1e-5))
        return self 
    
    def predict(self, X):
        X = self._to_tensor(X)
        predictions = []
        
   

        with torch.no_grad():
            for x in X:
                distances = torch.sum((self.X_train - x) ** 2, dim=1)  # Use squared distances
                weights = self.gaussian_kernel(distances)
                
                sum_weights = torch.sum(weights)
                
                if sum_weights < 1e-10:
                    nearest_idx = torch.argmin(distances)
                    pred = self.y_train[nearest_idx]
                else:
                    normalized_weights = weights / sum_weights
                    pred = torch.sum(normalized_weights * self.y_train)
                
                predictions.append(pred)
        
        

        return torch.stack(predictions)
    
class KNNKernelEstimator:
    def __init__(self, k=3, bandwidth=1.0):
        self.k = k
        self.bandwidth = bandwidth
        self.V_train = None
        self.X_train = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def gaussian_kernel(self, X, X_prime):
        """ Gaussian kernel function k(X, X') """
        return torch.exp(-0.5 * (X - X_prime) ** 2 / self.bandwidth ** 2)

    def fit(self, V, X):
        """ Save training data V and X """
        self.V_train = V.to(self.device)
        self.X_train = X.to(self.device)
        return self
    def predict_function(self, V):
        """
        Predict a kernel function k(X_i, .) for each V_i
        Returns a list of functions k_hat(X, .) which estimates k(X_i, .)
        """
        V = V.to(self.device)
        predicted_functions = []

        # For each V_i in the input set
        for v in V:
            # Find k nearest neighbors based on V_i
            distances = torch.sqrt(torch.sum((self.V_train - v) ** 2, dim=1))
            knn_indices = distances.topk(self.k, largest=False).indices
            
            # Use corresponding X_train to create the kernel function for each neighbor
            knn_X = self.X_train[knn_indices]
            
            # Define a function that averages over the k nearest neighbors' kernels
            def estimated_kernel(X):
                kernels = [self.gaussian_kernel(x, X) for x in knn_X]
                return sum(kernels) / len(kernels)
            
            predicted_functions.append(estimated_kernel)
        
        return predicted_functions
    

def permute_within_bins(X, Y, V_binned, W_binned):
    """
    Perform permutation within each bin defined by V_binned and W_binned.
    """
    Z = torch.vstack((X, Y))
    n = len(X)
    
    permuted_X = torch.zeros_like(X)
    permuted_Y = torch.zeros_like(Y)
    
    # Loop over each unique bin
    unique_bins = np.unique(V_binned)
    for bin_value in unique_bins:
        # print(unique_bins)
        # Find indices in this bin for both X and Y
        X_bin_indices = np.where(V_binned == bin_value)[0]
        Y_bin_indices = np.where(W_binned == bin_value)[0] + n  # offset for Y
        
        bin_indices = np.concatenate([X_bin_indices, Y_bin_indices])
        print(bin_indices)
        print("Z size:", Z.size(0))
        # Permute within this bin
        perm = torch.randperm(len(bin_indices))
        permuted_bin = Z[bin_indices][perm]
        
        # Split back into permuted X and Y
        permuted_X[X_bin_indices] = permuted_bin[:len(X_bin_indices)]
        permuted_Y[Y_bin_indices - n] = permuted_bin[len(X_bin_indices):]
    
    return permuted_X, permuted_Y
