import torch
import numpy as np
from scipy.spatial.distance import cdist

def gaussian_kernel(x, y=None, bandwidth=1.0, amplitude=1.0):
    """
    Gaussian (RBF) kernel function
    k(x, y) = amplitude * exp(-||x - y||^2 / (2 * bandwidth^2))
    
    Args:
        x: Input tensor/array
        y: Input tensor/array (if None, y = x)
        bandwidth: Bandwidth parameter
        amplitude: Amplitude parameter
    
    Returns:
        Kernel matrix
    """
    y = x if y is None else y
    
    # Convert to torch tensors if they aren't already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Calculate pairwise squared Euclidean distances
    dists = torch.cdist(x, y)
    squared_dists = dists ** 2
    
    # Compute kernel
    k = amplitude * torch.exp(-squared_dists / (2 * bandwidth * bandwidth))
    
    return k

def linear_kernel(x, y=None, bias=0.0):
    """
    Linear kernel function
    k(x, y) = x^T y + bias
    
    Args:
        x: Input tensor/array
        y: Input tensor/array (if None, y = x)
        bias: Bias term
    
    Returns:
        Kernel matrix
    """
    y = x if y is None else y
    
    # Convert to torch tensors if they aren't already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Compute linear kernel
    k = torch.mm(x, y.T) + bias
    
    return k

def polynomial_kernel(x, y=None, degree=2, bias=1.0, coef0=1.0):
    """
    Polynomial kernel function
    k(x, y) = (coef0 * x^T y + bias)^degree
    
    Args:
        x: Input tensor/array
        y: Input tensor/array (if None, y = x)
        degree: Polynomial degree
        bias: Bias term
        coef0: Coefficient
    
    Returns:
        Kernel matrix
    """
    y = x if y is None else y
    
    # Convert to torch tensors if they aren't already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Compute polynomial kernel
    k = torch.pow(coef0 * torch.mm(x, y.T) + bias, degree)
    
    return k

def laplacian_kernel(x, y=None, bandwidth=1.0, amplitude=1.0):
    """
    Laplacian kernel function
    k(x, y) = amplitude * exp(-||x - y|| / bandwidth)
    
    Args:
        x: Input tensor/array
        y: Input tensor/array (if None, y = x)
        bandwidth: Bandwidth parameter
        amplitude: Amplitude parameter
    
    Returns:
        Kernel matrix
    """
    y = x if y is None else y
    
    # Convert to torch tensors if they aren't already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Calculate pairwise L1 distances
    dists = torch.cdist(x, y, p=1)
    
    # Compute kernel
    k = amplitude * torch.exp(-dists / bandwidth)
    
    return k

def rbf_kernel(x, y=None, bandwidth=1.0, amplitude=1.0):
    """
    RBF kernel function (alias for gaussian_kernel)
    k(x, y) = amplitude * exp(-||x - y||^2 / (2 * bandwidth^2))
    
    Args:
        x: Input tensor/array
        y: Input tensor/array (if None, y = x)
        bandwidth: Bandwidth parameter
        amplitude: Amplitude parameter
    
    Returns:
        Kernel matrix
    """
    return gaussian_kernel(x, y, bandwidth, amplitude)

def constant_kernel(x, y=None, constant=1.0):
    """
    Constant kernel function
    k(x, y) = constant
    
    Args:
        x: Input tensor/array
        y: Input tensor/array (if None, y = x)
        constant: Constant value
    
    Returns:
        Kernel matrix
    """
    y = x if y is None else y
    
    # Convert to torch tensors if they aren't already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Create constant kernel matrix
    n_x = x.shape[0]
    n_y = y.shape[0]
    k = torch.full((n_x, n_y), constant, dtype=torch.float32)
    
    return k 