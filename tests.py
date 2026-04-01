import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns
from tqdm import tqdm
from utils import LinearRegressionLSE, KNN, KernelRegression, KNNKernelEstimator
from conditional import fcheck, fcheck_knn, fcheck_mc

def fhat(X, X_base, Y_base, kernel_func):
    # X, Y (shape: (n_samples, n_features))
    n1,_ = X_base.shape
    n2,_ = Y_base.shape
    
    sum_X = np.sum(kernel_func(X_base,X).cpu().numpy(), axis=0)
    term1 = (1 / n1) * sum_X
    
    sum_Y = np.sum(kernel_func(Y_base,X).cpu().numpy(), axis=0)
    term2 = (1 / n2) * sum_Y

    result = term1 - term2
    result = torch.tensor(result)
    
    return result


def to_tensor(data, device):
    if isinstance(data, torch.Tensor):  
        return data.clone().detach().to(device)  
    else:  
        return torch.tensor(data, dtype=torch.float, device=device)
    
def crossMMD2sampleUnpaired(X, Y, kernel_func):
    """
        Compute the studentized cross-MMD statistic
        Details in Section 2 of https://arxiv.org/pdf/2211.14908.pdf
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = to_tensor(X, device), to_tensor(Y, device)
    
    n, d = X.shape
    m, d_ = Y.shape
    # sanity check
    assert (d_==d) and (n>=2) and (m>=2)
    # split the dataset into two equal parts
    n1, m1 = n//2, m//2
    X1, X2 = X[:n1], X[n1:]
    Y1, Y2 = Y[:m1], Y[m1:]
    # comptue the gram matrices
    Kxx = kernel_func(X1, X2)
    Kyy = kernel_func(Y1, Y2)
    Kxy = kernel_func(X1, Y2)
    Kyx = kernel_func(Y1, X2)
    
    # compute the numerator of the statistic
    Ux = Kxx.mean() - Kxy.mean()
    Uy = Kyx.mean() - Kyy.mean()
    U = Ux - Uy
    
    fhatx = fhat(X1,X2,Y2,kernel_func)
    fhaty = fhat(Y1,X2,Y2,kernel_func)
    U2 = fhatx.mean() - fhaty.mean()
    
    # compute the denominator
    term1 = (Kxx.mean(dim=1) - Kxy.mean(dim=1) - Ux)**2
    sigX2 = term1.mean()
    term2 = (Kyx.mean(dim=1) - Kyy.mean(dim=1) - Uy)**2
    sigY2 = term2.mean()
    sig = torch.sqrt(sigX2/n1 + sigY2/m1)
    if not sig>0:
        print(f'term1={term1}, term2={term2}, sigX2={sigX2}, sigY2={sigY2}')
        raise Exception(f'The denominator is {sig}')
    # obtain the statistic
    T = U/sig
    return T

def crossSSMMD2sample(X,V,Y,W,kernel_func, method):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, V, Y, W = to_tensor(X, device), to_tensor(V, device), to_tensor(Y, device), to_tensor(W, device)

    n1, d = X.shape
    m1, d_v = V.shape
    n2, d_ = Y.shape
    m2, d_w = W.shape
    # sanity check
    assert (d_==d) and (n1>=2) and (n2>=2) and (m1>=n1) and (m2>=n2)

    # print(X.shape)
    # print(V.shape)
    
    m1, m2 = m1-n1, m2-n2
    n1, n2 = n1//2, n2//2

    X_label = X
    V_unlabel = V[2*n1:]
    V_label = V[:2*n1]

    Y_label = Y
    W_unlabel = W[2*n2:]
    W_label = W[:2*n2]

    # split the data again for training and testing
    n1_half = n1 // 2
    n2_half = n2 // 2
    m1_half = m1 // 2
    m2_half = m2 // 2
    
    V_label_1 = V_label[:n1_half]
    V_label_2 = V_label[n1_half:n1]
    V2 = V_label[n1:]

    X_label_1 = X_label[:n1_half]
    X_label_2 = X_label[n1_half:n1]
    V_unlabel_1 = V_unlabel[:m1_half]
    V_unlabel_2 = V_unlabel[m1_half:]

    W_label_1 = W_label[:n2_half]
    W_label_2 = W_label[n2_half:n2]
    W2 = W_label[n2:]

    Y_label_1 = Y_label[:n2_half]
    Y_label_2 = Y_label[n2_half:n2]
    W_unlabel_1 = W_unlabel[:m2_half]
    W_unlabel_2 = W_unlabel[m2_half:]

    # split the dataset into two equal parts
    X1, X2 = X_label[:n1], X_label[n1:]
    Y1, Y2 = Y_label[:n2], Y_label[n2:]

    datx1 = V_label_1  # Remove the cat operation since we're only using one tensor
    datx2 = V_label_2
    daty1 = W_label_1
    daty2 = W_label_2
    
    fhatx1 = fhat(X_label_1,X2,Y2,kernel_func)
    fhatx2 = fhat(X_label_2,X2,Y2,kernel_func)
    fhatx = torch.cat((fhatx1, fhatx2)).to(device)
    
    fhaty1 = fhat(Y_label_1,X2,Y2,kernel_func)
    fhaty2 = fhat(Y_label_2,X2,Y2,kernel_func)
    fhaty = torch.cat((fhaty1, fhaty2)).to(device)
    
    if method == "linearRegression":
        modelcondx1 = LinearRegressionLSE()
        modelcondx1.fit(datx1, fhatx1)
        modelcondx2 = LinearRegressionLSE()
        modelcondx2.fit(datx2, fhatx2)
        modelcondy1 = LinearRegressionLSE()
        modelcondy1.fit(daty1, fhaty1)
        modelcondy2 = LinearRegressionLSE()
        modelcondy2.fit(daty2, fhaty2)
        
    elif method == "KNN":
        modelcondx1 = KNN()
        modelcondx1.fit(datx1, fhatx1)
        modelcondx2 = KNN()
        modelcondx2.fit(datx2, fhatx2)
        modelcondy1 = KNN()
        modelcondy1.fit(daty1, fhaty1)
        modelcondy2 = KNN()
        modelcondy2.fit(daty2, fhaty2)
        
    elif method == "KernelRegression":
        modelcondx1 = KernelRegression()
        modelcondx1.fit(datx1, fhatx1)
        modelcondx2 = KernelRegression()
        modelcondx2.fit(datx2, fhatx2)
        modelcondy1 = KernelRegression()
        modelcondy1.fit(daty1, fhaty1)
        modelcondy2 = KernelRegression()
        modelcondy2.fit(daty2, fhaty2)
    
    elif method == "DecisionTree":
        modelcondx1 = DecisionTreeRegressor()
        modelcondx2 = DecisionTreeRegressor()
        modelcondy1 = DecisionTreeRegressor()
        modelcondy2 = DecisionTreeRegressor()
        modelcondx1.fit(V_label_1.cpu().numpy(), fhatx1.cpu().numpy())
        modelcondx2.fit(V_label_2.cpu().numpy(), fhatx2.cpu().numpy())
        modelcondy1.fit(W_label_1.cpu().numpy(), fhaty1.cpu().numpy())
        modelcondy2.fit(W_label_2.cpu().numpy(), fhaty2.cpu().numpy())
        
    elif method == "RandomForest":
        modelcondx1 = RandomForestRegressor()
        modelcondx2 = RandomForestRegressor()
        modelcondy1 = RandomForestRegressor()
        modelcondy2 = RandomForestRegressor()
        modelcondx1.fit(V_label_1.cpu().numpy(), fhatx1.cpu().numpy())
        modelcondx2.fit(V_label_2.cpu().numpy(), fhatx2.cpu().numpy())
        modelcondy1.fit(W_label_1.cpu().numpy(), fhaty1.cpu().numpy())
        modelcondy2.fit(W_label_2.cpu().numpy(), fhaty2.cpu().numpy())
    
    cond_fhatx1 = to_tensor(modelcondx2.predict(V_label_1.cpu().numpy()).reshape(-1), device=device).squeeze()
    cond_fhatx2 = to_tensor(modelcondx1.predict(V_label_2.cpu().numpy()), device=device).squeeze()
    
    
    cond_fhatx3 = to_tensor(modelcondx2.predict(V_unlabel_1.cpu().numpy()), device=device).squeeze()
    cond_fhatx4 = to_tensor(modelcondx1.predict(V_unlabel_2.cpu().numpy()), device=device).squeeze()
    
    cond_fhatx_label = torch.cat([cond_fhatx1, cond_fhatx2],dim=0).to(device)
    cond_fhatx_full = torch.cat([cond_fhatx1, cond_fhatx2, cond_fhatx3, cond_fhatx4],dim=0).to(device)

    cond_fhaty1 = to_tensor(modelcondy2.predict(W_label_1.cpu().numpy()), device=device).squeeze()
    cond_fhaty2 = to_tensor(modelcondy1.predict(W_label_2.cpu().numpy()), device=device).squeeze()

    cond_fhaty3 = to_tensor(modelcondy2.predict(W_unlabel_1.cpu().numpy()), device=device).squeeze()
    cond_fhaty4 = to_tensor(modelcondy1.predict(W_unlabel_2.cpu().numpy()), device=device).squeeze()
    cond_fhaty_label = torch.cat([cond_fhaty1, cond_fhaty2],dim=0).to(device)
    cond_fhaty_full = torch.cat([cond_fhaty1, cond_fhaty2, cond_fhaty3, cond_fhaty4],dim=0).to(device)

    # compute the gram matrices
    Kxx = kernel_func(X1, X2)
    Kyy = kernel_func(Y1, Y2)
    Kxy = kernel_func(X1, Y2)
    Kyx = kernel_func(Y1, X2)
    
    # compute the numerator of the statistic
    U = fhatx.mean() - fhaty.mean()
    U_cond1 = cond_fhatx_label.mean() - cond_fhaty_label.mean()
    U_cond2 = cond_fhatx_full.mean() - cond_fhaty_full.mean()
    U_fin = U - U_cond1 + U_cond2
    
    # compute the denominator
    termX1 = (fhatx - cond_fhatx_label)**2 if cond_fhatx_label.numel() > 0 else fhatx**2
    sigX1 = termX1.mean()
    
    termX2 = (cond_fhatx_full-cond_fhatx_full.mean())**2 if cond_fhatx_full.numel() > 0 else torch.tensor(0.0, device=device)
    sigX2 = termX2.mean() if cond_fhatx_full.numel() > 0 else torch.tensor(0.0, device=device)
    
    termY1 = (fhaty - cond_fhaty_label)**2 if cond_fhaty_label.numel() > 0 else fhaty**2
    sigY1 = termY1.mean()
    
    termY2 = (cond_fhaty_full-cond_fhaty_full.mean())**2 if cond_fhaty_full.numel() > 0 else torch.tensor(0.0, device=device)
    sigY2 = termY2.mean() if cond_fhaty_full.numel() > 0 else torch.tensor(0.0, device=device)
    
    sig_ss = torch.sqrt(sigX1 / n1 + sigX2 / (n1 + m1) + sigY1 / n2 + sigY2 / (n2 + m2))
    if sig_ss <= 0:
        print(f'termX1={termX1}, termX2={termX2}, termY1={termY1}, termY2={termY2}, sigX1={sigX1}, sigX2={sigX2}, sigY1={sigY1}, sigY2={sigY2}')
        raise Exception(f'The denominator is {sig_ss}')
    
    # obtain the statistic
    T_ss = U_fin/sig_ss
    return T_ss

def safe_crossSSMMD2sample(X, V, Y, W, kernel_func, method):
    try:
        return crossSSMMD2sample(X, V, Y, W, kernel_func, method)
    except torch._C._LinAlgError as e:
        print(f"Error in crossSSMMD2sample: {e}")
        return None

def TwoSampleMMDSquared(X, Y, kernel_func, unbiased=False,
                        return_float=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = to_tensor(X, device), to_tensor(Y, device)

    Kxx = kernel_func(X, X)
    Kyy = kernel_func(Y, Y)
    Kxy = kernel_func(X, Y)

    n, m = len(X), len(Y)

    term1 = Kxx.sum()
    term2 = Kyy.sum()
    term3 = 2*Kxy.mean()

    if unbiased:
        term1 -= torch.trace(Kxx)
        term2 -= torch.trace(Kyy)
        MMD_squared = (term1/(n*(n-1)) + term2/(m*(m-1)) - term3)
    else:
        MMD_squared = term1/(n*n) + term2/(m*m) - term3
    if return_float:
        return MMD_squared.item
    else:
        return MMD_squared
