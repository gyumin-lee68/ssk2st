import numpy as np
import torch
from copy import deepcopy
from scipy.linalg import solve as scp_solve
from warnings import warn
import utils
from functools import partial
from utils import RBFkernel, RBFkernel1, KNNKernelEstimator
import kernels
from functools import lru_cache

def get_xzy_randn(n_points, ground_truth='H0', dim=2, device=None, **ignored):
    y = torch.randn(n_points, dim, device=device)
    y /= torch.norm(y, dim=1, keepdim=True)

    noise = 0.1 * torch.randn(n_points, dim, device=device) / np.sqrt(dim)
    z = y + noise
    x = y.clone()

    if ground_truth == 'H1':
        x[:, 0] += noise[:, 0]
        x[:, 1:] += 0.1 * torch.randn_like(x[:, 1:], device=device) / np.sqrt(dim)
    elif ground_truth == 'H0':
        x += 0.1 * torch.randn_like(x) / np.sqrt(dim)
    else:
        raise NotImplementedError(f'{ground_truth} has to be H0 or H1')

    return x, z, y


def add_diag(x, val):
    if len(x.shape) != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f'x is not a square matrix: shape {x.shape}')

    idx = range(x.shape[0])
    y = x.clone()
    y[idx, idx] += val
    return y


def compute_cme_error(K_YY, K_QQ, K_Yy, K_Qq, K_qq, reg):
    n = K_YY.shape[0]
    Kinv = torch.linalg.solve(add_diag(K_YY, n * reg), K_Yy).T

    cme_error = (K_qq.diagonal() + (Kinv @ K_QQ @ Kinv.T).diagonal() - 2 * (Kinv @ K_Qq).diagonal()).mean()

    return cme_error


def compute_single_k_fold_error(K_yy, K_QQ, reg, k=2):
    n = K_yy.shape[0]
    idx = torch.randperm(n, device=K_yy.device)

    k_fold_error = 0
    for i in range(k):
        idx_test = idx[i * (n // k):min(n, (i + 1) * (n // k))]
        idx_train = torch.tensor(np.setdiff1d(idx.cpu().numpy(), idx_test.cpu().numpy()), device=idx.device)
        k_fold_error += compute_cme_error(K_YY=K_yy[idx_train][:, idx_train], K_QQ=K_QQ[idx_train][:, idx_train],
                                          K_Yy=K_yy[idx_train][:, idx_test], K_Qq=K_QQ[idx_train][:, idx_test],
                                          K_qq=K_QQ[idx_test][:, idx_test], reg=reg)

    return k_fold_error / k


def compute_single_loo_error(K_yy, K_QQ, reg, cpu_solver=False, cpu_dtype=np.float64):
    n = K_yy.shape[0]
    if cpu_solver:
        A = cpu_dtype(add_diag(K_yy, n * reg).cpu().numpy())
        B = cpu_dtype(K_yy.cpu().numpy())
        Kinv = torch.tensor(scp_solve(A, B, assume_a='pos')).float().to(K_yy.device).T
    else:
        Kinv = torch.linalg.solve(add_diag(K_yy, n * reg), K_yy).T

    # without reg2, Kinv.T = Kinv
    return ((K_QQ.diagonal() + (Kinv @ K_QQ @ Kinv.T).diagonal() -
             2 * (Kinv @ K_QQ).diagonal()) / (1 - Kinv.diagonal()) ** 2).mean()


def compute_loo_errors(K_yy, K_QQ, lambda_values=None, verbose=False, cpu_solver=False, cpu_dtype=np.float64):
    # Discard values below svd tolerance. Not multiplied by matrix size since it's done in compute_single_loo_error
    svd_tol = torch.linalg.matrix_norm(K_yy, ord=2) * torch.finfo(K_yy.dtype).eps

    if lambda_values is None:
        lambda_values = torch.logspace(2 + torch.log10(svd_tol), 5 + torch.log10(svd_tol), 3)
        lambda_values_tol = lambda_values
    else:
        lambda_values = torch.tensor(lambda_values, device=K_yy.device)
        lambda_values_tol = lambda_values[lambda_values >= svd_tol]

    if len(lambda_values_tol) == 0:
        raise ValueError(f'All lambda values < svd tolerance:\n{lambda_values} < {svd_tol}')

    loos = torch.zeros_like(lambda_values_tol)

    for i, value in enumerate(lambda_values_tol):
        loos[i] = compute_single_loo_error(K_yy, K_QQ, value, cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)

    min_idx = torch.argmin(loos)

    lambda_vals = lambda_values_tol[min_idx]

    loos = loos.flatten()

    # if verbose:
    #     print(f'lambda values: {lambda_values}\nsvd tolerance: {svd_tol}\nlambda > tol {lambda_values_tol}\n'
    #           f'LOOs: {loos}\nBest loo/loo: {loos[min_idx]}/{lambda_vals}')

    return loos[min_idx], lambda_vals


def leave_one_out_regressors_single_kernel(y, K_zz, kernel_y, lambda_values=None, param_dict=None, default_y_args=None,
                                           verbose=True, cpu_solver=False, cpu_dtype=np.float64):

    if param_dict is None:
        # if verbose:
        #     print('No parameters to test for LOO found.'
        #           ' LOO will be done with the passed/default ridge regression parameters')
        #     print(f'Kernel: {kernel_y}, default parameters: {default_y_args}')
        K_yy = eval(f'kernels.{kernel_y}_kernel(y, **default_y_args)')
        best_loo_error, best_loo_lambda = compute_loo_errors(K_yy, K_zz, lambda_values, verbose,
                                                             cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)
        kernel_y_args = deepcopy(default_y_args)
    else:
        params_mesh = torch.meshgrid(*param_dict.values(), indexing='ij')
        param_names = param_dict.keys()
        loo_errors = torch.zeros_like(params_mesh[0].flatten())

        loo_lambda = torch.zeros_like(loo_errors)

        # if verbose:
        #     print(f'Kernel: {kernel_y}, default parameters: {default_y_args}')

        for i in range(len(loo_errors)):
            kernel_y_args = deepcopy(default_y_args)
            for key_idx, key in enumerate(param_names):
                kernel_y_args[key] = params_mesh[key_idx].flatten()[i]

            K_yy = eval(f'kernels.{kernel_y}_kernel(y, **kernel_y_args)')
            loo_errors[i], best_loo_lambdas = compute_loo_errors(K_yy, K_zz, lambda_values, verbose,
                                                                 cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)
            loo_lambda[i] = best_loo_lambdas

        min_idx = torch.argmin(loo_errors)

        kernel_y_args = deepcopy(default_y_args)
        for key_idx, key in enumerate(param_names):
            kernel_y_args[key] = params_mesh[key_idx].flatten()[min_idx]

        best_loo_error = loo_errors[min_idx]
        best_loo_lambda = loo_lambda[min_idx]

    # if verbose:
    #     print(f'Best LOO: {best_loo_error}, Best parameters: lambda={best_loo_lambda} and {kernel_y_args}')

    return best_loo_lambda, kernel_y_args, best_loo_error
def leave_one_out_regressors(y, K_zz, kernel_y, lambda_values=None, param_dict=None, default_y_args=None, verbose=True,
                             cpu_solver=False, cpu_dtype=np.float64):
    if isinstance(kernel_y, list):
        best_loo_error = -1

        for kernel_y_name in kernel_y:
            loo_lambda, found_y_args, loo_error = leave_one_out_regressors_single_kernel(y, K_zz, kernel_y_name,
                                                                                         lambda_values,
                                                                                         param_dict[kernel_y_name],
                                                                                         default_y_args, verbose,
                                                                                         cpu_solver=cpu_solver,
                                                                                         cpu_dtype=cpu_dtype)

            if best_loo_error == -1 or best_loo_error > loo_error:
                best_loo_lambda = loo_lambda
                kernel_y_args = [kernel_y_name, found_y_args]
                best_loo_error = loo_error
        return best_loo_lambda, kernel_y_args, best_loo_error

    else:
        return leave_one_out_regressors_single_kernel(y, K_zz, kernel_y, lambda_values, param_dict, default_y_args,
                                                      verbose, cpu_solver=cpu_solver, cpu_dtype=np.float64)

def get_yz_regressors(y, z, kernel_y, kernel_z, kernel_y_args, kernel_z_args, param_dict=None, lambda_values=None,
                      verbose=True, cpu_solver=False, cpu_dtype=np.float64):
    n_points = y.shape[0]
    K_zz = eval(f'kernels.{kernel_z}_kernel(z, **kernel_z_args)')

    # if verbose:
        # print('Estimating regressions parameters with LOO')

    ridge_lambda, kernel_y_args, best_loo_error = leave_one_out_regressors(y, K_zz, kernel_y, lambda_values, param_dict,
                                                                           kernel_y_args, verbose,
                                                                           cpu_solver=cpu_solver, cpu_dtype=cpu_dtype)

    if isinstance(kernel_y, list):
        K_yy = eval(f'kernels.{kernel_y_args[0]}_kernel(y, **kernel_y_args[1])')
    else:
        K_yy = eval(f'kernels.{kernel_y}_kernel(y, **kernel_y_args)')

    # if verbose:
        # print('All gram matrices computed')

    K_yy = add_diag(K_yy, K_yy.shape[0] * ridge_lambda)
    K_zz = torch.cat((torch.eye(n_points, device=K_yy.device), K_zz), 1)

    if cpu_solver:
        A = cpu_dtype(K_yy.cpu().numpy())
        B = cpu_dtype(K_zz.cpu().numpy())
        W_all = torch.tensor(scp_solve(A, B, assume_a='pos')).float().to(K_yy.device)
    else:
        W_all = torch.linalg.solve(K_yy, K_zz)

    K_yy_inv = W_all[:, :n_points]  # (K_yy + lambda n I)^(-1)
    K_yy_inv_K_zz = W_all[:, n_points:]  # (K_yy + lambda n I)^(-1) K_zz

    # if verbose:
    #     print('W_all computed')

    return K_yy_inv, K_yy_inv_K_zz, kernel_y_args, kernel_z_args

def find_regressors(x, z, y_z, y_x=None, param_dict_yz=None, lambda_values_yz=None, param_dict_yx=None,
                        lambda_values_yx=None, verbose=False, cpu_solver=False, cpu_dtype=np.float64, **ignored):
        if y_x is None:
            y_x = y_z

        x_holdout = x.detach().clone()
        y_x_holdout = y_x.detach().clone()
        z_holdout = z.detach().clone()
        y_z_holdout = y_z.detach().clone()

        K_yy_inv_z, _, kernel_yz_args = \
            cme.get_yz_regressors(y_z, z, kernel_yz, kernel_z, kernel_yz_args, kernel_z_args,
                                  param_dict_yz, lambda_values_yz, verbose, cpu_solver, cpu_dtype)

        self.K_yy_inv_Phi_z = K_yy_inv_z @ self.z_holdout

        if isinstance(self.kernel_yz, list):
            self.kernel_yz = self.kernel_yz_args[0]
            self.kernel_yz_args = self.kernel_yz_args[1]

        K_yy_inv_x, _, self.kernel_yx_args = \
            cme.get_yz_regressors(y_x, x, self.kernel_yx, self.kernel_x, self.kernel_yx_args, self.kernel_x_args,
                                  param_dict_yx, lambda_values_yx, verbose, cpu_solver, cpu_dtype)
        self.K_yy_inv_Phi_x = K_yy_inv_x @ self.x_holdout

        if isinstance(self.kernel_yx, list):
            self.kernel_yx = self.kernel_yx_args[0]
            self.kernel_yx_args = self.kernel_yx_args[1]

        self._regression_done = True
        
def cme(X, V, X_train, V_train, kernel='gaussian'):
    K_vv_inv, K_vv_inv_K_xx, _, __ = get_yz_regressors(V_train,X_train,kernel,kernel,{'sigma2': V_train.std().item() ** 2},{'sigma2': X_train.std().item() ** 2})
    kernel_functions = {
        'gaussian': kernels.gaussian_kernel,
        # Add other kernels here if needed
    }
    
    kernel_function = kernel_functions.get(kernel)
    cme_X_V = kernel_function(V, V_train, __) @ K_vv_inv @ kernel_function(X_train, X, _)
    
    return cme_X_V

def fcheck(X, X_base, Y_base, V_base, V_plus, W_base, W_plus, kernel_func, kernel='gaussian'):
    # X, Y: 데이터 배열 (shape: (n_samples, n_features))
    n1,_ = X_base.shape
    n2,_ = Y_base.shape
    m1,_ = V_plus.shape
    m2,_ = W_plus.shape
    # print(X_base.shape)
    # print(Y_base.shape)
    
    
    # 첫 번째 항 계산: 1/n1 * sum(k(X_i, .)) for i in I_XV2
    sum_X = np.sum(kernel_func(X_base,X).cpu().numpy(), axis=0)
    term1 = (1 / n1) * sum_X
    # print(sum_X.shape)
    # 두 번째 항 계산: 1/n2 * sum(k(Y_i, .)) for i in I_YW2
    sum_Y = np.sum(kernel_func(Y_base,X).cpu().numpy(), axis=0)
    term2 = (1 / n2) * sum_Y

    # 최종 결과
    result_1 = term1 - term2
    result_1 = torch.tensor(result_1)
    
    X_base_1 = X_base[:n1//2]
    X_base_2 = X_base[n1//2:]
    Y_base_1 = Y_base[:n2//2]
    Y_base_2 = Y_base[n2//2:]
    
    V_base_1 = V_base[:n1//2]
    V_base_2 = V_base[n1//2:]
    W_base_1 = W_base[:n2//2]
    W_base_2 = W_base[:n2//2]
    
    cme_X_lab_1 = np.sum(cme(X, V_base_1, X_base_2, V_base_2, kernel).cpu().numpy(), axis=0)
    cme_X_lab_2 = np.sum(cme(X, V_base_2, X_base_1, V_base_1, kernel).cpu().numpy(), axis=0)
    term3 = (1/n1) * (cme_X_lab_1+cme_X_lab_2)
    cme_Y_lab_1 = np.sum(cme(X, W_base_1, Y_base_2, W_base_2, kernel).cpu().numpy(), axis=0)
    cme_Y_lab_2 = np.sum(cme(X, W_base_2, Y_base_1, W_base_1, kernel).cpu().numpy(), axis=0)
    term4 = (1/n2) * (cme_Y_lab_1+cme_Y_lab_2)
    
    result_2 = term3 - term4
    result_2 = torch.tensor(result_2)
    
    V_plus_1 = V_plus[:m1//2]
    V_plus_2 = V_plus[m1//2:]
    W_plus_1 = W_plus[:m2//2]
    W_plus_2 = W_plus[m2//2:]
    V_full_1 = torch.cat((V_plus_1,V_base_1),dim=0)
    V_full_2 = torch.cat((V_plus_2,V_base_2),dim=0)
    W_full_1 = torch.cat((W_plus_1,W_base_1),dim=0)
    W_full_2 = torch.cat((W_plus_2,W_base_2),dim=0)
    
    cme_X_full_1 = np.sum(cme(X, V_full_1, X_base_2, V_base_2, kernel).cpu().numpy(), axis=0)
    cme_X_full_2 = np.sum(cme(X, V_full_2, X_base_1, V_base_1, kernel).cpu().numpy(), axis=0)
    term5 = (1/(n1+m1)) * (cme_X_full_1+cme_X_full_2)
    cme_Y_full_1 = np.sum(cme(X, W_full_1, Y_base_2, W_base_2, kernel).cpu().numpy(), axis=0)
    cme_Y_full_2 = np.sum(cme(X, W_full_2, Y_base_1, W_base_1, kernel).cpu().numpy(), axis=0)
    term6 = (1/(n2+m2)) * (cme_Y_full_1+cme_Y_full_2)
    result_3 = term5 - term6
    result_3 = torch.tensor(result_3)
    
    result = result_1 - result_2 + result_3
    return result

def fcheck_knn2(X, X_base, Y_base, V_base, V_plus, W_base, W_plus, kernel_func, kernel='gaussian'):
    # X, Y: 데이터 배열 (shape: (n_samples, n_features))
    n1,_ = X_base.shape
    n2,_ = Y_base.shape
    m1,_ = V_plus.shape
    m2,_ = W_plus.shape
    # print(X_base.shape)
    # print(Y_base.shape)
    
    # 첫 번째 항 계산: 1/n1 * sum(k(X_i, .)) for i in I_XV2
    sum_X = np.sum(kernel_func(X_base,X).cpu().numpy(), axis=0)
    term1 = (1 / n1) * sum_X
    # print(sum_X.shape)
    # 두 번째 항 계산: 1/n2 * sum(k(Y_i, .)) for i in I_YW2
    sum_Y = np.sum(kernel_func(Y_base,X).cpu().numpy(), axis=0)
    term2 = (1 / n2) * sum_Y

    # 최종 결과
    result_1 = term1 - term2
    result_1 = torch.tensor(result_1)
    
    X_base_1 = X_base[:n1//2]
    X_base_2 = X_base[n1//2:]
    Y_base_1 = Y_base[:n2//2]
    Y_base_2 = Y_base[n2//2:]
    
    V_base_1 = V_base[:n1//2]
    V_base_2 = V_base[n1//2:]
    W_base_1 = W_base[:n2//2]
    W_base_2 = W_base[:n2//2]
    
    # modelknnx1 = KNNKernelEstimator(k=3)
    # modelknnx1.fit(V_base_2, X_base_2)
    # # print(X_base_2.shape[0])
    # # print(V_base_1.shape[0])
    # # print(V_base_2.shape[0])
    
    # modelknnx2 = KNNKernelEstimator(k=3)
    # modelknnx2.fit(V_base_1, X_base_1)
    # modelknny1 = KNNKernelEstimator(k=3)
    # modelknny1.fit(W_base_2, Y_base_2)
    # modelknny2 = KNNKernelEstimator(k=3)
    # modelknny2.fit(W_base_2, Y_base_2)
    
    # cond_kerx1 = modelknnx1.predict_function(V_base_1)
    # cond_kerx2 = modelknnx2.predict_function(V_base_2)
    # cond_kery1 = modelknny1.predict_function(W_base_1)
    # cond_kery2 = modelknny2.predict_function(W_base_2)
    
    # knn_X_lab_1 = np.sum(torch.tensor([[kernel(x) for x in X] for kernel in cond_kerx1]).cpu().numpy(), axis=0)
    # knn_X_lab_2 = np.sum(torch.tensor([[kernel(x) for x in X] for kernel in cond_kerx2]).cpu().numpy(), axis=0)
    # knn_Y_lab_1 = np.sum(torch.tensor([[kernel(x) for x in X] for kernel in cond_kery1]).cpu().numpy(), axis=0)
    # knn_Y_lab_2 = np.sum(torch.tensor([[kernel(x) for x in X] for kernel in cond_kery2]).cpu().numpy(), axis=0)

    # term3 = (1/n1) * (knn_X_lab_1+knn_X_lab_2)
    # term4 = (1/n2) * (knn_Y_lab_1+knn_Y_lab_2)
    
    # result_2 = term3 - term4
    # result_2 = torch.tensor(result_2)
    
    # V_plus_1 = V_plus[:m1//2]
    # V_plus_2 = V_plus[m1//2:]
    # W_plus_1 = W_plus[:m2//2]
    # W_plus_2 = W_plus[m2//2:]
    # V_full_1 = torch.cat((V_plus_1,V_base_1),dim=0)
    # V_full_2 = torch.cat((V_plus_2,V_base_2),dim=0)
    # W_full_1 = torch.cat((W_plus_1,W_base_1),dim=0)
    # W_full_2 = torch.cat((W_plus_2,W_base_2),dim=0)
    
    # cond_ker_full_x1 = modelknnx1.predict_function(V_full_1)
    # cond_ker_full_x2 = modelknnx2.predict_function(V_full_2)
    # cond_ker_full_y1 = modelknny1.predict_function(W_base_1)
    # cond_ker_full_y2 = modelknny2.predict_function(W_full_1)
    
    # knn_X_full_1 = np.sum(torch.tensor([[kernel(x) for x in X] for kernel in cond_ker_full_x1]).cpu().numpy(), axis=0)
    # knn_X_full_2 = np.sum(torch.tensor([[kernel(x) for x in X] for kernel in cond_ker_full_x2]).cpu().numpy(), axis=0)
    # knn_Y_full_1 = np.sum(torch.tensor([[kernel(x) for x in X] for kernel in cond_ker_full_y1]).cpu().numpy(), axis=0)
    # knn_Y_full_2 = np.sum(torch.tensor([[kernel(x) for x in X] for kernel in cond_ker_full_y2]).cpu().numpy(), axis=0)
    
    # term5 = (1/(n1+m1)) * (knn_X_full_1+knn_X_full_2)
    # term6 = (1/(n2+m2)) * (knn_Y_full_1+knn_Y_full_1)
    # result_3 = term5 - term6
    # result_3 = torch.tensor(result_3)
    
    # result = result_1 - result_2 + result_3
    # return result
    k = 3  # 원래 k값 유지
    models = {
        'x1': KNNKernelEstimator(k=k).fit(V_base_2, X_base_2),
        'x2': KNNKernelEstimator(k=k).fit(V_base_1, X_base_1),
        'y1': KNNKernelEstimator(k=k).fit(W_base_2, Y_base_2),
        'y2': KNNKernelEstimator(k=k).fit(W_base_2, Y_base_2)
    }
    
    # 4. 조건부 커널 예측 (배치 처리)
    batch_size = 256
    
    def batch_predict_functions(model, data):
        functions = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            functions.extend(model.predict_function(batch))
        return functions
    
    cond_kernels = {
        'x1': batch_predict_functions(models['x1'], V_base_1),
        'x2': batch_predict_functions(models['x2'], V_base_2),
        'y1': batch_predict_functions(models['y1'], W_base_1),
        'y2': batch_predict_functions(models['y2'], W_base_2)
    }
    
    # 5. 커널 적용 (배치 처리)
    def apply_kernels(kernels, X):
        results = []
        for kernel_fn in kernels:
            # kernel_fn은 estimated_kernel 함수
            batch_results = torch.tensor([kernel_fn(x.to(kernel_fn.__closure__[0].cell_contents.device)) for x in X])
            results.append(batch_results)
        return torch.stack(results).sum(dim=0)
    
    knn_results = {
        'X1': apply_kernels(cond_kernels['x1'], X),
        'X2': apply_kernels(cond_kernels['x2'], X),
        'Y1': apply_kernels(cond_kernels['y1'], X),
        'Y2': apply_kernels(cond_kernels['y2'], X)
    }
    
    result_2 = (1/n1) * (knn_results['X1'] + knn_results['X2']) - \
               (1/n2) * (knn_results['Y1'] + knn_results['Y2'])
    
    # 6. 전체 데이터에 대한 예측
    V_full_1 = torch.cat((V_plus[:m1//2], V_base_1))
    V_full_2 = torch.cat((V_plus[m1//2:], V_base_2))
    W_full_1 = torch.cat((W_plus[:m2//2], W_base_1))
    W_full_2 = torch.cat((W_plus[m2//2:], W_base_2))
    
    
    full_kernels = {
        'x1': batch_predict_functions(models['x1'], V_full_1),
        'x2': batch_predict_functions(models['x2'], V_full_2),
        'y1': batch_predict_functions(models['y1'], W_base_1),
        'y2': batch_predict_functions(models['y2'], W_full_1)
    }
    
    full_results = {
        'X1': apply_kernels(full_kernels['x1'], X),
        'X2': apply_kernels(full_kernels['x2'], X),
        'Y1': apply_kernels(full_kernels['y1'], X),
        'Y2': apply_kernels(full_kernels['y2'], X)
    }
    
    result_3 = (1/(n1+m1)) * (full_results['X1'] + full_results['X2']) - \
               (1/(n2+m2)) * (full_results['Y1'] + full_results['Y2'])
    
    # 중간 결과를 즉시 삭제하여 메모리 확보
    del models['x1']
    del models['x2']
    torch.cuda.empty_cache()  # GPU 메모리 정리
    
    return result_1 - result_2 + result_3

def fcheck_knn(X, X_base, Y_base, V_base, V_plus, W_base, W_plus, kernel_func, kernel='gaussian'):
    
    # 기본 계산 (변경 없음)
    n1, _ = X_base.shape
    n2, _ = Y_base.shape
    m1, _ = V_plus.shape
    m2, _ = W_plus.shape
    
    # 첫 번째 항 계산: 1/n1 * sum(k(X_i, .)) for i in I_XV2
    sum_X = np.sum(kernel_func(X_base,X).cpu().numpy(), axis=0)
    term1 = (1 / n1) * sum_X
    # 두 번째 항 계산: 1/n2 * sum(k(Y_i, .)) for i in I_YW2
    sum_Y = np.sum(kernel_func(Y_base,X).cpu().numpy(), axis=0)
    term2 = (1 / n2) * sum_Y

    # 최종 결과
    result_1 = term1 - term2
    result_1 = torch.tensor(result_1)
    
    X_base_1 = X_base[:n1//2]
    X_base_2 = X_base[n1//2:]
    Y_base_1 = Y_base[:n2//2]
    Y_base_2 = Y_base[n2//2:]
    
    V_base_1 = V_base[:n1//2]
    V_base_2 = V_base[n1//2:]
    W_base_1 = W_base[:n2//2]
    W_base_2 = W_base[:n2//2]
    
    k = 3  # k값 설정
    models = {
        'x1': KNNKernelEstimator(k=k).fit(V_base_2, X_base_2),
        'x2': KNNKernelEstimator(k=k).fit(V_base_1, X_base_1),
        'y1': KNNKernelEstimator(k=k).fit(W_base_2, Y_base_2),
        'y2': KNNKernelEstimator(k=k).fit(W_base_2, Y_base_2)
    }
    
    # 배치 처리
    batch_size = 512
    
    def batch_predict_functions(model, data):
        functions = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            functions.extend(model.predict_function(batch))
        return functions
    
    cond_kernels = {
        'x1': batch_predict_functions(models['x1'], V_base_1),
        'x2': batch_predict_functions(models['x2'], V_base_2),
        'y1': batch_predict_functions(models['y1'], W_base_1),
        'y2': batch_predict_functions(models['y2'], W_base_2)
    }
    
    def apply_kernels(kernels, X):
        results = []
        for kernel_fn in kernels:
            # kernel_fn은 estimated_kernel 함수
            batch_results = torch.stack([kernel_fn(x.to(kernel_fn.__closure__[0].cell_contents.device)) for x in X])
            results.append(batch_results)
        return torch.stack(results).sum(dim=0)
    
    knn_results = {
        'X1': apply_kernels(cond_kernels['x1'], X),
        'X2': apply_kernels(cond_kernels['x2'], X),
        'Y1': apply_kernels(cond_kernels['y1'], X),
        'Y2': apply_kernels(cond_kernels['y2'], X)
    }
    
    result_2 = (1/n1) * (knn_results['X1'] + knn_results['X2']) - \
               (1/n2) * (knn_results['Y1'] + knn_results['Y2'])
    
    # 전체 데이터에 대한 예측
    V_full_1 = torch.cat((V_plus[:m1//2], V_base_1))
    V_full_2 = torch.cat((V_plus[m1//2:], V_base_2))
    W_full_1 = torch.cat((W_plus[:m2//2], W_base_1))
    W_full_2 = torch.cat((W_plus[m2//2:], W_base_2))
    
    full_kernels = {
        'x1': batch_predict_functions(models['x1'], V_full_1),
        'x2': batch_predict_functions(models['x2'], V_full_2),
        'y1': batch_predict_functions(models['y1'], W_base_1),
        'y2': batch_predict_functions(models['y2'], W_full_1)
    }
    
    full_results = {
        'X1': apply_kernels(full_kernels['x1'], X),
        'X2': apply_kernels(full_kernels['x2'], X),
        'Y1': apply_kernels(full_kernels['y1'], X),
        'Y2': apply_kernels(full_kernels['y2'], X)
    }
    
    result_3 = (1/(n1+m1)) * (full_results['X1'] + full_results['X2']) - \
               (1/(n2+m2)) * (full_results['Y1'] + full_results['Y2'])
    
    return result_1 - result_2 + result_3

# Monte Carlo CME estimation
def montecarlo(X, V, X_train, V_train, kernel, num_samples=10000):
    """
    Estimate conditional mean embedding E[k(X, \cdot) | V] using Monte Carlo.

    Args:
        X (np.ndarray): Target data (e.g., test samples).
        V (np.ndarray): Conditional variables (e.g., test conditions).
        X_train (np.ndarray): Training data for X.
        V_train (np.ndarray): Training data for V.
        kernel (callable): Kernel function k(x, y).
        num_samples (int): Number of Monte Carlo samples.

    Returns:
        np.ndarray: Estimated conditional mean embedding.
    """
    kernel_functions = {
        'gaussian': kernels.gaussian_kernel,
        # Add other kernels here if needed
    }
    
    kernel_function = kernel_functions.get(kernel)
    cme_estimates = []
    device = X.device if hasattr(X, 'device') else 'cpu'
    for v in V:
        # Monte Carlo Sampling: Filter X_train conditioned on V_train close to v
        indices = torch.randperm(len(V_train))[:num_samples]  # Sample random indices
        sampled_X = X_train[indices]  # Corresponding X samples

        # Kernel Mean: Compute k(sampled_X, X) and take the mean
        kernel_values = kernel_function(sampled_X, X)  # Shape: [num_samples, len(X)]
        cme = kernel_values.mean(dim=0)  # Mean over sampled points

        cme_estimates.append(cme)
        

    return torch.stack(cme_estimates).to(device)  # Shape: [len(V), len(X)]

def fcheck_mc(X, X_base, Y_base, V_base, V_plus, W_base, W_plus, kernel_func, kernel='gaussian'):
    # X, Y: 데이터 배열 (shape: (n_samples, n_features))
    n1,_ = X_base.shape
    n2,_ = Y_base.shape
    m1,_ = V_plus.shape
    m2,_ = W_plus.shape
    # print(X_base.shape)
    # print(Y_base.shape)
    
    
    # 첫 번째 항 계산: 1/n1 * sum(k(X_i, .)) for i in I_XV2
    sum_X = np.sum(kernel_func(X_base,X).cpu().numpy(), axis=0)
    term1 = (1 / n1) * sum_X
    # print(sum_X.shape)
    # 두 번째 항 계산: 1/n2 * sum(k(Y_i, .)) for i in I_YW2
    sum_Y = np.sum(kernel_func(Y_base,X).cpu().numpy(), axis=0)
    term2 = (1 / n2) * sum_Y

    # 최종 결과
    result_1 = term1 - term2
    result_1 = torch.tensor(result_1)
    
    X_base_1 = X_base[:n1//2]
    X_base_2 = X_base[n1//2:]
    Y_base_1 = Y_base[:n2//2]
    Y_base_2 = Y_base[n2//2:]
    
    V_base_1 = V_base[:n1//2]
    V_base_2 = V_base[n1//2:]
    W_base_1 = W_base[:n2//2]
    W_base_2 = W_base[:n2//2]
    
    cme_X_lab_1 = np.sum(montecarlo(X, V_base_1, X_base_2, V_base_2, kernel).cpu().numpy(), axis=0)
    cme_X_lab_2 = np.sum(montecarlo(X, V_base_2, X_base_1, V_base_1, kernel).cpu().numpy(), axis=0)
    term3 = (1/n1) * (cme_X_lab_1+cme_X_lab_2)
    cme_Y_lab_1 = np.sum(montecarlo(X, W_base_1, Y_base_2, W_base_2, kernel).cpu().numpy(), axis=0)
    cme_Y_lab_2 = np.sum(montecarlo(X, W_base_2, Y_base_1, W_base_1, kernel).cpu().numpy(), axis=0)
    term4 = (1/n2) * (cme_Y_lab_1+cme_Y_lab_2)
    
    result_2 = term3 - term4
    result_2 = torch.tensor(result_2)
    
    V_plus_1 = V_plus[:m1//2]
    V_plus_2 = V_plus[m1//2:]
    W_plus_1 = W_plus[:m2//2]
    W_plus_2 = W_plus[m2//2:]
    V_full_1 = torch.cat((V_plus_1,V_base_1),dim=0)
    V_full_2 = torch.cat((V_plus_2,V_base_2),dim=0)
    W_full_1 = torch.cat((W_plus_1,W_base_1),dim=0)
    W_full_2 = torch.cat((W_plus_2,W_base_2),dim=0)
    
    cme_X_full_1 = np.sum(montecarlo(X, V_full_1, X_base_2, V_base_2, kernel).cpu().numpy(), axis=0)
    cme_X_full_2 = np.sum(montecarlo(X, V_full_2, X_base_1, V_base_1, kernel).cpu().numpy(), axis=0)
    term5 = (1/(n1+m1)) * (cme_X_full_1+cme_X_full_2)
    cme_Y_full_1 = np.sum(montecarlo(X, W_full_1, Y_base_2, W_base_2, kernel).cpu().numpy(), axis=0)
    cme_Y_full_2 = np.sum(montecarlo(X, W_full_2, Y_base_1, W_base_1, kernel).cpu().numpy(), axis=0)
    term6 = (1/(n2+m2)) * (cme_Y_full_1+cme_Y_full_2)
    result_3 = term5 - term6
    result_3 = torch.tensor(result_3)
    
    result = result_1 - result_2 + result_3
    return result
