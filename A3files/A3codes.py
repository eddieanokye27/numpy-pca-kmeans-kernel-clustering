import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from A3helpers import augmentX, gaussKernel, plotModel, generateData, plotPoints


# Q1
#a)
def minMulDev(X, Y):
    n, d = X.shape
    k = Y.shape[1]

    def objective(w_flat):
        Wmat = w_flat.reshape(d, k)

        scr = X @ Wmat
        loss_val = np.mean(logsumexp(scr, axis=1) - np.sum(Y * scr, axis=1))
        return loss_val

    initW = np.zeros(d * k)

    opt_result = minimize(objective, initW, method='BFGS')

    return opt_result.x.reshape(d, k)

#b)
def classify(Xtest, W):
    n = Xtest.shape
    k = W.shape

    test_scores = Xtest @ W
    pred_idx = np.argmax(test_scores, axis=1)

    m, k = test_scores.shape
    Yhat = np.zeros((m, k))
    Yhat[np.arange(m), pred_idx] = 1
    return Yhat

#c)
def calculateAcc(Yhat, Y):
    pred_lbl = np.argmax(Yhat, axis=1)
    true_lbl = np.argmax(Y, axis=1)
    return np.mean(pred_lbl == true_lbl)


#Q2

#a)
from scipy.linalg import eigh

def PCA(X, k):
    X = X.astype(float)
    n, d = X.shape
    
    mean_vec = np.mean(X, axis=0)
    
    X_centered = X - mean_vec
    
    cov_mat = X_centered.T @ X_centered
    
    eigvals, eigvecs = eigh(cov_mat)
    
    U = eigvecs[:, -k:].T
    
    return U, mean_vec

#b)
def projPCA(Xtest, mu, U): 
    X_centered_test = Xtest - mu
    return X_centered_test @ U.T  
   
#c)
def kernelPCA(X, k, kernel_func):
    X = X.astype(float)
    n = X.shape[0]
    
    Kmat = kernel_func(X, X).astype(float)
    
    cent_mat = np.ones((n, n)) / n
    K_centered = Kmat - cent_mat @ Kmat - Kmat @ cent_mat + cent_mat @ Kmat @ cent_mat

    eigvals, eigvecs = eigh(K_centered)
    
    A = eigvecs[:, -k:].T
    
    return A, K_centered

#d)
def projKernelPCA(Xtest, Xtrain, kernel_func, A):
    Xtest = Xtest.astype(float)
    Xtrain = Xtrain.astype(float)
    
    m = Xtest.shape[0]
    n = Xtrain.shape[0]
    
    K_test_train = kernel_func(Xtest, Xtrain).astype(float)
    K_train_train = kernel_func(Xtrain, Xtrain).astype(float)
    
    cent_mt = np.ones((m, n)) / n
    cent_nn = np.ones((n, n)) / n
    
    K_centered_test = (
        K_test_train
        - cent_mt @ K_train_train
        - K_test_train @ cent_nn
        + cent_mt @ K_train_train @ cent_nn
    )
    
    return K_centered_test @ A.T

#e)
def synClsExperimentsPCA():
    n_runs = 100
    n_train = 128
    n_test = 1000
    dim_list = [1, 2]
    gen_model_list = [1, 2]

    train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])

    np.random.seed(38)

    for run_idx in range(n_runs):
        for dim_i, k in enumerate(dim_list):
            for model_j, gen_model in enumerate(gen_model_list):

                Xtrain, Ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, Ytest = generateData(n=n_test, gen_model=gen_model)

                U, mu = PCA(Xtrain, k)

                Xtrain_emb = projPCA(Xtrain, mu, U)
                Xtest_emb = projPCA(Xtest, mu, U)

                Xtrain_emb = augmentX(Xtrain_emb)
                Xtest_emb = augmentX(Xtest_emb)

                W = minMulDev(Xtrain_emb, Ytrain)

                Yhat_train = classify(Xtrain_emb, W)
                Yhat_test = classify(Xtest_emb, W)

                train_acc[dim_i, model_j, run_idx] = calculateAcc(Yhat_train, Ytrain)
                test_acc[dim_i, model_j, run_idx] = calculateAcc(Yhat_test, Ytest)

    train_acc_avg = np.mean(train_acc, axis=2)
    test_acc_avg = np.mean(test_acc, axis=2)

    return train_acc_avg, test_acc_avg

#q3
#a
def kmeans(X, k, max_iter=1000):
    n, d = X.shape
    assert max_iter > 0 and k < n
    randIndx = np.random.choice(n, k, replace=False)
    U = X[randIndx].copy()
    for _ in range(max_iter):
        D = cdist(X, U, metric="sqeuclidean")
        cluster_idx = np.argmin(D, axis=1)
        Y = np.zeros((n, k), dtype=int)
        Y[np.arange(n), cluster_idx] = 1
        old_U = U.copy()
        Y_pinv = np.linalg.pinv(Y)
        U = Y_pinv @ X
        if np.allclose(old_U, U):
            break
    obj_val = (0.5 / n) * np.sum(D.min(axis=1))
    return Y, U, obj_val


#b
def repeatKmeans(X, k, n_runs=100):
    best_obj = float("inf")
    Y_best, U_best = None, None
    for _ in range(n_runs):
        Y, U, obj = kmeans(X, k)
        if obj < best_obj:
            best_obj = obj
            Y_best = Y
            U_best = U
    return Y_best, U_best, best_obj


#c
def chooseK(X, k_candidates=[2, 3, 4, 5, 6, 7, 8, 9]):
    obj_vals = []
    for k in k_candidates:
        _, _, obj = repeatKmeans(X, k)
        obj_vals.append(obj)
    return obj_vals


#d
def kernelKmeans(X, kernel_func, k, init_Y, max_iter=1000):
    n, d = X.shape
    assert max_iter > 0 and k < n
    K = kernel_func(X, X)
    Y = init_Y.copy()
    for _ in range(max_iter):
        old_Y = Y.copy()
        Y_pinv = np.linalg.pinv(Y)
        M = Y_pinv @ K @ Y_pinv.T
        diag_M = np.diag(M)
        term1 = np.diag(K).reshape(-1, 1)
        term2 = diag_M.reshape(1, -1)
        term3 = 2 * (K @ Y_pinv.T)
        D = term1 + term2 - term3
        idx = np.argmin(D, axis=1)
        Y = np.zeros((n, k), dtype=int)
        Y[np.arange(n), idx] = 1
        if np.allclose(Y, old_Y):
            break
    obj_val = (0.5 / n) * np.sum(np.min(D, axis=1))
    return Y, obj_val
