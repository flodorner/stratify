from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import joblib as jbl

def get_full_benchmark_mmlu_pro():
    data = jbl.load("Data/mmlu_pro.jbl")
    return data


def get_all_targets_mmlu():
    return np.load("Data/targets_mmlu.npy")

def get_all_predictions_mmlu(model):
    if model=="llama3.1":
        model = "llama_405B_numeric"
       
    if not model == "llama3.1_rounded":
        return np.load("Data/"+model + "_predictions_mmlu.npy")
    else:
        pre_rounded = np.load("Data/llama_405B_numeric_predictions_mmlu.npy")
        idx = pre_rounded.argmax(axis=-1)  
        one_hot = np.eye(pre_rounded.shape[-1])[idx]
        return one_hot
    
def make_predictions_judge(model):
    preds = get_all_predictions_mmlu("llama3.1")
    targets = get_all_targets_mmlu()
    gpt = get_all_predictions_mmlu(model)
    
    f = (gpt*preds).sum(1)
    y = (targets*gpt).sum(1)
    return f,y 




def get_ppi_data(f,y,n=100,k=100,with_replacement=False):
    assert not with_replacement, "Not implemented"
    keys = np.random.random((k, len(f)))
    
    idx = np.argpartition(keys, kth=n-1, axis=1)     # O(k*N)
    sub_idx  = idx[:, :n]
    comp_idx = idx[:, n:]
    
    
    subs_f = np.take(f, sub_idx).T     # shape (k, n)
    comps_f = np.take(f, comp_idx).T
    
    subs_y = np.take(y, sub_idx).T     # shape (k, n)
    return subs_f,comps_f,subs_y,np.mean(y)

def subsamples_n_k(x, n, k):
    """Return (k, n) array: k independent subsamples of size n from 1D x, each without replacement."""
    x = np.asarray(x)
    m = x.size

    keys = np.random.random((k, m))
    idx = np.argpartition(keys, n - 1, axis=1)[:, :n]
    return x[idx].T

def shuffled_copies(a, k):
    a = np.asarray(a)
    n = a.shape[0]

    keys = np.random.random((n, k))          # independent random keys per (i, col)
    idx = np.argsort(keys, axis=0)     # permutation indices per column
    return a[idx]   






def baseline(f,y, n=100, k=1000,with_replacement=False):
    f_l, f_u, y_l ,target = get_ppi_data(f,y,n=n,k=k,with_replacement=with_replacement)
    sum_y_l  = (y_l).sum(axis=0)
    mean_y_l = sum_y_l / n
    est = mean_y_l
    return np.mean((est-target)**2)*n 

def ppi_k(f,y, n=100, k=1000,with_replacement=False):
    f_l, f_u, y_l ,target = get_ppi_data(f,y,n=n,k=k,with_replacement=with_replacement)
    sum_y_l  = (y_l).sum(axis=0)
    sum_f_l  = (f_l).sum(axis=0)
    sum_yf_l = (y_l * f_l).sum(axis=0)
    mean_y_l = sum_y_l / n
    mean_f_l = sum_f_l / n
    mean_f_u = f_u.mean()
    var_f_u = f_u.var()

    # Covariance on labeled set: matches np.cov(yl, fl)[0,1] default ddof=1
    cov_yf_l = (sum_yf_l - n * mean_y_l * mean_f_l) / (n - 1)
    denom = var_f_u
    lam = np.where(np.abs(denom) > 0.0, cov_yf_l / denom, 0.0)
    est = mean_y_l + lam * (mean_f_u - mean_f_l)
 
    return np.mean((est-target)**2)*n 

def bound(sd,step,weight,beta):
    return weight * (sd + 2*beta / np.sqrt(step)) / step 

def adaptive_strat_k(f,y,n=100,k=1024,with_replacement=False,beta="default"):    
    if beta == "default":
        beta = np.sqrt(4.5*np.log(n))
        
    #Warmup?!
    #Idea: Every step does index-based stratum queries (How to parallelize?)
    #Then: Batched calculations of the exploration indexes 
    
    masks = [f == value for value in np.unique(f)]
    #assert min([mask.sum() for mask in masks]) > n or with_replacement, str(n)+"  "+str(min([mask.sum() for mask in masks]))
    weights = [mask.sum() / len(f)  for mask in masks]
    
    #For each arm: pre-sample draws without replacement
    #rs = [np.random.choice(y[masks[i]], size = (n,k) )  for i in range(len(masks)) ]
    if with_replacement:
        rs = [np.random.choice(y[masks[i]], size = (n,k) )  for i in range(len(masks)) ]
    else:
        rs = [shuffled_copies(y[masks[i]],k)  for i in range(len(masks)) ]
        
    # Start by sampling twice per arm 
    ns = np.zeros((len(masks),k),dtype=int) + 2
    
    sums = [rs[i][:2].sum(0) for i in range(len(masks))]
    
    idx = np.arange(k)
    
    for _ in range(max(n - 2*len(masks), 0)):
        stds = [np.sqrt((sums[i] - (sums[i])**2 / ns[i]) / (ns[i]-1)) for i in range(len(masks))]
        bs = [bound(stds[i], ns[i], weights[i], beta)  for i in range(len(masks))]
        best_b = np.argmax(np.stack(bs),0) #One per parallel run!
        
        chooses = [best_b == i for i in range(len(masks))]
        
        for i in range(len(masks)):
            
            if chooses[i].any():
                index = idx[chooses[i]]
                sums[i][index] += rs[i][ns[i][index],index]
                ns[i][index] += 1
    
    means = [sums[i]/ns[i] for i in range(len(masks))]
    est = np.stack([weights[i]*means[i] for i in range(len(masks))]).sum(0)
    #print(sum([sum(ns[i]) for i in range(len(masks))]))
    return np.mean((est-np.mean(y))**2)*n 

def simple_strat_k(f,y,n=100,k=1024,with_replacement=False):
    masks = [f == value for value in np.unique(f)]
    weights = [mask.sum() / len(f)  for mask in masks]
    samples = [np.floor(n*weight) for weight in weights]
    assert np.all([masks[i].sum() >= samples[i] for i in range(len(masks))]) or with_replacement, str(n)+"  "+str(min([mask.sum() for mask in masks]))
    
    if with_replacement:
        rs = [np.random.choice(y[masks[i]], size = (n,k) )  for i in range(len(masks)) ]
    else:
        rs = [shuffled_copies(y[masks[i]],k)  for i in range(len(masks)) ]
    i=0
    
    means = [rs[i][:int(samples[i]),:].sum(0)/samples[i] for i in range(len(masks))]
    
    est = np.stack([weights[i]*means[i] for i in range(len(masks))]).sum(0)
    return np.mean((est-np.mean(y))**2)*n 

def round_strata(f,n_strata):
    return np.clip(np.rint(f * (n_strata-1)) / (n_strata-1), 0.0, 1.0)