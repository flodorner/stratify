from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import joblib as jbl


def make_predictions_irt_mmlu_pro(index):
    y = jbl.load("Data/mmlu_pro.jbl")[index]
    f = np.load("Data/irt2p_mmlu_pro.npy")
    return f,y 

def make_predictions_mean_mmlu_pro(index):
    data = jbl.load("Data/mmlu_pro.jbl")
    y = data[index]
    f = data.mean(0)
    return f,y 


def make_predictions_full_mmlu_pro(index):
    data = jbl.load("Data/mmlu_pro.jbl")
    y = data[index]
    f = np.vstack((data[:index], data[index+1:])).T
    return f,y 


def make_predictions_diff_omnimath(index):
    data = np.load("Data/omnimath.npy")
    y = data[index]
    f = np.load("Data/diffs_omnimath.npy")
    return f,y 


def make_predictions_domain_omnimath(index):
    data = np.load("Data/omnimath.npy")
    y = data[index]
    f = np.load("Data/domains_omnimath.npy")
    return f,y 


def make_predictions_cross_omnimath(index):
    data = np.load("Data/omnimath.npy")
    y = data[index]
    
    diffs = np.load("Data/diffs_omnimath.npy")
    diffs = np.round(diffs)/10
    domains = np.load("Data/domains_omnimath.npy")
    
    f = np.array([domains[i]+str(diffs[i]) for i in range(len(y))])
    return f,y
    
    



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

def make_predictions_judge_unc(model):
    preds = get_all_predictions_mmlu("llama3.1")
    targets = get_all_targets_mmlu()
    gpt = get_all_predictions_mmlu(model)
    
    f = (preds).max(1)
    y = (targets*gpt).sum(1)
    return f,y 


def make_predictions_difficulty(model):
    targets = get_all_targets_mmlu()
    gpt = get_all_predictions_mmlu(model)
    
    f = np.load("Data/difficulty_llama70b.npy")
    y = (targets*gpt).sum(1)
    return f,y 

def make_predictions_subtasks(model):
    targets = get_all_targets_mmlu()
    gpt = get_all_predictions_mmlu(model)
    
    f = np.load("Data/mmlu_subtasks.npy")
    y = (targets*gpt).sum(1)
    return f,y 



def get_ppi_data(f,y,n=100,k=100,with_replacement=False):
    keys = np.random.random((k, len(f)))
    if not with_replacement:
        idx = np.argpartition(keys, kth=n-1, axis=1)     # O(k*N)
        sub_idx  = idx[:, :n]
        comp_idx = idx[:, n:]


        subs_f = np.take(f, sub_idx).T     # shape (k, n)
        comps_f = np.take(f, comp_idx).T

        subs_y = np.take(y, sub_idx).T     # shape (k, n)
    
    if with_replacement:
        # sample k groups of size n, independently, with replacement
        sub_idx = np.random.randint(0, len(f), size=(k, n))  # shape (k, n)

        subs_f = np.take(f, sub_idx).T                  # (n, k) -> transpose matches original pattern
        subs_y = np.take(y, sub_idx).T                  # (n, k)

        # "complement" isn't well-defined with replacement (duplicates, no strict complement).
        # Return the full dataset replicated per group for compatibility: shape (N, k).
        comps_f = np.broadcast_to(f, (k, len(f))).T          # (N, k)
    
    return subs_f,comps_f,subs_y,float(np.mean(y))


def get_base_data(f,y,n=100,k=100,with_replacement=False):
    keys = np.random.random((k, len(f)))
    if not with_replacement:
        idx = np.argpartition(keys, kth=n-1, axis=1)     # O(k*N)
        sub_idx  = idx[:, :n]
        subs_y = np.take(y, sub_idx).T     # shape (k, n)
    
    if with_replacement:
        # sample k groups of size n, independently, with replacement
        sub_idx = np.random.randint(0, len(f), size=(k, n))  # shape (k, n)
        subs_y = np.take(y, sub_idx).T                  # (n, k)
    
    return subs_y,float(np.mean(y))

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
    y_l ,target = get_base_data(f,y,n=n,k=k,with_replacement=with_replacement)
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
    sd_upper_bound = sd + 2*beta / np.sqrt(step)
    print(sd_upper_bound)
    return weight * (sd + 2*beta / np.sqrt(step)) / step 

def bound_capped(sd,step,weight,beta):
    sd_upper_bound = sd + 2*beta / np.sqrt(step)
    return weight * (np.minimum(sd_upper_bound,0.5)) / step 

def adaptive_strat_k(f,y,n=100,k=1024,with_replacement=False,beta="default",cap_bound=False,beta_const=4.5):    
    if beta == "default":
        beta = np.sqrt(beta_const*np.log(n))
    #Warmup?!
    #Idea: Every step does index-based stratum queries (How to parallelize?)
    #Then: Batched calculations of the exploration indexes 
    
    masks = [f == value for value in np.unique(f)]
    assert len(masks) * 2 < n, "n not large enough for warmup phase"
    sizes = [mask.sum() for mask in masks]
    #assert min([mask.sum() for mask in masks]) > n or with_replacement, str(n)+"  "+str(min([mask.sum() for mask in masks]))
    weights = [size / len(f)  for size in sizes]
    
    #For each arm: pre-sample draws without replacement
    #rs = [np.random.choice(y[masks[i]], size = (n,k) )  for i in range(len(masks)) ]
    if with_replacement:
        rs = [np.random.choice(y[masks[i]], size = (n,k) )  for i in range(len(masks)) ]
    else:
        rs = [shuffled_copies(y[masks[i]],k)  for i in range(len(masks)) ]
               
    # Start by sampling twice per arm, unless it only has a single element. 
    ns = np.zeros((len(masks),k),dtype=int) + 1 
    for i in range(len(masks)):
        if sizes[i] > 1 or with_replacement:
            ns[i] += 1 
    
    sums = [rs[i][:ns[i,0]].sum(0) for i in range(len(masks))] #All entries at i are copies here, just pick the first one
    
    idx = np.arange(k)
    
    for _ in range(max(n - ns[:,0].sum(), 0)):
        stds = [np.sqrt((sums[i] - (sums[i])**2 / ns[i]) / (ns[i]-1)) for i in range(len(masks))]
        if not cap_bound:
            bs = np.stack([bound(stds[i], ns[i], weights[i], beta)  for i in range(len(masks))]) # num_masks times k runs
        else:
            bs = np.stack([bound_capped(stds[i], ns[i], weights[i], beta)  for i in range(len(masks))])
        if not with_replacement: #Ensure we don't sample from any "exhausted" strata
            samples_left = np.expand_dims(np.array(sizes),1) > ns
            bs[~samples_left] = -np.inf
        
        best_b = np.argmax(bs,0) #One per parallel run!
        
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
    weights = np.array([mask.sum() / len(f)  for mask in masks])
    
    
    q = (n - len(weights)) * weights
    samples = 1 + np.floor(q).astype(int)  # Guarantee one sample per stratum
    samples[np.argsort(-(q - np.floor(q)))[: n - samples.sum()]] += 1 #Reassign rather than rounding down
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
    f = (f - np.min(f)) / (np.max(f)-np.min(f))  #Normalize...
    return np.clip(np.rint(f * (n_strata-1)) / (n_strata-1), 0.0, 1.0)



def tpr(x,y):
    return (x*y).sum() / y.sum()

def tnr(x,y):
    return ((1-x)*(1-y)).sum() / (1-y).sum()

def var_strat(f,y):
    f_cal = np.zeros_like(f,dtype=float)
    for value in np.unique(f):
        f_cal[f==value] = y[f==value].mean()
    return (f_cal-y).var()


def var_adaptive_strat(f,y):
    optimized_variance = 0.0
    for value in np.unique(f):
        prob = (f == value).mean()
        conditional_std = y[f == value].std()
        optimized_variance += prob*conditional_std 
    return optimized_variance**2
    


