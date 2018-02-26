
# coding: utf-8

# In[ ]:


import numpy as np

def epsilonmax(x, eps):
    n = len(x)
    p = []
    for i in range(n):
        p.append(eps/n)
    p[np.argmax(x)] += 1 - eps
    return p

def softmax(x, scale = 1):
    x = np.array(x)/scale
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    p = e_x/e_x.sum()
    p = p/p.sum()
    return p

def softV(x, scale = 1):
    x = np.array(x)/scale
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    e_sum = e_x.sum()
    e_sum = scale * (np.log(e_sum) + max_x)
    return e_sum

def sparsetau(x):
    x = np.array(x)
    sorted_x = np.sort(x)[::-1]
    S = np.array([])
    for i in range(0,len(x)):
        if 1+(i+1)*sorted_x[i] >= (sorted_x[0:(i+1)]).sum():
            S = np.append(S,sorted_x[i])
    tau = (S.sum() - 1)/S.size
    return tau, S

def sparsedist(x, scale = 1):
    x = np.array(x/scale)
    tau, _ = sparsetau(x)
    p = x - tau
    p[p<0] = 0
    if p.sum() > 0.0:
        p = p/p.sum()
    else:
        p = np.ones_like(x)/x.shape[0]
    return p

def sparse_eps_dist(x, eps, scale = 1):
    x = np.array(x/scale)
    tau, _ = sparsetau(x)
    p = x - tau
    p[p<0] = 0
    if p.sum() > 0.0:
        p = p/p.sum()
    else:
        p = np.ones_like(x)/x.shape[0]
        
    n = len(x)
    for i in range(n):
        p[i] = eps/n + (1-eps)*p[i]
    return p


def sparsemax(x,scale = 1):
    x = np.array(x/scale)
    tau, S = sparsetau(x)
    spmax_x = 0.5*(S**2 - tau**2).sum() + 0.5
    spmax_x = scale*spmax_x
    return spmax_x

def Q_value(Exp, reward, dis, action_Q):
    if Exp == 'softmax':
        action_V = softV(action_Q)
        return reward + dis * action_V
    elif Exp == 'sparsemax' or Exp == 'sparse+eps':
        action_max = sparsemax(action_Q)
        return reward + dis * action_max
    else:
        print("Exploration Type is wrong.")
        return 0

def choice_action(Exp, eps, action_Q):
    if Exp == 'softmax':
        action_max = softmax(action_Q)
        return np.random.choice(len(action_max),size=1,p=action_max)[0]
    elif Exp == 'sparsemax':
        action_dist = sparsedist(action_Q)
        return np.random.choice(len(action_dist),size=1,p=action_dist)[0]
    elif Exp == 'sparse+eps':
        action_dist = sparse_eps_dist(action_Q, eps)
        return np.random.choice(len(action_dist),size=1,p=action_dist)[0]
    elif Exp == 'epsilon':
        action_max = epsilonmax(action_Q, eps)
        return np.random.choice(len(action_max),size=1,p=action_max)[0]
    else:
        print("Exploration Type is wrong.")
        return 0
    
        