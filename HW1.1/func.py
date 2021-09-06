import numpy as np
def normalize(x):
    xmean = np.mean(x)
    xstd = np.std(x, ddof=1)
    x = (x - xmean) / xstd
    return x

def model(x, p,model_type):
    if (model_type == "linear"): return p[0] * x + p[1]
    if (model_type == "logistic"): return p[0] + p[1] * (1.0 / (1.0 + np.exp(-(x - p[2]) / (p[3] + 0.00001))))


def loss(p,norx,nory,model_type):
    n = len(norx)
    y = model(norx,p,model_type)
    loss = (np.sum((y-nory)**2))/n
    return loss
  
    
