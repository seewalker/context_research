import numpy as np
from scipy.stats import pearsonr
import random
M = np.load('metric.csv.npy')
N = M.shape[0]
nf = 4
simdim = N//nf
vals,pos,neg = [],[],[]
rs = range(N)
for ki in range(nf):
    for kj in range(nf):
        for i in range(simdim):
            for j in range(simdim):
                if i == j:
                    continue
                if i+ki*simdim < M.shape[0] and j+kj*simdim < M.shape[1]:
                    vals.append(M[i,j])
                    pos.append(M[i+ki*simdim,j+kj*simdim])
                    randi = random.choice(rs)
                    randj = random.choice(rs)
                    while randj == randi:
                        randj = random.choice(rs)
                    neg.append(M[randi,randj])

print("pos:",pearsonr(vals,pos))
print("neg:",pearsonr(vals,neg))
