import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cPickle
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from scipy.ndimage.filters import gaussian_filter
import pickle
import pandas as pd
import os
co = cPickle.load(open('co.pkl','rb'))
sigmas = [0,5,10]
correlations = pd.DataFrame(columns=['catpair1','catpair2','sigma','corr','pvalue'])
for key in co.keys():
    catpair1,catpair2 = key
    bnames = ['/data/aseewald/good/results/occur/{}_{}_{}.png'.format(catpair1,catpair2,sigma) for sigma in sigmas]
    if all([os.path.exists(bname) for bname in bnames]):
        continue
    for i,sigma in enumerate(sigmas):
        smooth = gaussian_filter(co[key],sigma)
        sns.heatmap(smooth)
        plt.savefig(bnames[i])
        plt.close() 
        for key2 in co.keys():
            smooth2 = gaussian_filter(co[key2],sigma)
            print(np.sum(smooth.flatten() * smooth2.flatten()))
            if np.sum(smooth.flatten() * smooth2.flatten()) < 30:
                print("Not considering {}:{} because not enough intersection")
            if correlations[(correlations['catpair1'] == key2) & (correlations['catpair2'] == key)].count > 0: continue
            corr,pvalue = pearsonr(smooth,smooth2)
            correlations.loc[len(correlations)] = [key,key2,sigma,corr,pvalue]
    correlations.to_pickle('paircorr.pkl')
