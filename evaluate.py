'''
Alexander Seewald 2016
aseewald@indiana.edu

pyamg is a dependency with the code as written, or you can remove eigen_solver='pyamg' to remove the dependency.

This program has an argparse-style command line interface. Running this program with the '--help' flag will describe the way to use it.
'''
import os
import time
import subprocess
import re
import sys
import pickle
import gzip
import itertools
import memcache
import time
import random
import multiprocessing as mp
import pandas as pd
from skimage.feature import ORB
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from skimage.io import imread,imsave
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from scipy.stats import entropy,pearsonr
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from collections import OrderedDict
from tqdm import tqdm
from functools import *
import objectGraph
from dataproc import *
import constants
import hyperparams as hp
import mywebsite

parser = argparse.ArgumentParser()
parser.add_argument('cluster_amounts')
parser.add_argument('trial',type=int)
parser.add_argument('dataset')
parser.add_argument('--multiproc',action='store_true',default=False)
parser.add_argument('--nthreads',default=5,type=int)
parser.add_argument('--take_prop',default=(1/3),type=float)
subparsers = parser.add_subparsers(dest='action',help=" ")
arch_parser = subparsers.add_parser('arch')
greedy_parser = subparsers.add_parser('greedy')
greedy_parser.add_argument('nickname')
greedy_parser.add_argument('--num_candidates',default=3000)
args = parser.parse_args()

#if args.action == "arch":
    

__author__ = "Alex Seewald"

def purity(labels, clusters, num_clusters, nearest_distances,quantiles):
    '''
    labels: array of string ground truth.
    clusters: array of integer predictions by clustering process.
    num_clusters: self-explanatory.
    nearest_distances: distancess of points from their cluster.
    Returns a tuple of:
        1) Net purity.
        2) Dictionary of purity of (class,takeProportion) pairs.
        3) Dictionary of purity of (cluster,takeProportion) pairs.
        4) Array of whether each object candidate has been correctly clustered.
    '''
    if np.all(nearest_distances == 0): #this value indicates nearest_distances data not available.
        quantiles = [1.0]
    net_purity,which = {quantile : 0 for quantile in quantiles}, {}
    by_class_purities = {quantile : {label : 0 for label in np.unique(labels)} for quantile in quantiles}
    by_class_counts = {quantile : {label : 0 for label in np.unique(labels)} for quantile in quantiles}
    by_cluster_purities = {quantile : {} for quantile in quantiles}
    # These dictionaries are just lookup tables for the str->int and int-> str functions.
    intlabel = {c : i for i,c in enumerate(np.unique(labels))}
    strlabel = {i : c for i,c in enumerate(np.unique(labels))}
    intlabels = np.array([intlabel[label] for label in labels])
    iscorrect = {quantile : np.zeros(labels.size) for quantile in quantiles}
    for quantile in quantiles:
        which[quantile] = (nearest_distances <= np.percentile(nearest_distances,100 * quantile))
        for clusterid in range(num_clusters):
            mask = (clusters == clusterid) & which[quantile]
            matching_labels = labels[mask]
            matching_intlabels = intlabels[mask]
            frequencies = np.bincount(matching_intlabels)
            if matching_labels.size > 0:
                by_cluster_purities[quantile][clusterid] = np.max(frequencies) / matching_labels.size
            if frequencies.size > 0:
                net_purity[quantile] += np.max(frequencies)
                maximum_cat = strlabel[np.argmax(frequencies)]
                for index in np.where(clusters == clusterid)[0]:
                    if labels[index] == maximum_cat:
                        iscorrect[quantile][index] = 1
                by_class_purities[quantile][maximum_cat] += np.max(frequencies)
                by_class_counts[quantile][maximum_cat] += matching_labels.size
        net_purity[quantile] = net_purity[quantile] / np.sum(which[quantile])
    for quantile in quantiles:
        for label in np.unique(labels):
            if by_class_counts[quantile][label] == 0:
                del(by_class_purities[quantile][label]) #avoiding things that don't exist to avoid divide by zero answers.
    by_class_purities = {quantile : {label : by_class_purities[quantile][label] / by_class_counts[quantile][label] for label in by_class_purities[quantile].keys()} for quantile in by_class_purities.keys()}
    return(net_purity,by_class_purities,by_cluster_purities,iscorrect,which)

def embed_data_greedy(hyperparams,csvname,quantile,nickname,K,X,row_stats,predictions,byclass,bycluster,iscorrect,splitid,which,perfect,even,num_fields,root='/home/aseewald/public_html'):
    '''
    Writes CSV ready for my webpage of the form:
    Each row is an object candidate.
    ismax is true if a object candidate is of the most common class in its cluster.
    TSNE_x (x position on screen) ,TSNE_y (y position on screen),clusterID (color),purityofcluster(for textual display),ismax(for textual display)ground_truth(for textual display),URL(static URL )

    which expresses which are in the current quantile.
    rows expresses which rows and included as samples at the specified number of samples..
    '''
    reglambda = readsql(f"SELECT reg_lambda FROM greedy_nicknames WHERE nickname = '{nickname}'",hyperparams)['reg_lambda']
    webdirs = ["/home/aseewald/public_html/data/embed/","/home/aseewald/public_html/static/embed/"]
    for webdir in webdirs:
        if not os.path.exists(webdir): subprocess.call(["mkdir","-p",webdir])
    # d3 needs this header.
    csvname = '/home/aseewald/public_html/'+csvname
    if not os.path.exists(csvname):
        csvout = open(csvname,'w')
        csvout.write("nickname,quantile,tsnex,tsney,cluster,clust_purity,cat_purity,correctness,label,url,description,numobj,perfect,even,num_fields\n")
    else:
        csvout = open(csvname,'a')
    kern_model = TSNE(n_components=2,random_state=0)
    feat_model = TSNE(n_components=2,random_state=0)
    current_X = X[which[quantile]]
    current_K = K[which[quantile]]
    kern_tn = kern_model.fit_transform(current_X)
    feat_tn = feat_model.fit_transform(current_K)
    count = 0
    for i,label in enumerate(row_stats['gt']):
        if not which[quantile][i]:
            continue
        numobj=row_stats['numobj'].iloc[i]['num'].values[0]
        rowcanid=row_stats['rowids'].iloc[i]
        ktnx,ktny = kern_tn[count]
        ftnx,ftny = feat_tn[count]
        # I need to extract something from rowimgname.
        if perfect:
            inurl = rowcanid[1]
            static_fs = os.path.join(root,"static/embed",os.path.split(rowcanid[1])[1])
            static_url = "http://madthunder.soic.indiana.edu/~aseewald/static/embed/" +os.path.split(rowcanid[1])[1]
        else:
            inurl = hyperparams.root(f"val_candidateimgs/{rowimgname}_0_objectness_{rowcanid}.jpg")
            static_fs = f"/home/aseewald/public_html/static/embed/{rowimgname}_{rowcanid}.png"
            static_url = f"http://madthunder.soic.indiana.edu/~aseewald/static/embed/{rowimgname}_{rowcanid}.png"
        if not os.path.exists(static_fs):
            subprocess.call(["cp",inurl,static_fs])
        cluster = predictions[i]
        try:
            stmt = f"{nickname},{quantile},{ktnx},{ktny},{ftnx},{ftny},{cluster},{np.mean(bycluster[cluster])},{byclass[label]},{iscorrect[i]},{label},{static_url},{numobj},{perfect},{even},{num_fields}\n"
            print("stmt=",stmt)
            csvout.write(stmt)
        except:
            print("couldn't write")
            continue
        count += 1
        return True

def embed_data_arch(csvname,hyperparams,X,row_imgnames,row_categories,row_canids,predictions,byclass,bycluster,iscorrect,nickname,num_clusters,num_samples,perfect,even):
    '''
    Writes CSV ready for my webpage of the form:
    Each row is an object candidate.
    ismax is true if a object candidate is of the most common class in its cluster.
    TSNE_x (x position on screen) ,TSNE_y (y position on screen),clusterID (color),purityofcluster(for textual display),ismax(for textual display)ground_truth(for textual display),URL(static URL )
    '''
    webdirs = ["/home/aseewald/public_html/data/embed/","/home/aseewald/public_html/static/embed/"]
    for webdir in webdirs:
        if not os.path.exists(webdir): subprocess.call(["mkdir","-p",webdir])
    if os.path.exists(csvname):
        csvout = open(csvname,'a')
    else:
        csvout = open(csvname,'w')
        csvout.write("tsnex,tsney,cluster,clust_purity,cat_purity,correctness,label,url,num_clusters,num_samples,perfect,even\n")
    model = TSNE(n_components=2,random_state=0)
    tn = model.fit_transform(X)
    for i,gt in enumerate(row_categories):
        tnx,tny = tn[i]
        if perfect:
            origdir,name = os.path.split(row_canids.ix[i])
            static_fs = f"/home/aseewald/public_html/static/embed/{name}"
            static_url = f"http://madthunder.soic.indiana.edu/~aseewald/static/embed/{name}"
            if not os.path.exists(static_fs):
                subprocess.call(["cp",row_canids.ix[i],static_fs])
        else:
            static_fs = f"/home/aseewald/public_html/static/embed/{row_imgnames.ix[i]}_{row_canids.ix[i]}.png"
            static_url = "http://madthunder.soic.indiana.edu/~aseewald/static/embed/{}_{}.png".format(row_imgnames.ix[i],row_canids.ix[i])
            if not os.path.exists(static_fs):
                subprocess.call(["cp",hyperparams.root("/data/aseewald/COCO/val_candidateimgs/") + row_imgnames.ix[i] + "_0_objectness_" + str(row_canids.ix[i]) + ".jpg",static_fs])
        cluster = predictions[i]
        try:
            csvout.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(tnx,tny,cluster,bycluster[cluster],byclass[gt],iscorrect[i],gt,static_url,num_clusters,num_samples,perfect,even))
        except:
            csvout.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(tnx,tny,cluster,bycluster[cluster],-1,iscorrect[i],gt,static_url,num_clusters,num_samples,perfect,even))

def accuracy_vs_neighbors(labels,clusters,neighbor_counts):
    '''
    '''
    incorrect = {num_neighbors : 0 for num_neighbors in np.unique(neighbor_counts)}
    total = {count: len(np.where(neighbor_counts == count)[0]) for count in np.unique(neighbor_counts)}
    intlabel = {c : i for i,c in enumerate(np.unique(labels))}
    strlabel = {i : c for i,c in enumerate(np.unique(labels))}
    intlabels = np.array([intlabel[label] for label in labels])
    for clusterid in range(len(np.unique(clusters))):
        matching_labels = labels[clusters == clusterid]
        matching_intlabels = intlabels[clusters == clusterid]
        frequencies = np.bincount(matching_intlabels)
        if frequencies.size > 0:
            maximum_cat = strlabel[np.argmax(frequencies)]
            for i, ismaxcat in enumerate(labels == maximum_cat):
                if not ismaxcat and clusters[i] == clusterid:
                    incorrect[neighbor_counts[i]] += 1
    for k in incorrect.keys():
        if total[k] == 0:
            del(incorrect[k])
            del(total[k])
    return {k: ((total[k] - incorrect[k]) / total[k]) for k in incorrect.keys()}

def cluster(mat,num_clusters,full=True,approx_m=3):
    if full: #spectral clustering (slow)
        clust = SpectralClustering(n_clusters=num_clusters)
        predictions = clust.fit_predict(mat)
        dists = np.zeros(predictions.size) #this data is unfortunately not kept by sklearn.
    else:   # fast approximate spectral clustering
        nfst = int(mat.shape[0] / approx_m)
        raw_clust = KMeans(n_clusters=nfst,init='k-means++',tol=1e-7,max_iter=1700)
        raw_pred = raw_clust.fit_predict(mat)
        raw_centroids = raw_clust.cluster_centers_
        nearest,dists = [],[]
        for centroid in raw_centroids: #find points closest to centroids.
            dist = np.apply_along_axis(np.linalg.norm,1,mat - centroid)
            nearest.append(np.argmin(dist))
        for rowid in range(mat.shape[0]): #store distances from each point to its closest centroid
            pred = raw_pred[rowid] 
            dists.append(euclidean(mat[rowid],raw_centroids[pred]))
        spect_clust = SpectralClustering(n_clusters=num_clusters)
        centroid_mat = mat[nearest]
        spect_pred = spect_clust.fit_predict(centroid_mat)
        # now we assign each spectral cluster label to the whole centroid in the raw problem 
        predictions = [spect_pred[raw_pred[i]] for i in range(len(raw_pred))]
    return predictions,dists

def arch_purities(hyperparams,cluster_amounts,kfiles,train_trial,num_spectral_trials=5,sample_amounts=[700,800,1000,1200,1800,2200,2700],full=True):
    '''
    TODO - save both kernel and original repr.
    '''
    #kfiles = reversed([params.root('kernels/embed'),params.root('kernels/vgg'),params.root('kernels/embed_perfect'),params.root('kernels/vgg_perfect'),
              #params.root("kernels/mryoo_x"),params.root("kernels/mryoo"),params.root("kernels/mryoo_perfect"),params.root("kernels/ryoo2_x"),params.root("kernels/ryoo2"),params.root("kernels/ryoo2_perfect")])
    psdfs = []
    dosql("CREATE TABLE IF NOT EXISTS arch_purity (kfile TEXT, num_clusters INT, num_trials INT, num_samples INT, splitid INT, nickname TEXT, quantile FLOAT, netpurity FLOAT, timestep INT, even INt, perfect INT, isfull INT)",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS arch_byclasspurity (kfile TEXT, num_clusters INT,num_trials INT, num_samples INT, splitid INT, nickname TEXT, quantile FLOAT, netpurity FLOAT, timestep INT, even INt, perfect INT,class TEXT)",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS arch_byclusterpurity (kfile TEXT, num_clusters INT,num_trials INT, num_samples INT, splitid INT, nickname TEXT, quantile FLOAT, netpurity FLOAT, timestep INT, even INt, perfect INT,cluster INT)",hyperparams)
    print("created tables")
    for kfile in kfiles:
        pfname = hyperparams.root('kernels/') + kfile
        pick = pickle.load(open(pfname,'rb'))
        if len(pick) == 11:   # back when only COCO, didn't include dataset.
            Kf,Xf,row_dataf,row_imgnames,row_canids,groupmin,timestep,splitid,nickname,_,perfect = pick
            dataset = 'COCO'
        elif len(pick) == 12: # dataset is now exposed.
            Kf,Xf,dataset,row_dataf,row_imgnames,row_canids,groupmin,timestep,splitid,nickname,_,perfect = pick
        else:
            sys.stderr.write("pickle file {} not as expected".format(pfname))
            sys.exit(1)
        if dataset == 'COCO':
            num_existing = readsql("select count(*) from splitcats WHERE dataset = '{}' AND splitid = {} AND seen = 0".format('COCO',splitid),hyperparams)['count'].ix[0]
        elif dataset == 'pascal':
            pasc = readsql("select distinct(category) from splitcats WHERE dataset = '{}'".format('pascal',splitid),hyperparams)
            num_existing = len(transfer_exclude(pasc,'COCO',splitid,'pascal'))
        even = row_dataf.value_counts().min() > (row_dataf.value_counts().max() / 1.2) # there is a little variation still, somehow among the even.
        quantiles =[0.1,0.25,0.5,0.75,1.0]
        ckeys,ikeys = set([]),set([])
        ns = len(row_dataf)
        if ns == 0: continue #I corrupted one of them.
        for numsamp in sample_amounts:
            if len(row_dataf) < numsamp:
                print("Not enough data for sample amount",numsamp,"having {} on kfile={}".format(len(row_dataf),kfile))
                continue
            numsamp = int(numsamp)
            rows = random.sample(range(len(row_dataf)),numsamp)
            # the transposing business is to select certain rows and columns.
            K,X,row_lab = Kf[rows].T[rows].T,Xf[rows],row_dataf.values[rows]
            for num_clusters in (num_existing * np.array(cluster_amounts)).astype(np.int):
                ps,accuracies,iscorrect,purities,cpurities,ipurities,which = [],  [], [],{},{},{},[]
                cols = {'num_clusters' : num_clusters, 'num_samples' : numsamp, 'timestep' : timestep, 'nickname' : nickname, 'splitid' : splitid}
                ips,cps = None,None
                for trial in range(num_spectral_trials):
                    try:
                        print("starting work on",nickname,"even=",even,"numsamp=",numsamp,"splitid=",splitid,"num_clusters=",num_clusters)
                        predictions,nearest_dists = cluster(X,num_clusters,full=full)
                        pure = purity(np.array(row_lab), np.array(predictions), num_clusters, nearest_dists,quantiles)
                        if set(pure[0].keys()) != set(quantiles):
                            print("due to clustering, only quantile=1 available.")
                        for q in pure[0].keys():
                            if q not in purities.keys():
                                purities[q] = [pure[0][q]]
                                ipurities[q] = [pure[1][q]]
                                cpurities[q] = [pure[2][q]]
                            else:
                                purities[q].append(pure[0][q])
                                ipurities[q].append(pure[1][q])
                                cpurities[q].append(pure[2][q])
                        #accuracies.append(accuracy_vs_neighbors(row_lab,predictions,neighbor_counts))
                        print(kfile,num_clusters,pure[0])
                        ps.append(pure[0])
                        idfnew,cdfnew = [],[]
                        for q in pure[0].keys():
                            for cat in pure[1][q].keys():
                                idict = copy.deepcopy(cols)
                                idict['class'] = cat
                                ikeys.add(cat)
                                idict['purity'] = pure[1][q][cat]
                                idict['quantile'] = q
                                idfnew.append(idict)
                            for clust in pure[2][q].keys():
                                cdict = copy.deepcopy(cols)
                                cdict['cluster'] = clust
                                ckeys.add(clust)
                                cdict['purity'] = pure[2][q][clust]
                                cdict['quantile'] = q
                                cdfnew.append(cdict)
                        if ips is None: ips = pd.DataFrame(idfnew)
                        else: ips.append(pd.DataFrame(idfnew))
                        if cps is None: cps = pd.DataFrame(cdfnew)
                        else: cps.append(pd.DataFrame(cdfnew))
                        iscorrect.append(pure[3])
                        which.append(pure[4])
                    except:
                        print("A problem occured")
                        continue
                if len(ps) > 0:
                    print("In practice, having quantiles",ps[0].keys())
                for quant in ps[0].keys():
                    avg_iscorrect = np.array([v for v in pd.DataFrame(iscorrect)[quant]])
                    idata,cdata = pd.DataFrame(ipurities[quant]),pd.DataFrame(cpurities[quant])
                    netpure,ipure,cpure = np.mean(purities[quant]),{cat : idata[cat].mean() for cat in idata.keys()},{clust : cdata[clust].mean() for clust in cdata.keys()}
                    psdfs.append({'even' : even,'num_clusters': num_clusters,'splitid':splitid,'nickname' : nickname,'quantile' : quant, 'purity' : netpure,'kfile' : kfile,
                    
})
                    timestep = -1 if timestep is None else timestep #baselines don't really have 'timestep' notion.
                    dosql("INSERT INTO arch_purity VALUES('{}',{},{},{},{},'{}',{},{},{},{},'{}')".format(kfile,num_clusters,num_spectral_trials,numsamp,splitid,nickname,quant,netpure,timestep,int(even),int(perfect),int(full),dataset),hyperparams)
                    for cat,p in ipure.items():
                        dosql("INSERT INTO arch_byclasspurity VALUES('{}',{},{},{},{},'{}',{},{},{},'{}',{},'{}')".format(kfile,num_clusters,num_spectral_trials,numsamp,splitid,nickname,quant,p,timestep,int(even),int(perfect),cat,int(full),dataset),hyperparams)
                    for clust,p in cpure.items():
                        dosql("INSERT INTO arch_byclusterpurity VALUES('{}',{},{},{},{},'{}',{},{},{},{},{},'{}')".format(kfile,num_clusters,num_spectral_trials,numsamp,splitid,nickname,quant,p,timestep,int(even),int(perfect),clust,int(full),dataset),hyperparams)
                    try:
                        qs = quantiles=ps[0].keys()
                        webnames = mywebsite.webfmt(False,nickname,splitid,num_clusters,numsamp,perfect,even,train_trial,quantiles=qs)
                        correct_stats = np.mean(avg_iscorrect,axis=0)
                        embed_data_arch(webnames[quant][0],hyperparams,X,row_imgnames,row_lab,row_canids,predictions,ipure,cpure,correct_stats,nickname,num_clusters,numsamp,perfect,even)
                    except:
                        continue
                pd.DataFrame(psdfs).to_pickle('pickles/net-{}-{}-{}-{}.pkl'.format(time.time(),kfile,numsamp,num_clusters))
                ips.to_pickle('pickles/byclass-{}-{}-{}-{}.pkl'.format(time.time(),kfile,numsamp,num_clusters))
                cps.to_pickle('pickles/bycluster-{}-{}-{}-{}.pkl'.format(time.time(),kfile,numsamp,num_clusters))
                # the real set of quantiles that actually get used is keys in the ps dicts.
                webnames = mywebsite.webfmt(False,nickname,splitid,num_clusters,numsamp,perfect,even,train_trial,quantiles=qs)
                for quantile in qs:
                    mywebsite.mkpage(webnames[quantile][0],webnames[quantile][1],'arch',nickname,splitid,quantile,perfect,even)
    
def greedy_purities(cluster_amounts,hyperparams,nickname,num_candidates,even,perfect,train_trial,num_spectral_trials=5,sample_amounts=[700,1200,1800,2600]):
    '''
    This studies the effect of kernel types and hyperparameters.
    This does not study receptive field angles (this is for my hand crafted problem.
    '''
    if even:
        description = 'even'
    else:
        description = 'none' # add more later possibly.
    results = pd.DataFrame(columns=['splitid', 'quantile','nickname','k_type','num_fields','num_clusters', 'pooltype', 'netpurity', 'byclass','bycluster','gt','avgiscorrect','iscorrect','description','num_candidates','even','perfect'])
    quantiles =[0.1,0.25,0.5,0.75,1.0]
    grouped_results = pd.DataFrame(columns=[])
    dosql('''CREATE TABLE IF NOT EXISTS purities_greedy (splitid INT, quantile FLOAT, nickname TEXT, k_type TEXT,num_fields INT,num_clusters INT, pooltype TEXT, netpurity FLOAT, byclass TEXT,bycluster TEXT,gt TEXT,avg_iscorrect FLOAT,iscorrect TEXT,description TEXT,num_candidates INT)''',hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS numobj_counts(nickname TEXT, num_candidates INT, quantile FLOAT, numobj INT, avg_iscorrect FLOAT, description TEXT, ktype TEXT)",hyperparams)
    kerns_name = affinity_outfmt('results',hyperparams.splitid,nickname,num_candidates,even,perfect,hyperparams)
    rowdata_name = affinity_outfmt('rowdata',hyperparams.splitid,nickname,num_candidates,even,perfect,hyperparams)
    hyperparams.splitid = int(hyperparams.splitid) #wtf
    try:
        row_data = pickle.load(open(rowdata_name,'rb'))
        row_lab, neighbor_counts, row_ids = row_data['gt'], row_data['num_labeled_neighbors'], row_data['imgname']
        print("unpickling",kerns_name)
        kerns = pickle.load(open(kerns_name,'rb'))
    except:
        print("The kernel does not exist. Try running affinity")
        return
    kerns = kerns.iloc[::-1] # reverse it because i already did forwards.
    for row in kerns.iterrows():
        k_type, K, X, num_fields, hyperparams.splitid = row[1]['field_type'],row[1]['kernel'],row[1]['features'],row[1]['num_fields'],row[1]['splitid']
        print("Starting on ktype=",k_type)
        if K.shape[0] != K.shape[1]: #I accidentally saved some backwards, so reverse that error here.
            K,X = X,K
        if k_type == "dense": #this data is messed up at the moment.
            continue
        pooltype = 'mean'
        print(k_type)
        if not np.array_equal(np.nan_to_num(K),K):
            print("Warning nans exist")
            K = np.nan_to_num(K)
        if not np.array_equal(K,K.T):
            print("Warning, kernel was not saved properly, so rebuilding it from X")
            for i in tqdm(range(X.shape[0])):
                for j in range(i,X.shape[0]):
                    K[i,j] = euclidean(X[i],X[j])
                    K[j,i] = K[i,j]
            kerns.ix[row[0]]['kernel'] = K
            pickle.dump(kerns,open(kerns_name,'wb'))
        # checking for zero-rows *after* filling in with transpose.
        if any([np.count_nonzero(ki) == 0 for ki in K]) or any([np.count_nonzero(xi) == 0 for xi in X]):
            print("There is a zero row, so this is evidently an incomplete design matrix. continuing to the next one...")
            continue
        assert(np.allclose(K,K.T)), "panic: Kernel is somehow not symmetric"
        t1 = time.time()
        if hyperparams.dataset == 'COCO':
            num_existing = readsql("select count(*) from splitcats WHERE dataset = '{}' AND splitid = {} AND seen = 0".format('COCO',hyperparams.splitid),hyperparams)['count'].ix[0]
        elif hyperparams.dataset == 'pascal':
            pasc = readsql("select distinct(category) from splitcats WHERE dataset = '{}'".format('pascal',hyperparams.splitid),hyperparams)
            num_existing = len(transfer_exclude(pasc,'COCO',splitid,'pascal'))
        for num_clusters in (num_existing * np.array(cluster_amounts)).astype(np.int):
            for num_samples in sample_amounts:
                if num_samples > len(X):
                    print("Not enough data for {} samples, continuing".format(num_samples))
                    continue
                rows = random.sample(range(len(X)),num_samples)
                current_row_stats = {'gt' : row_lab[rows],'numobj' : neighbor_counts[rows],'rowids' : row_ids[rows]}
                Xp = X[rows]
                Kp = K[rows].T[rows].T
                ps, ips,accuracies,cps,iscorrect,which = [], [], [], [], [], []
                for trial in range(num_spectral_trials):
                    try:
                        predictions,nearest_dists = cluster(Xp,num_clusters,full=True)
                        print("about to compute purity")
                        pure = purity(np.array(current_row_stats['gt'],dtype=np.str), np.array(predictions), num_clusters, nearest_dists,quantiles)
                        #accuracies.append(accuracy_vs_neighbors(row_lab,predictions,neighbor_counts))
                        ps.append(pure[0])
                        ips.append(pure[1])
                        cps.append(pure[2])
                        iscorrect.append(pure[3])
                        which.append(pure[4])
                        print("With ktype = {}, finished trial {},{},{},{}".format(k_type,hyperparams.splitid,num_clusters,nickname,pure[0]))
                    except:
                        print("An error in computing purity")
                        continue
                if pd.isnull(num_fields): num_fields = -1
                if (ps[0].keys() != set(quantiles)):
                    print("Altert: set of purity keys different from set of quantiles")
                qs = quantiles=ps[0].keys()
                webnames = mywebsite.webfmt(True,nickname,hyperparams.splitid,num_clusters,num_samples,perfect,even,train_trial,quantiles=qs,k_type=k_type)
                for quantile in ps[0].keys():
                    mywebsite.mkpage(webnames[quantile][0],webnames[quantile][1],'greedy',nickname,k_type,hyperparams.splitid,quantile,perfect,even)
                    avg_purity = np.mean([p[quantile] for p in ps])
                    avg_byclass = pd.concat([pd.DataFrame(ip) for ip in ips]).transpose().mean( )
                    avg_bycluster = pd.concat([pd.DataFrame(cp) for cp in cps]).transpose().mean( )
                    qcorrect = np.array([x[quantile] for x in iscorrect])
                    print("Avg purity = {}".format(avg_purity))
                    print("Avg byclass = {}".format(avg_byclass))
                    entry = [hyperparams.splitid, quantile, nickname, k_type, num_fields, num_clusters, pooltype, avg_purity, str(avg_byclass), str(avg_bycluster),str(row_lab),np.mean(qcorrect),floatserial(qcorrect,0),description,num_samples,int(even),int(perfect)]
                    results.loc[-1] = entry
                    results.index += 1
                    stmt = "INSERT INTO purities_greedy VALUES({},{},'{}','{}',{},{},'{}',{},'{}','{}','{}',{},'{}','{}',{},{},{})".format(*entry)
                    print(stmt)
                    dosql(stmt,hyperparams)
                    filtcorrect = {}
                    try:
                        # row_lab is the labels of the current set of rows.
                        if embed_data_greedy(hyperparams,webnames[quantile][0],quantile,nickname,Kp,Xp,current_row_stats,predictions,avg_byclass,avg_bycluster,filtcorrect,hyperparams.splitid,which[0],perfect,even,num_fields):
                            print("Successful web write ")
                        else:
                            print("Unsuccessful web write")
                    except:
                        print("Try block for embed_data_greedy failed")
                    print("About to run numboj correctness stuff.")
                    for i in range(qcorrect.shape[1]):
                        if random.random() < 0.2: #this is slow, so for now speed things up by doing this rarely.
                            correct = []
                            for j in range(qcorrect.shape[0]):
                                if which[j][quantile][i]:
                                    correct.append(qcorrect[j,i])
                            if len(correct) > 0:
                                dosql("INSERT INTO numobj_counts VALUES('{}',{},{},{},{},'{}','{}')".format(nickname,num_candidates,quantile,row_data.ix[i].num_labeled_neighbors.num.values[0],np.mean(correct),description,k_type),hyperparams)
                                filtcorrect[i] = np.mean(correct)
                            else: #insert NULL.
                                dosql("INSERT INTO numobj_counts VALUES('{}',{},{},{},null,'{}','{}')".format(nickname,num_candidates,quantile,row_data.ix[i].num_labeled_neighbors.num.values[0],description,k_type),hyperparams)
                    print("Ran numobj correctness stuff")
                # pd.to_sql I should figure this out.
            #results.to_pickle("{}/puritydata_{}_{}_{}_{}.pkl".format(hyperparams.root("results"),hyperparams.splitid,nickname,num_samples,num_clusters))
        print("clustering k_type={} took {} seconds".format(k_type,str(time.time() - t1)))
    return results

def numobj_correlations( ):
    '''
    Is there a correlation between number of objects and easiness to discover?
    '''
    X = readsql("SELECT * FROM numobj_counts",hyperparams)
    netcorrect,netnum = np.nan_to_num(X['avg_iscorrect'].values), X['numobj'].values
    print("net correlation",pearsonr(netcorrect,netnum))
    X_ab = readsql("SELECT * FROM numobj_counts WHERE ktype = 'above_below'",hyperparams)
    X_cnnapp = readsql("SELECT * FROM numobj_counts WHERE ktype = 'cnnapp'",hyperparams)
    X_metric = readsql("SELECT * FROM numobj_counts WHERE ktype = 'metric'",hyperparams)
    print("We expect metric correlation is higher than above below correlation")
    print("above below correlation",pearsonr(np.nan_to_num(X_ab['avg_iscorrect'].values),X_ab['numobj'].values))
    print("cnnapp correlation",pearsonr(np.nan_to_num(X_cnnapp['avg_iscorrect'].values),X_cnnapp['numobj'].values))
    print("metric correlation",pearsonr(np.nan_to_num(X_metric['avg_iscorrect'].values),X_metric['numobj'].values))

def frequency_correlations(by_class_purities ):
    bcp = pd.DataFrame(by_class_purities)
    if not os.path.exists( ): #the count file.
        counts = readsql("SELECT count(*),category FROM ground_truth",hyperparams) #should probably make a WHERE list of unknowns.
    df = bcp.join(counts) #figure out real pandas join syntax.
    print("frequency correlation: ",pearsonr(df['frequency'].values,df['purity'].values))

def quantile_plot(nickname):
    X = readsql("SELECT * FROM purities_greedy WHERe nickname = '{}'".format(nickname),hyperparams)
    g = sns.FacetGrid(X,col='quantile',hue='k_type')
    g.map(plt.scatter,'num_clusters','netpurity').add_legend
    plt.savefig('quantile.png')

def num_can_plot(nickname):
    '''
    Relationship between number of candidates and purity.
    '''
    X = readsql("SELECT * FROM purities_greedy WHERE nickname = '{}' AND quantile = 1.0".format(nickname),hyperparams)
    Xg = X.groupby(['k_type','nickname','num_candidates']).mean( )
    g = sns.FacetGrid(X,row="k_type",hue="num_candidates")
    g.map(plt.scatter,"num_clusters","netpurity").add_legend()
    plt.savefig('can.png')

def num_field_plot(nickname,num_candidates):
    var_types = ["'metric'","'euclidean'","'metricinnerproduct'","'arb'","'innerproduct'"]
    X =  readsql("SELECT * FROM purities_greedy WHERE nickname = '{}' AND k_type IN ({}) AND quantile = 1.0 AND num_candidates = {}".format(nickname,",".join(var_types),num_candidates),hyperparams)
    num = len(np.unique(X['num_fields']))
    g = sns.FacetGrid(X,row="k_type",hue="num_fields")
    g.map(plt.plot,"num_clusters","netpurity").add_legend()
    plt.savefig('nf.png')

if __name__ == '__main__':
    if args.action == 'arch':
        hyperparams = hp.minimal_dset[args.dataset]
    elif args.action == 'greedy':
        hyperparams = hp.greedy_hp[args.nickname]
    cluster_amounts = [float(x) for x in args.cluster_amounts.split(',')]
    kfiles = os.listdir(hyperparams.root('kernels'))
    random.shuffle(kfiles)
    client = memcache.Client([(constants.memcache_server,11211)])
    if args.action == 'arch':
        for i,kf in enumerate(kfiles):
            if kf.split('_')[0] != 'arch':
                del kfiles[i]
                continue
            if (int(kf.split('_')[-1]) == args.trial):
                del kfiles[i]
                continue
            if client.get(kf) == 1:
                del kfiles[i]
                continue
        kfiles = [kf for kf in kfiles if (kf.split('_')[0] == 'arch')]
        if args.multiproc:
            with mp.Pool(args.nthreads) as pool:
                pool.map(lambda x:arch_purities(hyperparams,cluster_amounts,[x]),kfiles,args.trial)
        else:
            arch_purities(hyperparams,cluster_amounts,kfiles,args.trial)
    elif args.action == 'greedy':
        for even in [True,False]:
            for perfect in [True,False]:
                    greedy_purities(cluster_amounts,hyperparams,args.nickname,args.num_candidates,even,perfect,args.trial)
