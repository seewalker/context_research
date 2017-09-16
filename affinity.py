'''
Alexander Seewald 2016
aseewald@indiana.edu

These functions save the kernels, reading from data previously stored. In the greedy case, that is perfect_densevecs and perfect_candistsrs in the SQL database. In the arch case, that is arch_ctx_reprs in the SQL database. This is necessary for purity-style evaluation.

This program has an argparse-style command line interface. Running this program with the '--help' flag will describe the way to use it.
'''
import pickle
import constants
import copy
import functools
import xml.etree.ElementTree as ET
import gzip
import subprocess
import multiprocessing
import numpy as np
from numpy import array,nan,float32 #eval needs these names in current namespace.
import pandas as pd
from skimage.io import imsave, imread
import os
import sys
import sqlite3
import sys
import random
import math
import tempfile
import line_profiler
from scipy.spatial.distance import euclidean
import scipy.linalg
import matplotlib.pyplot as plt
from hyperparams import chisquared, greedy_hp, arch_hp
import objectGraph
from utils import *
import argparse

__author__ = "Alex Seewald"

parser = argparse.ArgumentParser()
parser.add_argument('nickname')
parser.add_argument('dataset')
parser.add_argument('--num_candidates',default=3000,type=int)
parser.add_argument('--trial',default=0,type=int)
parser.add_argument('--train_dataset',default='COCO')
subparsers = parser.add_subparsers(dest='action',help=" ")
baseline_parser = subparsers.add_parser('baseline')
greedy_parser = subparsers.add_parser('greedy')
greedy_parser.add_argument('--perfect',action='store_false',default=True)
greedy_parser.add_argument('-metric_num',default=5,type=int)
greedy_parser.add_argument('--even',action='store_true',default=False)
greedy_parser.add_argument('--hackRestart',action='store_true',default=False)
arch_parser = subparsers.add_parser('arch')
# greedy_parser doesn't have a splitid choice because that's predetermined.
arch_parser.add_argument('splitid',type=int)
arch_parser.add_argument('--timestep',type=int)

args = parser.parse_args()
if args.action == 'greedy':
    hyperparams = greedy_hp[args.nickname]
elif args.action == 'arch':
    hyperparams = arch_hp[args.nickname]

class OnTheFlySuperpixel( ):
    def __init__(self,object_distribution,centroid):
        self.object_distribution = object_distribution
        self.centroid = centroid

class OnTheFlyCandidate( ):
    def __init__(self,object_distribution,centroid):
        self.object_distribution = object_distribution
        self.centroid = centroid

def row_normalize(matrix:np.ndarray) -> np.ndarray:
    '''Normalizes a matrix by rows.
       If a row is zero, keep it zero rather than trying to have length one.
    '''
    row_sums = matrix.sum(axis=1)
    iszero = np.where(row_sums == 0)[0]
    matrix = matrix / row_sums[:, np.newaxis]
    # set zeros afterwards because the above will create nans on all-zero rows.
    matrix[iszero] = 0 
    return matrix

def arch_baseline_kernels(splitid,canmax,dataset,train_dataset):
    cans,descrs,iseven,isperfect = [],[],[],[]
    for nick in ["vggnet","random","embed"]:
        for can_t in ['detected_natfreq','perfect_natfreq','detected_even','perfect_even']:
            if dataset != train_dataset:
                catdf = readsql(f"select distinct(category) from splitcats where dataset = '{dataset}'",hyperparams)
                catdf = transfer_exclude(catdf,train_dataset,splitid,dataset)['category']
            else:
                catdf = readsql(f"select category from splitcats where splitid = {splitid} AND dataset = '{dataset}' AND seen = 0",hyperparams)['category']
            catlist = ','.join(["'" + cat + "'" for cat in catdf])
            x = readsql(f"SELECT imgname,canid,category,repr FROM raw_ctx_reprs WHERE nickname = '{nick}' AND type = '{can_t}' AND category in ({catlist}) AND dataset = '{dataset}'",hyperparams)
            if 50 < len(x):
                x = x.sample(min(canmax,len(x)))
                cans.append(x)
                descrs.append(f"{nick}-{can_t}")
                iseven.append("even" in can_t)
                isperfect.append("perfect" in can_t)
    for i,can in enumerate(cans):
        cans[i] = can
        cans[i]['repr'] = cans[i]['repr'].apply(eval)
        cans[i]['repr'] = cans[i]['repr'].apply(lambda x: np.array(x).flatten())
        cans[i].reset_index(inplace=True)
    for j,candidates in enumerate(cans): #iterate over type of candidates.
        N_j = len(candidates)
        K = np.zeros((N_j,N_j))
        X = np.array([np.array(xi) for xi in candidates['repr']]) #stack the reprs.
        print(f"{descrs[j]},X.shape={X.shape}")
        for i in range(N_j):
            if i % 10 == 0: print(descrs[j],i/N_j)
            distmat = X - np.array(candidates.ix[i]['repr'])
            K[i] = np.apply_along_axis(np.linalg.norm,1,distmat)
        name = descrs[j]
        out = params.root(f'kernels/baseline_{dataset}_{splitid}_{descrs[j]}_none')
        pickle.dump((K,X,dataset,candidates['category'],candidates['imgname'],candidates['canid'],None,None,splitid,None,iseven[j],isperfect[j]),open(out,'wb'))

# Need to figure out the perfect vs naturalfreq naming mixup.
def arch_kernels(nickname,splitid,timestep,trial,dataset,train_dataset,include_baselines=False):
    descrs = []
    cans = []
    iseven = []
    isperfect= []
    cans.append(readsql(f"SELECT imgname,canid,category,repr FROM arch_ctx_reprs WHERE nickname = '{nickname}' AND splitid = {splitid} AND isperfect =  1 AND iseven = 1 AND timestep = {timestep} AND trial = {trial} AND dataset = '{dataset}'",hyperparams))
    iseven.append(1);isperfect.append(1)
    cans.append(readsql("SELECT imgname,canid,category,repr FROM arch_ctx_reprs WHERE nickname = '{}' AND splitid = {} AND isperfect =  1 AND iseven = 0 AND timestep = {} AND trial = {} AND dataset = '{}'".format(nickname,splitid,timestep,trial,dataset),hyperparams))
    iseven.append(0);isperfect.append(1)
    cans.append(readsql("SELECT imgname,canid,category,repr FROM arch_ctx_reprs WHERE nickname = '{}' AND splitid = {} AND isperfect =  0 AND iseven = 1 AND timestep = {} AND trial = {} AND dataset = '{}'".format(nickname,splitid,timestep,trial,dataset),hyperparams))
    iseven.append(1);isperfect.append(0)
    cans.append(readsql("SELECT imgname,canid,category,repr FROM arch_ctx_reprs WHERE nickname = '{}' AND splitid = {} AND isperfect =  0 AND iseven = 0 AND timestep = {} AND trial = {} AND dataset = '{}'".format(nickname,splitid,timestep,trial,dataset),hyperparams))
    iseven.append(0);isperfect.append(0)
    descrs.append("{}-{}-perfect-even".format(nickname,timestep))
    descrs.append("{}-{}-perfect-naturalfreq".format(nickname,timestep))
    descrs.append("{}_detected-even".format(nickname))
    descrs.append("{}-{}-detected-naturalfreq".format(nickname,timestep))
    maxlen = max([len(can) for can in cans])
    #N = max(min([len(can) for can in cans]),maxlen/2) # don't discard more than half of data compared to max no matter what.
    groupmin = min([len(can) for can in cans]) # don't discard more than half of data compared to max no matter what.
    for i,can in enumerate(cans):
        if len(can) == 0:
            print("No data for {}, continuing".format(descrs[i]))
            continue
        cans[i] = can
        cans[i]['repr'] = cans[i]['repr'].apply(eval)
        cans[i]['repr'] = cans[i]['repr'].apply(lambda x: np.array(x).flatten())
        cans[i].reset_index(inplace=True)
    for j,candidates in enumerate(cans): #iterate over type of candidates.
        N_j = len(candidates)
        K = np.zeros((N_j,N_j))
        X = np.array([np.array(xi) for xi in candidates['repr']]) #stack the reprs.
        for i in range(N_j):
            if i % 100 == 0: print(descrs[j],i/N_j)
            distmat = X - np.array(candidates.ix[i]['repr'])
            K[i] = np.apply_along_axis(np.linalg.norm,1,distmat)
        name = descrs[j]
        out = hyperparams.root(f'kernels/arch_{dataset}_{nickname}_{splitid}_{name}_{timestep}_{trial}')
        pickle.dump((K,X,dataset,candidates['category'],candidates['imgname'],candidates['canid'],groupmin,timestep,splitid,nickname,iseven[j],isperfect[j]),open(out,'wb'))
         
def greedy_kernels(hyperparams,nickname,pixnickname,splitid,perfect,hackRestart,metric_num=7,num_candidates=8000,easiness=False,restartNum=3000,redoMetric=False,ignore=set(),even=False,savestep=400):
    '''
    TODO - add the 'even' idea. 
    '''
    dosql("CREATE TABLE IF NOT EXISTS kernels (nickname TEXT,splitid TEXT, trial INT, num_candidates INT)",hyperparams)
    # natural join with densevecs to guarantee that a corresponding densevec exists.
    # densevecs "nickname" is not to be used. I'm just considering that to be global so far.
    if perfect:
        relation = f"perfect_densevecs NATURAL JOIN (SELECT nickname,patchname,vec AS appearance FROM perfect_candistrs WHERE vec NOT LIKE '%nan%') R NATURAL JOIN (SELECT * FROM perfect_isknown WHERe seen = 0 AND splitid = {splitid}) S NATURAL JOIN (SELECT patchname,miny,maxy,minx,maxx FROM perfect_bbox) T"
        can_data = readsql(f"SELECT imgname,patchname,category AS classname,appearance,miny,maxy,minx,maxx FROM {relation}",hyperparams)
    else:
        relation = '''ground_truth
        NATURAL JOIN candidate_object_distribution
        NATURAL JOIN candidate_centroid
        INNER JOIN densevecs ON candidate_centroid.imgname = densevecs.imgname AND candidate_centroid.canid = densevecs.canid
        '''
        can_data = readsql( f'''SELECT candidate_centroid.imgname,candidate_centroid.canid,distr,classname,y,x
                               FROM {relation}''',hyperparams)
    if even:
        freqs = readsql(f"SELECT category,count(*) FROM {relation} GROUP BY category",hyperparams)
        # sometimes its either way so failover.
        try:
            minfreq = min(freqs['count(*)'])
        except:
            minfreq = min(freqs['count'])
        # randomly shuffle rows so that 'head' does proper random sampling.
        can_data.reindex(np.random.permutation(can_data.index))
        can_data = can_data.groupby('classname').head(minfreq)
    else:
        can_data = can_data.sample(num_candidates)
    results = pd.DataFrame(columns=['field_type','kernel','features','num_fields','splitid','perfect','even'] )
    rel = "perfect_densevecs" if perfect else "densevecs"
    denseeg = readsql(f"SELECT vec FROM {rel} WHERE nickname = '{nickname}' LIMIT 1",hyperparams)['vec']
    densenumfeats = np.fromstring(denseeg.ix[0],sep=",").size
    nclasses = len(hyperparams.possible_splits[splitid]['known']) + 1 #recall, fully_conv has a 'None' option.
    dims = {'objectGraph' : ((2 * 20) + 1) * nclasses,'appearance' : nclasses-1,'sparse' : 10, 'rand': 61 * nclasses, 'bag' : nclasses}
    row_data = pd.DataFrame(columns=['gt', 'num_labeled_neighbors','imgname'])
    imgname_txt = ",".join(["'" + x + "'" for x in np.unique(can_data['imgname'])])
    sp_data = readsql(f"SELECT y,x,imgname,spid,vec FROM sp_centroid NATURAL JOIN sp_object_distribution WHERE imgname IN ({imgname_txt}) AND splitid = {splitid} AND nickname = '{pixnickname}' ORDER BY imgname,spid ASC",hyperparams).set_index('imgname')
    can_data = can_data[can_data['imgname'].isin(sp_data.index)]
    N = len(can_data)
    K_ab = (np.zeros((N, N)),np.zeros((N,dims['objectGraph']))) #find out if I can use a lower-memory dtype.
    K_ab_exclusive = np.zeros((N, N))
    K_cnnapp = (np.zeros((N, N)),np.zeros((N,dims['appearance']+1)))
    K_bag = (np.zeros((N, N)),np.zeros((N,dims['bag'])))
    K_lr = (np.zeros_like(K_ab[0]),np.zeros((N,dims['objectGraph'])))
    K_quad = (np.zeros_like(K_ab[0]),np.zeros((N,dims['objectGraph'])))
    K_rand = (np.zeros_like(K_ab[0]),np.zeros((N,dims['rand'])))
    metric_fields = readsql(f"SELECT distinct(num_fields) FROM metrics WHERE nickname = '{nickname}'",hyperparams).values.squeeze()
    metric_fields.sort()
    metric_fields = metric_fields[np.linspace(1,len(metric_fields)-1,metric_num).astype(np.int)]
    raw_field_max = readsql(f"SELECT max(fullidx) FROM subset_correspondence WHERE nickname = '{nickname}'",hyperparams).values[0][0]
    # K_inner is u dot v.
    # K_innerwithmetric u dot M dot v.
    # K_euc is simply euclidean distance of vectors, where receptive fields *are* the learned ones.
    K_inner,K_innerwithmetric,K_euc = {num_fields : (np.zeros_like(K_ab[0]),np.zeros((N,num_fields))) for num_fields in metric_fields},{num_fields : (np.zeros_like(K_ab[0]),np.zeros((N,num_fields))) for num_fields in metric_fields},{num_fields : (np.zeros_like(K_ab[0]),np.zeros((N,num_fields))) for num_fields in metric_fields}
    K_dense = (np.zeros((N,densenumfeats)),np.zeros((N,N)))
    metric_data = readsql("SELECT * FROM metrics WHERE num_fields IN ({}) AND nickname = '{}'".format(",".join(map(str,metric_fields.tolist())),nickname),hyperparams)
    metrics = {num_fields : None  for  num_fields in metric_fields}
    active_subsets = {num_fields : None  for  num_fields in metric_fields}
    vecs = {k : {num_fields : np.zeros((N,num_fields)) for num_fields in metric_fields}for k in {'arb','union'}}
    maxfield = 0
    # K_metric is the full thing.
    K_metric = {num_fields : (np.zeros_like(K_ab[0]),np.zeros((N,num_fields))) for num_fields in metric_fields}
    # random fields, same number as main idea.
    K_arb = {num_fields : (np.zeros_like(K_ab[0]),np.zeros((N,num_fields))) for num_fields in metric_fields}
    for num_fields in metric_fields:
        dat = metric_data[(metric_data.num_fields == num_fields)]
        metrics[num_fields] = np.array(eval(dat['metric'].values[0]))
        active_subsets[num_fields] = np.fromstring(dat['active_subset'].values[0][1:-1],sep=",").astype(np.int)
    arb = {n : random.sample(range(raw_field_max),n) for n in metric_fields}
    print(f"candidates={N},splitid={splitid},nickname={nickname}")
    num_neighbors = readsql("SELECT num,imgname FROM numobjects",hyperparams)
    Knames = {'above_below' : affinity_outfmt('greed_abovebelow',splitid,nickname,restartNum,even,perfect,hyperparams),
              'left_right' : affinity_outfmt('greed_leftright',splitid,nickname,restartNum,even,perfect,hyperparams),
              'quad' : affinity_outfmt('greed_quad',splitid,nickname,restartNum,even,perfect,hyperparams),
              'dense' : affinity_outfmt('greed_dense',splitid,nickname,restartNum,even,perfect,hyperparams),
              'cnnapp' : affinity_outfmt('greed_cnnapp',splitid,nickname,restartNum,even,perfect,hyperparams),
              'bag' : affinity_outfmt('greed_bag',splitid,nickname,restartNum,even,perfect,hyperparams),
              'vecs' : affinity_outfmt('greed_vecs',splitid,nickname,restartNum,even,perfect,hyperparams),
              'arb' : {num_fields : affinity_outfmt(f'arb{num_fields}',splitid,nickname,restartNum,even,perfect,hyperparams) for num_fields in metric_fields},
              'euc' : {num_fields : affinity_outfmt(f'euc{num_fields}',splitid,nickname,restartNum,even,perfect,hyperparams) for num_fields in metric_fields},
              'inner' : {num_fields : affinity_outfmt(f'inner{num_fields}',splitid,nickname,restartNum,even,perfect,hyperparams) for num_fields in metric_fields},
              'metric' : {num_fields : affinity_outfmt(f'metric{num_fields}',splitid,nickname,restartNum,even,perfect,hyperparams) for num_fields in metric_fields},
              'innerwithmetric' : {num_fields : affinity_outfmt(f'innerwithmetric{num_fields}',splitid,nickname,restartNum,even,perfect,hyperparams) for num_fields in metric_fields}}
    def logdump(K,name):
        print("dumping {}: p(K[0] = 0) = {} is p(K[1] = 0) is {}".format(name,np.count_nonzero(K[0])/K[0].size,np.count_nonzero(K[1])/K[1].size))
        pickle.dump(((row_normalize(K[0]),K[1]),row_data), open(name,'wb'))
    if hackRestart:
        # row data only needs to be loaded once, strictly speaking, so do it for above_below.
        (K_ab,row_data) = pickle.load(open(Knames['above_below'],'rb'))
        (K_lr,_) = pickle.load(open(Knames['left_right'],'rb'))
        (K_quad,_) = pickle.load(open(Knames['quad'],'rb'))
        (K_dense,_) = pickle.load(open(Knames['dense'],'rb'))
        (K_cnnapp,_) = pickle.load(open(Knames['cnnapp'],'rb'))
        (K_bag,_) = pickle.load(open(Knames['bag'],'rb'))
        vecs,_ = pickle.load(open(Knames['vecs'],'rb'))
        for num_fields in metric_fields:
            inp,_ = pickle.load(open(Knames['arb'][num_fields],'rb'))
            K_arb[num_fields] = inp
            inp,_ = pickle.load(open(Knames['euc'],'rb'))
            K_euc[num_fields] = inp
            inp,_ = pickle.load(open(Knames['metric'],'rb'))
            K_metric[num_fields] = inp
            inp,_ = pickle.load(open(Knames['innerwithmetric'],'rb'))
            K_innerwithmetric[num_fields] = inp
            inp,_ = pickle.load(open(Knames['inner'],'rb'))
            K_inner[num_fields ] = inp
        numdone = len(row_data)
    else:
        numdone = 0
    for i in range(numdone,len(can_data)):
        can = can_data.iloc[i]
        print("nickname={},even={},{}/{}={} of the way done".format(nickname,str(even),str(i),str(len(can_data)),str(float(i) / len(can_data))))
        idkey = 'patchname' if perfect else 'canid'
        row_data.loc[len(row_data)] = [can['classname'], num_neighbors[num_neighbors['imgname'] == can['imgname'] + ".jpg"],(can['imgname'],can[idkey])]
        if perfect:
            dense = readsql(f"SELECT * FROM perfect_densevecs WHERE splitid = {splitid} AND imgname = '{can['imgname']}' AND patchname = '{can['patchname']}'",hyperparams)
        else:
            dense = readsql(f"SELECT * FROM densevecs WHERE splitid = {splitid} AND imgname = '{can['imgname']}' AND canid = {can['canid']}",hyperparams)
        dense_vector = np.fromstring(dense['vec'].values[0],sep=',')
        K_dense[0][i] = dense_vector
        sp = sp_data.ix[can['imgname']]
        row_superpixels = [OnTheFlySuperpixel(np.fromstring(sp.ix[i]['vec'][1:-1],sep=","),sp.ix[i][['y','x']]) for i in range(len(sp))]
        row_can = OnTheFlyCandidate(np.array(eval(can['appearance'])),can[['y','x']])
        row_og_ab = objectGraph.object_graph(objectGraph.above_below,row_can,row_superpixels, include_appearance=True, pooltypes=['mean'],visualize=True)['mean']
        row_og_lr = objectGraph.object_graph(objectGraph.left_right,row_can,row_superpixels,include_appearance=True, pooltypes=['mean'])['mean']
        row_og_rand = objectGraph.object_graph(objectGraph.randomly,row_can,row_superpixels,include_appearance=True, pooltypes=['mean'])['mean']
        row_og_quad = objectGraph.object_graph(objectGraph.quadrants,row_can,row_superpixels, include_appearance=True, pooltypes=['mean'],visualize=True)['mean']
        row_og_bag = np.mean(np.array([neighbor.object_distribution for neighbor in row_superpixels]),axis=0)
        # assign the raw features.
        K_ab[1][i],K_lr[1][i],K_quad[1][i],K_bag[1][i] = row_og_ab.flatten(), row_og_lr.flatten(), row_og_quad.flatten(), row_og_bag.flatten()
        K_cnnapp[1][i] = row_can.object_distribution
        for num_fields in metric_fields:
            vecs['arb'][num_fields][i] = dense_vector[arb[num_fields]]
            vecs['union'][num_fields][i] = dense_vector[active_subsets[num_fields]]
            K_arb[num_fields][1][i] = vecs['arb'][num_fields][i]
            K_euc[num_fields][1][i] = vecs['union'][num_fields][i]
            K_metric[num_fields][1][i] = vecs['union'][num_fields][i]
            K_innerwithmetric[num_fields][1][i] = vecs['union'][num_fields][i]
            K_inner[num_fields][1][i] = vecs['union'][num_fields][i]
        for j in range(i):
            K_ab[0][i,j] = euclidean(K_ab[1][i],K_ab[1][j])
            K_lr[0][i,j] = euclidean(K_lr[1][i],K_lr[1][j])
            K_quad[0][i,j] = euclidean(K_quad[1][i],K_quad[1][j])
            K_bag[0][i,j] = euclidean(K_bag[1][i],K_bag[1][j])
            K_cnnapp[0][i,j] = euclidean(K_cnnapp[1][i],K_cnnapp[1][j])
            K_dense[0][i,j] = euclidean(K_dense[1][i],K_dense[1][j])
            for num_fields in metric_fields:
                K_arb[num_fields][0][i,j] = euclidean(vecs['arb'][num_fields][i],vecs['arb'][num_fields][j])
                K_euc[num_fields][0][i,j] = euclidean(vecs['union'][num_fields][i],vecs['union'][num_fields][j])
                diff = vecs['union'][num_fields][i] - vecs['union'][num_fields][j]
                K_metric[num_fields][0][i,j] = math.sqrt(np.dot(np.dot(diff,metrics[num_fields]),diff))
                K_innerwithmetric[num_fields][0][i,j] = np.dot(np.dot(vecs['union'][num_fields][i],metrics[num_fields]),vecs['union'][num_fields][j])
                K_inner[num_fields][0][i,j] = np.dot(vecs['union'][num_fields][i],vecs['union'][num_fields][j])
        # saving the kernels.
        if (i % savestep == (savestep-1)) or (i == len(can_data)-1):
            logdump(K_ab,Knames['above_below'])
            logdump(K_lr,Knames['left_right'])
            logdump(K_quad,Knames['quad'])
            logdump(K_dense,Knames['dense'])
            logdump(K_cnnapp,Knames['cnnapp'])
            logdump(K_bag,Knames['bag'])
            pickle.dump((vecs,row_data),open(Knames['vecs'],'wb'))
            for num_fields in metric_fields:
                logdump(K_arb[num_fields],Knames['arb'][num_fields])
                logdump(K_euc[num_fields],Knames['euc'][num_fields])
                logdump(K_metric[num_fields],Knames['metric'][num_fields])
                logdump(K_innerwithmetric[num_fields],Knames['innerwithmetric'][num_fields])
                logdump(K_inner[num_fields],Knames['inner'][num_fields])
    results.loc[len(results)] = ['above_below', row_normalize(K_ab[0]),K_ab[1],None,splitid,perfect,even]
    results.loc[len(results)] = ['left_right', row_normalize(K_lr[0]),K_lr[1],None,splitid,perfect,even]
    results.loc[len(results)] = ['quad', row_normalize(K_quad[0]),K_quad[1],None,splitid,perfect,even]
    #results.loc[len(results)] = ['random', row_normalize(K_rand),None,splitid,perfect,even]
    results.loc[len(results)] = ['dense', row_normalize(K_dense[0]),K_dense[1],None,splitid,perfect,even]
    results.loc[len(results)] = ['cnnapp', row_normalize(K_cnnapp[0]),K_cnnapp[1],None,splitid,perfect,even]
    for num_fields in metric_fields:
        results.loc[len(results)] = ['arb', row_normalize(K_arb[num_fields][0]),vecs['arb'][num_fields],num_fields,splitid,perfect,even]
        results.loc[len(results)] = ['metric', row_normalize(K_metric[num_fields][0]),vecs['union'][num_fields],num_fields,splitid,perfect,even]
        results.loc[len(results)] = ['euclidean', row_normalize(K_euc[num_fields][0]),vecs['union'][num_fields],num_fields,splitid,perfect,even]
        results.loc[len(results)] = ['innerproduct', row_normalize(K_inner[num_fields][0]),vecs['union'][num_fields],num_fields,splitid,perfect,even]
        results.loc[len(results)] = ['metricinnerproduct', row_normalize(K_innerwithmetric[num_fields][0]),vecs['union'][num_fields],num_fields,splitid,perfect,even]
    # randomly shuffle the rows because order doesn't matter for correctness, but it is nice to get evaluation results occuring on a mixture of types
    # as time goes on.
    results = results.reindex(np.random.permutation(results.index))
    results.to_pickle(affinity_outfmt('results',splitid,nickname,num_candidates,even,perfect,hyperparams))
    pickle.dump(row_data,open(affinity_outfmt('rowdata',splitid,nickname,num_candidates,even,perfect,hyperparams),'wb'))

if args.action == "greedy":
    greedy_kernels(hyperparams,args.nickname,hyperparams.pixnickname,hyperparams.splitid,args.perfect,args.hackRestart,metric_num=args.metric_num,num_candidates=args.num_candidates,even=args.even)
elif args.action == "arch":
    arch_kernels(args.nickname,args.splitid,args.timestep,args.trial,args.dataset,args.train_dataset)
elif args.action == "baseline":
    arch_baseline_kernels(args.splitid,args.num_candidates,args.dataset,args.train_dataset)
