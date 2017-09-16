'''
Alex Seewald 2016.
aseewald@indiana.edu

This module relates to the receptive field learning approach.

Note, this program requires the changes to the metric_learn.lsml module specified in the lsml_custom.py file in this directory.
'''
import random
import sys
import subprocess
import sqlite3
import gc
import datasets
import time
import argparse
import collections
import psutil
import warnings
import pickle
import multiprocessing
import socket
import scipy.stats as ss
import seaborn as sns
import deepdish as dd
import pandas as pd
import numpy as np
from numpy import array,float32 #when things get 'eval'd, the names are bare.
np.set_printoptions(threshold=np.nan)
from sklearn.neighbors import KernelDensity
from sklearn import random_projection
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.linear_model import SGDClassifier
from metric_learn.lsml import LSML
import hyperparams as hp
from utils import *
from densevecs import *
import dataproc
from mytypes import *

__author__ = "Alex Seewald"

parser = argparse.ArgumentParser()
parser.add_argument('nickname',help="nickname associated with this metric learning.")
subparsers = parser.add_subparsers(dest='action')
train_parser= subparsers.add_parser('learn_metric')
data_parser = subparsers.add_parser('mkdense')
pixel_parser = subparsers.add_parser('superpix')
field_parser = subparsers.add_parser('generate_fields')
vis_parser= subparsers.add_parser('visualize')
parser.add_argument('--trial',default=0,help="",type=int)
pixel_parser.add_argument('--tstep',help="if making superpixels, iteration to go from",type=int)
pixel_parser.add_argument('--numfuse',default=2,help="if making superpixels",type=int)
parser.add_argument('--densenum',default=1000,help="when generating receptive fields, produce (densenum * nclasses) dimensional dense vectors.")
train_parser.add_argument('--from_scratch',action='store_true',default=False)
train_parser.add_argument('--num_fields',default=360,help="Stop learning receptive fields when we have this many.")
train_parser.add_argument('--prop_memory',default=0.4,type=float)
train_parser.add_argument('--perfect',default=True,help="If true, train with exact COCO segmenatations rather than BING things which happen to overlap",type=bool)
train_parser.add_argument('--max_in_memory',default=10000,help="Maximum matrix width fitting in memory.")
train_parser.add_argument('--num_select_constraints',default=1000,help="Computing gradient takes time proportional to number of constraints currently considered. When making big selection gradient, use this many. Should be less than num_select_constraints.")
train_parser.add_argument('--num_step_constraints',default=6000,help="Computing gradient takes time proportional to number of constraints currently considered. When making less-costly step gradients, use this many. Should be greater than num_select_constraints.")
train_parser.add_argument('--test_per_field',default=10000,help="For every receptive field added, do this many testing iterations to check effect on num_violations (violations of constraints).")
train_parser.add_argument('--init_stepsize',default=1e-4,type=float)
args = parser.parse_args()
assert(args.nickname in hp.greedy_hp.keys()),"You must pick a nickname specified in hyperparams.py"
hyperparams = hp.greedy_hp[args.nickname]

# keep track of separate best stepsizes for selection SGD and tuning SGD.
if args.action == 'learn_metric':
    best_stepsize = {'step' : args.init_stepsize, 'select' : args.init_stepsize}

# save hostname for ease of keeping track because i run this in parallel on a few machines.
suffix_by_host = f'_{time.time()}_{socket.gethostname()}_{hyperparams.pixnickname}.tsv'
tsv_perfect = open('/fast-data/aseewald/denseperfect' + suffix_by_host,'a')
tsv = open('/fast-data/aseewald/dense' + suffix_by_host,'a')

# global variables for use by "inner" function, because it is called in parallel to avoid passing overhead.
dosql("CREATE TABLE IF NOT EXISTS densevecs (nickname TEXT, imgname TEXT, canid INT, type TEXT, vector TEXT, splitid TEXT, trial INT, dataset TEXT, isperfect INT, PRIMARY KEY(nickname,imgname,canid,type,splitid,trial,dataset,isperfect))",hyperparams,whichdb="postgres")
done = readsql("SELECT imgname, splitid, canid, type FROM densevecs",hyperparams)
imgshapes = readsql("SELECT * FROM imgsize",hyperparams)
if args.action != "generate_fields":
    try:
        offset_candidates = np.loadtxt(hyperparams.root(f'fields/split_offset_candidates_{hyperparams.pixnickname}.npy'))
        angle_candidates = np.loadtxt(hyperparams.root(f'fields/split_angle_candidates_{hyperparams.pixnickname}.npy'))
    except:
        print(f"Before doing action {args.action} you have to generate the fields")
        sys.exit(1)

def perfect_inner(x,just_pik=False,imgdir="val_images/"):
    imgName,splitid,dataset,trial,isperfect,split,hyperparams = x
    row = imgshapes[imgshapes['imgname'] == imgName + ".jpg"]
    if row.size == 0:
        row = imgshapes[imgshapes['imgname'] == imgName]
    if row.size == 0:
        return
    imgshape = (row['height'].values[0],row['width'].values[0])
    print(imgName)
    perfect_bbox = readsql(f"SELECT * FROM perfect_bbox WHERE imgname = '{hyperparams.root(imgdir+imgName+\".jpg\")}' AND isexpanded = 0 AND isxl = 0")
    if len(perfect_bbox) == 0:
        return
    for df in perfect_bbox.iterrows():
        # ground_truth is encoded in the patchname
        df = df[1]
        num_underscore = df['patchname'].count('_')
        parts = df['patchname'].split('_')
        if num_underscore == 3:
            ground_truth = parts[2]
        elif num_underscore == 4: #some like 'hog_dog' have underscores. I chose an inconvenient delimiter.
            ground_truth = '_'.join(parts[2:4])
        print(df['patchname'],ground_truth)
        isknown = int(ground_truth in split)
        insert_ifnotexists(f"SELECT * FROM perfect_isknown WHERE splitid = '{splitid}' AND imgname = '{imgName}' AND patchname = '{df['patchname']}'",(f"INSERT INTO perfect_isknown VALUES ('{imgName}','{splitid}','{df['patchname']}','{ground_truth}',{isknown})"),hyperparams)
        if just_pik:
            continue
        try:
            candidate_centroid = pd.DataFrame({'y' : [df['maxy'] - df['miny']/2],
                                               'x' : [df['maxx'] - df['minx']/2]})
            sp_centroids = readsql(f"SELECT * FROM sp_centroid WHERE imgname = '{imgName}' ORDER BY spid ASC")
            sp_centroids = np.array((sp_centroids['y'].values, sp_centroids['x'].values)).T
            sp_distr = readsql(f"SELECT * FROM sp_object_distribution WHERE imgname = '{imgName}' AND splitid = {splitid} AND nickname = '{hyperparams.pixnickname}' ORDER BY spid ASC")['vec']
            if len(sp_centroids) != len(sp_distr):
                print("Missing centroid data, continuing")
                return
            if (sp_distr.values.size == 0):
                print("Missing distr data, continuing")
                return
            # should add here, if already in the table, continue. This means I should make an index in the convenient place.
            sp_distr = [np.array(eval(distr)) for distr in sp_distr.values]
        except:
            return
        split_vec1 = split_field_vector(candidate_centroid, sp_centroids,sp_distr,imgshape,offset_candidates,angle_candidates,scales=[1])
        tsv_perfect.write(f"{imgName}\t{df['patchname']}\t[1]\tsplit\t{floatserial(split_vec1.tolist(),12)}\t{splitid}\t{dataset}\t{trial}\t{args.nickname}\n")
        print("successfully wrote perfect")

def inner(x):
    '''
    TODO - add code for doing this on a training set.
    '''
    t00 = time.time()
    imgName,splitid,dataset,trial,isperfect,split,hyperparams = x
    row = imgshapes[imgshapes['imgname'] == imgName + ".jpg"]
    if row.size == 0:
        row = imgshapes[imgshapes['imgname'] == imgName]
    if row.size == 0:
        return
    print(imgName)
    imgshape = (row['height'].values[0],row['width'].values[0])
    candidate_centroids = readsql(f"SELECT * FROM candidate_centroid WHERE imgname = '{imgName}'")
    if len(candidate_centroids) == 0:
        return
    for canid in range(max(candidate_centroids['canid'])):
        ground_truth = readsql(f"SELECT classname FROM ground_truth WHERE imgname = '{imgName}' AND canid = {canid}")
        if len(ground_truth) == 0: #exepcted to be there for some candidates and not others.
            continue
        elif ground_truth['classname'][0] == 'None':
            print(f"imgname={imgName},canid={canid} does not have any ground truth")
            continue
        ground_truth = ground_truth['classname'].values[0]
        isknown = int(ground_truth in split)
        insert_ifnotexists(f"SELECT * FROM isknown WHERE splitid = {splitid} AND imgname = '{imgName}' AND canid = {canid}",(f"INSERT INTO isknown VALUES ('{imgName}',{canid},{splitid},{isknown})"),hyperparams)
        try:
            candidate_centroid = candidate_centroids[candidate_centroids['canid'] == canid]
            sp_centroids = readsql(f"SELECT * FROM sp_centroid WHERE imgname = '{imgName}' ORDER BY spid ASC")
            sp_centroids = np.array((sp_centroids['y'].values, sp_centroids['x'].values)).T
            # previously there was a typo, so use of the term 'canid' below is intentional.
            sp_distr = readsql(f"SELECT * FROM sp_object_distribution WHERE imgname = '{imgName}' AND splitid = {splitid} AND nickname = '{args.nickname}' ORDER BY spid ASC")['vec']
            if len(sp_centroids) != len(sp_distr):
                print("Missing centroid data, continuing")
                return
            if (sp_distr.values.size == 0):
                print("Missing distr data, continuing")
                return
            sp_distr = [np.array(eval(distr)) for distr in sp_distr.values]
        except:
            return
        split_vec1 = split_field_vector(candidate_centroid, sp_centroids,sp_distr,imgshape,offset_candidates,angle_candidates,scales=[1])
        tsv.write(f'{imgName}\t{canid}\t[1]\tsplit\t{floatserial(split_vec1.tolist(),12)}\t{splitid}\t{dataset}\t{trial}\t{args.nickname}\n')
    print(f"full inner took {time.time() - t00} seconds")
    return True

def mkdense(args:argparse.Namespace,multiproc=True,perfect=True,just_pik=False):
    '''
    This iterates over the validation set because using context representations on the training set would be cheating.
    This takes 8ish hours.
    '''
    trial = 0
    db_check = readsql(f"SELECT * FROM sp_object_distribution WHERE nickname = '{hyperparams.pixnickname}' LIMIT 10",whichdb="postgres")
    ds = np.unique(db_check['dataset'])
    assert(len(ds) == 1)
    dataset = ds[0]
    cats = np.squeeze(readsql(f"select category FROM splitcats WHERE splitid = {args.splitid} AND dataset = '{dataset}' AND seen = 1").values)
    if dataset == 'COCO':
        vnames = hyperparams.val_names()
    elif dataset == 'pascal':
        vnames = hyperparams.val_names_gt()
    match = f" WHERE nickname = '{args.nickname}' AND splitid = {args.splitid}"
    perfect_ignore = set(readsql("SELECT imgname FROM perfect_densevecs" + match)['imgname'].values)
    norm_ignore = set(readsql("SELECT imgname FROM densevecs" + match)['imgname'].values)
    print(f"Skipping {len(perfect_ignore)} perfect imgnames and {len(norm_ignore)} normal imgnames, because they are already done")
    random.shuffle(vnames)
    perf_todo = list(set(vnames) - perfect_ignore)
    norm_todo = list(set(vnames) - norm_ignore)
    mapfn = perfect_inner if perfect else inner
    if multiproc:
        with multiprocessing.Pool(5) as p:
            p.map(mapfn,[(name,str(args.splitid),dataset,trial,int(perfect),cats) for name in perf_todo])
    else:
        for name in perf_todo:
            mapfn((name,str(args.splitid),dataset,trial,int(perfect),cats),just_pik=just_pik)
        if just_pik:
            for name in hyperparams.train_names():
                mapfn((name,str(args.splitid),dataset,trial,int(perfect),cats),just_pik=just_pik,imgdir="train_images/")

def dimensionality_reduction(nickname):
    # on the split dataset
    dosql("CREATE TABLE IF NOT EXISTS projvecs (nickname TEXT, imgname TEXT, canid INT, type TEXT, vector TEXT, splitid TEXT)")
    conn = sqlite3.connect(hyperparams.db)
    cursor = conn.cursor()
    Xsplit = readsql(f"SELECT vec FROM densevecs WHERE nickname = '{nickname}' AND type = 'split'")
    transformer = random_projection.SparseRandomProjection()
    t0 = time.time()
    # if this takes too long, 
    X_low = transformer.fit_transform(Xsplit)
    print(f"Fitting took {time.time() - t0} seconds")
    for i,xi in X_low:
        ins.append([nickname,imgnames[i],canids[i],'split',xi,splitid])
        if i % 50 == 0:
            cursor.executemany("INSERT INTO projvecs VALUES (%s,%s,%s,%s,%s,%s)")
            conn.commit()
    # on the xyd dataset
    X3d = readsql(" ")
    transformer = random_projection.SparseRandomProjection()
    X_low = transformer.fit_transform(X3d)

def classification_learnf(num_fields):
    for cid, heldoutcat in enumerate(cats):
        if heldoutcat not in hyperparams.possible_splits[splitid]['known']:
            continue
        if readsql(f"SELECT * FROM learnedsplits WHERE splitid = {splitid} AND category = '{heldoutcat}' AND field_type = 'grid'",hyperparams).values.size > 0:
            continue
        print(heldoutcat)
        print(f"working on grid,{heldoutcat} learned with splitid = {splitid}")
        max_drop, min_negative = 0.7, 2 #meaning drop at most 80% of negative samples, keeping at least 2 times as many negative examples as there are positive.
        dat = readsql(f'''SELECT R.vector, R.classname FROM (densevecs NATURAL JOIN ground_truth) R
                          WHERE densevecs.splitid = {splitid} AND densevecs.type = 'grid' '''
        Xgrid, y = np.array([np.fromstring(dat['R.vector'].values[i][1:-1],sep=",") for i in range(dat.values.shape[0])]), np.array([t == heldoutcat for t in dat['R.classname'].values],dtype=np.int)

        num_positive = np.sum(y)
        if num_positive == 0:
         continue
        deletion_amount = int(min(max_drop * y.size, y.size - min_negative * np.sum(y)))
        print("Before discarding {deletion_amount} negative examples, {num_positive} positive examples which is {np.sum(y) / y.size} of all examples")
        for_removal = random.sample(np.where(y == 0)[0].tolist(), deletion_amount) # remove such that at least as many negative as positive, and up to removing 4/5 of negative examples.
        y = np.delete(y,for_removal)
        y[y == 0] = -1 #make this change so that gradient descent can work.
        Xgrid = np.delete(Xgrid,for_removal,axis=0)
        query_str = lambda a,b: "INSERT INTO learnedsplits VALUES('{}','{}','{}',{},{},{},{},{})".format(heldoutcat, a, 'grid',splitid,len(a),1,b,np.sum(y))
        active_subsets = off_shelf_graft(Xgrid,y,query_str,num_fields=300)
        del(Xgrid);del(y);del(dat);
        continue
        dat = readsql('''SELECT R.vector, R.classname FROM (densevecs NATURAL JOIN ground_truth) R
                          WHERE densevecs.splitid = {} AND densevecs.type = 'split1' '''.format(splitid))
        Xsplit1, y = np.array([np.fromstring(dat['R.vector'].values[i][1:-1],sep=",") for i in range(dat.values.shape[0])]), np.array([t == heldoutcat for t in dat['R.classname'].values],dtype=np.int)
        num_positive = np.sum(y)
        deletion_amount = int(min(max_drop * y.size, y.size - min_negative * np.sum(y)))
        print("Before discarding {} negative examples, {} positive examples which is {} of all examples".format(deletion_amount,num_positive,np.sum(y) / y.size))
        for_removal = random.sample(np.where(y == 0)[0].tolist(), deletion_amount) # remove such that at least as many negative as positive, and up to removing 4/5 of negative examples.
        y = np.delete(y,for_removal)
        y[y == 0] = -1 #make this change so that gradient descent can work.
        Xsplit1 = np.delete(Xsplit1,for_removal,axis=0)
        query_str = lambda a,b: "INSERT INTO learnedfields VALUES('{}','{}','{}',{},{},{},{},{})".format(heldoutcat, a, 'split1',splitid,len(a),1,b,np.sum(y))
        off_shelf_graft(Xsplit1,y,query_str,num_fields=500)
        del(Xsplit1);del(y);del(dat);

def create_tables( ):
    dosql("CREATE TABLE IF NOT EXISTS greedy_variances(nickname TEXT, numfields INT, isdense TEXT, variance FLOAT, mean FLOAT)",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS greedy_loss (nickname TEXT, numfields INT, type TEXT, val FLOAT, nsamples INT)",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS greedy_nicknames (nickname TEXT, field_type TEXT, splitid TEXT, include_entropy INT, reg_lambda FLOAT, entropy_const FLOAT, PRIMARY KEY(nickname))",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS greedy_trialinfo (pixnickname TEXT, nickname TEXT, trial INT, sample_proportion FLOAT, PRIMARY KEY(nickname,trial))",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS learnedfields (nickname TEXT, trial INT, category TEXT, active_subset TEXT, num_fields INT, error FLOAT ,num_affirmative INT, FOREIGN KEY (nickname) REFERENCES greedy_nicknames(nickname))",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS metrics (nickname TEXT, trial INT, active_subset TEXT, num_fields INT, nondiag FLOAT, rowstd FLOAT, metric TEXT, select_stepsize FLOAT, step_stepsize FLOAT, FOREIGN KEY (nickname) REFERENCES greedy_nicknames(nickname))",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS subset_correspondence(nickname TEXT, trial INT, fullidx INT, subidx INT, type TEXT, PRIMARY KEY(nickname,trial,fullidx))",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS grad_variance(nickname TEXT, trial INT, type TEXT, num_fields INT, timestep INT, variance FLOAT)",hyperparams)

def roc( ):
    pass

def memlimit(query:str,hyperparams,samplesize=500) -> int:
    rowsize = sys.getsizeof(readsql(query + f" LIMIT {samplesize}",hyperparams)) / samplesize
    avail = psutil.virtual_memory().available
    maxfit = int(math.floor(args.prop_memory * avail / rowsize))
    print(f"About to fit {maxfit} into virtual memory")
    return maxfit

def learn_metric(hyperparams,args):
    '''
    A pretext task.
    What is the way to correlate which receptive fields are useful?
    '''
    assert(not(hyperparams.field_t == 'mix' and hyperparams.include_entropy)) # how to compute entropy with multiple types of params? not yet defined.
    nclasses = len(hyperparams.possible_splits[hyperparams.splitid]['known']) + 1
    create_tables()
    if os.path.exists(hyperparams.root("results/split_learned.pkl")):
        results = pickle.load(open(hyperparams.root("results/split_learned.pkl"),'rb'))
    else:
        results = pd.DataFrame(columns=['category','active_subset','field_type','splitid','num_fields','error','num_affirmative','reg-lambda'])
    knowncats = np.squeeze(readsql(f"SELECT category FROM splitcats WHERE dataset = '{hyperparams.dataset}' AND splitid = {hyperparams.splitid} AND seen = 1",hyperparams).values)
    unknowncats = np.squeeze(readsql("SELECT category FROM splitcats WHERE dataset = '{}' AND splitid = {} AND seen = 0".format(hyperparams.dataset,hyperparams.splitid),hyperparams).values)
    entropy_literal = 'null' if hyperparams.entropy_const is None else hyperparams.entropy_const
    insert_ifnotexists("SELECT * FROM greedy_nicknames WHERE nickname = '{}'".format(args.nickname),"INSERT INTO greedy_nicknames VALUES ('{}','{}',{},{},{},{})".format(args.nickname,hyperparams.field_t,hyperparams.splitid,int(hyperparams.include_entropy),hyperparams.reg_lambda,entropy_literal),hyperparams)
    if not args.from_scratch:
        trial = args.trial #assume it's specified.
        sofar = readsql('''SELECT num_fields,active_subset,metric FROM metrics WHERE nickname = '{0}' AND trial = {1} AND
                           num_fields = (SELECT max(num_fields) FROM metrics WHERE nickname = '{0}')'''.format(args.nickname,trial),hyperparams)
    else:
        sofar = []
        if args.trial < 0:
            trial = readsql("SELECT max(trial) as maxtrial FROM metrics WHERE nickname = '{}'".format(args.nickname),hyperparams)['maxtrial'].ix[0]
            if args.trial is None:
                trial = 0
            else:
                trial = trial + 1
        else:
            trial = args.trial
        #insert_ifnotexists("SELECT * FROM greedy_trialinfo WHERE nickname = '{}' AND trial = {} AND sample_proportion = {}".format(nickname,trial,subset_proportion),
                           #"INSERT INTO greedy_trialinfo VALUES ('{}',{},{})".format(nickname,trial,subset_proportion))
    cats = datasets.coco_allcats
    print("working on splitid={},trial={},lamba={},field_type={}".format(hyperparams.splitid,args.trial,hyperparams.reg_lambda,hyperparams.field_t))
    if len(sofar) == 0 or sofar['num_fields'].values[0] < args.num_fields:
        #X, y = np.array([eval('[' + dat['R.vector'].values[i] + ']') for i in range(dat.values.shape[0])]), dat['R.classname'].values
        fname = hyperparams.root('cache/dense_{}.npy'.format(args.nickname))
        if not os.path.exists(fname):
            if hyperparams.field_t == "mix": #don't select by type, get a mixture instead.
                dat_grid = readsql('''SELECT R.vector, R.classname, R.type FROM (densevecs NATURAL JOIN ground_truth) R
                                 WHERE densevecs.splitid = {} AND densevecs.type = 'grid' ORDER BY imgname ASC, canid ASC'''.format(hyperparams.splitid),hyperparams)
                dat_split = readsql('''SELECT R.vector, R.classname, R.type FROM (densevecs NATURAL JOIN ground_truth) R
                                 WHERE densevecs.splitid = {} AND densevecs.type = 'split' ORDER BY imgname ASC, canid ASC'''.format(hyperparams.splitid),hyperparams)
                X_grid, y_grid = np.array(np.fromstring(dat['R.vector'].values[i],sep=",") for i in range(dat_grid.values.shape[0])), dat_grid['R.classname'].values.astype(np.str)
                X_split, y_split = np.array(np.fromstring(dat['R.vector'].values[i],sep=",") for i in range(dat_grid.values.shape[0])), dat_split['R.classname'].values.astype(np.str)
                mix_sep = X_grid.shape[0]
                assert(np.array_equal(y_grid,y_split))
                X, y = np.hstack((X_grid,X_split)), y_grid
            else:
                if args.perfect:
                    qr = "SELECT R.vec, R.category AS classname, R.type FROM (perfect_densevecs NATURAL JOIN perfect_isknown) R WHERE R.seen = 1 AND R.nickname = '{}' AND R.splitid = {} AND R.type = '{}' ORDER BY imgname ASC, patchname ASC".format(hyperparams.datasrc_nickname,hyperparams.splitid,hyperparams.field_t)
                    limit = memlimit(qr,hyperparams)
                    known_dat = readsql(qr + " LIMIT " + str(limit),hyperparams)
                else:
                    dat = readsql("SELECT R.vec, R.classname, R.type FROM (densevecs NATURAL JOIN ground_truth) R WHERE R.nickname = '{}' AND R.splitid = {} AND R.type = '{}' ORDER BY imgname ASC, canid ASC".format(hyperparams.datasrc_nickname,hyperparams.splitid,field_t))
                    known_dat = dat[dat['classname'].isin(knowncats)]
                    del(dat)
                assert(len(known_dat) > 0), f"No data for splitid={splitid} yet"
        #        unknown_dat = dat[dat['classname'].isin(unknowncats)]
                X, y = np.array([np.fromstring(known_dat['vec'].values[i],sep=",") for i in range(known_dat.values.shape[0])]).astype(np.float32), known_dat['classname'].values.astype(np.str)
                del(known_dat)
                gc.collect()
                mix_sep = None
            dd.io.save(fname,(X,y,mix_sep))
        else:
            X,y,mix_sep = dd.io.load(fname) #if not using field_t == "mix", mix_sep is None.
        param_savefile = hyperparams.root(f'cache/field_param_{args.nickname}')
        # use the min function because if everything fits in memory, then we are outside the domain where we need to think about proportions.
        subset_proportion = min(args.max_in_memory / (nclasses * args.densenum),1)
        print(f"In order to fit everything in memory, taking {subset_proportion} of possible receptive fields")
        if not args.from_scratch:
            subset = readsql(f"SELECT fullidx FROM subset_correspondence WHERE nickname = '{args.nickname}' AND trial = {args.trial} ORDER BY subidx ASC",hyperparams)['fullidx'].values
            field_params = pickle.load(open(param_savefile,'rb'))
            print("Loaded subset and field_params")
            assert(subset.size > 0)
        else:
            subset = []
            print(f"Making new subset as trial {trial}")
            if hyperparams.field_t == "mix":
                amount_scale = mix_sep / X.shape[0]
                grid_scale = hyperparams.prop_grid / (mix_sep / X.shape[1])
                split_scale = (1 - hyperparams.prop_grid) / (X.shape[1] - mix_sep / X.shape[1])
                thresh = lambda i: subset_proportion * (grid_scale / (grid_scale + split_scale) if i < mix_sep else (split_scale / (grid_scale + split_scale)))
            else:
                thresh = lambda i: subset_proportion
            if hyperparams.field_t == "split":
                fparams = readsql(f"SELECT id,offset_y,offset_x,angle FROM split_field_candidates WHERE nickname = '{hyperparams.datasrc_nickname}'",hyperparams)
            elif hyperparams.field_t == "grid":
                fparams = readsql(f"SELECT id,miny,maxy,minx,maxx FROM grid_field_candidates WHERE nickname = '{hyperparams.datasrc_nickname}'",hyperparams)
            field_params = []
            for i in range(X.shape[1]):
                p = fparams[fparams['id'] == (i//nclasses)]
                assert(len(p) == 1)
                del(p['id']) #so that .values does not get redundant info.
                if random.random() < thresh(i):
                    if hyperparams.field_t == "mix":
                        field_type = 'grid' if i < mix_sep else 'split'
                    else:
                        field_type = hyperparams.field_t
                    dosql("INSERT INTO subset_correspondence VALUES ('{}',{},{},{},'{}')".format(args.nickname,args.trial,i,len(subset),field_type),hyperparams)
                    subset.append(i)
                    field_params.append(np.squeeze(p.values.astype(np.float32)))
            print("Using subset size = {}".format(len(subset)))
            field_params = np.array(field_params)
            pickle.dump(field_params,open(param_savefile,'wb'))
        X = X[:,subset]
        print("Selected {}-column subset of X with dtype {}".format(len(subset),X.dtype))
        sample = sample_wrap(X,y,hyperparams.splitid,args.trial,args.nickname)
        query_str = lambda a,b,c,d: "INSERT INTO metrics VALUES('{}',{},'{}',{},{},{},'{}',{},{})".format(args.nickname,trial,repr(a),len(a),(np.sum(np.abs(b)) - np.trace(np.abs(b))) / np.sum(np.abs(b)),np.std(np.sum(np.abs(b),axis=1)),repr(b),c,d)
        if len(sofar) != 0:
            A0, subset0 = np.array(eval(sofar['metric'].values[0]),dtype=np.float32), eval(sofar['active_subset'].values[0])
        else:
            A0, subset0 = np.array([[1,0],[0,1]]).astype(np.float32), [1,10]
        active_subsets = metric_graft(hyperparams,X,y,query_str,A0,subset0,args.num_fields,hyperparams.reg_lambda,args.num_select_constraints,args.num_step_constraints,args.nickname,field_params,sample,hyperparams.entropy_const,include_entropy=hyperparams.include_entropy)
        results.to_pickle(hyperparams.root("results/split_learned_{}.pkl".format(nickname)))
    else:
        print("This combination of parameters is done training")

def sample_wrap(X,y,splitid:int,trial:int,nickname:str,ndistinct=8,k_top=1,class_balance_scale=2.0):
    '''
    Returns (m x 4) matrix of ints
        (a,b,c,d) indices into X, such that d(X[a],X[b]) < d(X[c],X[d])
    d(X[a],X[b]) < d(X[c],X[d]) is true if a and b are of the same class and c and d are of different classes.
    This is what is expected by metric_learn.
    '''
    # pre-allocate memory because this was slow with python list memory management.
    bufs,bufsize = [],100000000
    if len(y) < 2 ** 16:
        min_t = np.uint16
        print("Using 16 bit numbers for constraints because it fits")
    else:
        min_t = np.uint32
        print("Using 32 bit numbers for constraints because we have big data")
    num_constraints,cycle = 0,0
    # take an equal number of positive and negative constraints.
    fname = hyperparams.root('cache/constraints_{}_{}_{}.npy'.format(splitid,trial,nickname))
    final_fname = hyperparams.root('cache/sample_constraints_{}_{}_{}.npy'.format(splitid,trial,nickname))
    if not os.path.exists(fname):
        buf = -1 * np.ones((bufsize,4),dtype=min_t)
        print("Beginning to gather constraints.")
        for i,ci in enumerate(y):
            if i % 500 == 499:
                print(f"Got {num_constraints} constraints from {i} iterations")
            ispos = np.where(y == ci)[0]
            ispos = ispos[ispos > i] #don't look back, for sake of symmetry.
            isneg =  np.where(y != ci)[0].tolist()
            newnum = min(len(ispos),len(isneg))
            isneg = np.array(random.sample(isneg,newnum))
            ispos = np.array(random.sample(ispos.tolist(),newnum))
            num_constraints += newnum
            if cycle + newnum < bufsize:
                buf[cycle:cycle+newnum,0] = i
                buf[cycle:cycle+newnum,1] = ispos
                buf[cycle:cycle+newnum,2] = i
                buf[cycle:cycle+newnum,3] = isneg
            else:
                print("Previous buffer filled up, making new buffer.")
                buf = buf[0:cycle]
                bufs.append(buf)
                buf = -1 * np.ones((bufsize,4))
                cycle = 0
            cycle += newnum
        if cycle < bufsize:
            buf = buf[0:cycle]
        bufs.append(buf)
        constraints = np.vstack(bufs)
        del(buf)
        del(bufs)
        np.save(fname,constraints)
    else:
        if os.path.exists(fname) and not os.path.exists(final_fname):
            print("Loading pre-saved constraints, continuing by making sampled versions.")
            constraints = np.load(fname)
    # make a few class-balanced versions.
    if not os.path.exists(final_fname):
        frequencies = collections.Counter(y)
        lf = list(frequencies.values())
        lf.sort(reverse=True)
        avg_freq = np.mean(lf)
        top_k_thresh = lf[k_top]
        constraintsT = constraints.T
        sample_choices,rms = {},{}
        for c in np.unique(y):
            indexes = np.where(y == c)[0]
            class_share = class_balance_scale * (indexes.size / len(y))
            if indexes.size >= top_k_thresh:
                guaranteed = (indexes.size - lf[k_top+1]) / lf[k_top+1]
                limits = [guaranteed,class_balance_scale * class_share,0.9]
                rm = min(limits) # make removal amount proportional to proportion of gt.
                rms[c] = rm
                print(f"Removing class {c} with proportion {rm} (active limit being {p.argmin(limits)})")
            else:
                rms[c] = 0
        for k in range(ndistinct):
            discard_rows = np.zeros(constraints.shape[0],dtype=np.bool)
            rands = np.random.uniform(0,1,size=constraints.shape[0])
            for i,con in enumerate(constraints):
                if i % 10000 == 0:
                    print(f"Removed {np.sum(discard_rows)} so far")
                rm_max = max([rms[y[con_i]] for con_i in con])
                if rm_max > rands[i]:
                    discard_rows[i] = True
            sample_choices[k] = np.delete(constraints,np.where(discard_rows == True)[0],axis=0)
            print(f"Made the {k}th randomly generated, class balanced dataset to sample from")
        del(constraints)
        del(constraintsT)
        np.save(final_fname,sample_choices)
    else:
        sample_choices = np.load(final_fname).item()
    def sample(sample_prop=0.1,Xshape=None,take_amount=None):
        which = random.choice(list(range(ndistinct)))
        constraints = sample_choices[which]
        if Xshape != None:
            assert(np.min(constraints) >= 0)
            assert(np.max(constraints) < Xshape[0])
        if take_amount == None:
            take_amount = int(len(constraints) * sample_prop)
        take_amount = min(take_amount,len(constraints))
        which_rows = np.array(random.sample(list(range(len(constraints))),take_amount))
        return constraints[which_rows].astype(min_t)
    return sample

# I should make candidate_data a global variable.
def index_to_params(index,splitid,field_type:str):
    '''
    I lost the grid information because I made it too big and had to randomly trim.
    Pretty sure this is how to recover the info.
    '''
    if field_type == 'split1':
        num_scales = 1
    elif field_type == 'split3':
        num_scales = 3
    structured_shape =  (2 * candidate_data.shape[0], num_scales, len(datasets.coco[splitid]['known']))
    onehot = np.zeros(structured_shape[0] * structured_shape[1] * structured_shape[2])
    onehot[index] = 1
    row_placement, scale, objclass = np.squeeze(np.where(onehot.reshape(structured_shape) == 1))
    return candidate_data[row_placement / 2], objclass
    
def information_gains(X,active_subset,candidate_data,field_params) -> np.ndarray:
    '''
    field_params is of shape (num_param_types,num_fields). e.g. (angle,offset_x,offset_y) -> num_param_types = 3.

    I think I need to transform the current counts of parameters into density estimates.
    '''
    kd = KernelDensity()
    kd.fit(field_params[active_subset])
    scores = np.exp(kd.score_samples(field_params))
    scores[active_subset] = 1
    return (1 - scores) #low probability makes more likely to be chosen.

def metric_selection(hyperparams,X,y,A_active,active_subset,reg_lambda,num_constraints,field_params,sample,entropy_const,candidate_data=None,batchsize=8,include_entropy=False,etnropy_const=0.16,minstep=1e-5):
    '''
    
    '''
    global best_stepsize
    lsml = LSML(regularization_lambda=reg_lambda,max_iter=1,chatty=True,minstep=minstep)
    constraints = sample(take_amount=num_constraints,Xshape=X.shape)
    A = np.eye(X.shape[1])
    for i, active_field_i in enumerate(active_subset):
        for j, active_field_j in enumerate(active_subset):
            A[active_field_i,active_field_j] = A_active[i,j]
    t0 = time.time()
    stepsize,num_violations,grads = lsml.fit(X,constraints,int(0.5 * num_constraints),warm_start=A,verbose=True,num_steps=2,l_prev=best_stepsize['select'],wall_timeout=600,prior="identity",skip_linesearch=True,poisson_k=0.1,skip_projection=True)
    best_stepsize['select'] = stepsize
    grads = np.array(grads)
    if grads.shape[0] > 1:
        coord_variances = np.var(grads,axis=0)
        dosql(f"INSERT INTO greedy_variances VALUES('{args.nickname}',{len(active_subset)},'full',{np.mean(coord_variances)},{np.mean(np.abs(grads))})",hyperparams)
    criterion = np.sum(np.abs(lsml.M),axis=0)
    print(f"Taking grad with {num_constraints} constraints took {time.time() - t0} seconds")
    if include_entropy:
        gains = information_gains(X,active_subset,candidate_data,field_params)
        update = (entropy_const * np.ptp(criterion) / np.ptp(gains)) * gains
        criterion = criterion + update
        print("Distance between entropy-adjusted and ordinary over norm of updated=",np.linalg.norm(criterion-update)/np.linalg.norm(criterion))
    max_fields = []
    criterion[active_subset] = np.finfo(np.float32).min #minimum possible floating point value.
    for b in range(batchsize):
        current = np.argmax(criterion)
        max_fields.append(current)
        criterion[current] = np.finfo(np.float32).min #minimum possible floating point value, so I can get the next max for the current batch.
    print(f"There are now {len(active_subset)} learned receptive fields, having chosen {max_fields}")
    return max_fields

def test(hyperparams,lsml,sample,X_active,active_subset):
    N = args.test_per_field
    constraints = sample(take_amount=N)
    comparison_loss,regularization_loss,num_violations = lsml.score(X_active,constraints,lsml.M)
    prefix = f"INSERT INTO greedy_loss VALUES ('{args.nickname}',{len(active_subset)},"
    dosql(prefix + f"'comparison',{comparison_loss / N},N)",hyperparams)
    dosql(prefix + f"'regularization',{regularization_loss / N},N)",hyperparams)
    dosql(prefix + f"'num_violations',{num_violations / N},N)",hyperparams)

def metric_step(hyperparams,A_old:np.ndarray,X:np.ndarray,y:np.ndarray,active_subset:np.ndarray,selected,reg_lambda:float,sample,num_step_constraints:int,minstep=1e-5) -> np.ndarray:
    '''
    newidx is an active_subset index, not a global index.

    todo - collect variance information here.
    '''
    global best_stepsize
    newidx = np.where(active_subset == selected)[0][0]
    A = np.zeros((active_subset.size,active_subset.size),dtype=np.float32)
    A[newidx,newidx] = 1
    # handle the continuous block cases separately for sake of index bounds.
    if selected == min(active_subset):
        A[1:,1:] = A_old
    elif selected == max(active_subset):
        A[0:A_old.shape[0],0:A_old.shape[1]] = A_old
    else: # have to assign the 4 blocks separately to make room for the new zero row/col with a one at the diagonal.
        upper_left = A_old[0:newidx,0:newidx]
        if upper_left.size > 0:
            A[0:newidx,0:newidx] = upper_left
        upper_right = A_old[0:newidx,newidx:A.shape[1]]
        if upper_right.size > 0:
            A[0:newidx,newidx+1:A.shape[1]] = upper_right
        lower_left = A_old[newidx:A.shape[0],0:newidx]
        if lower_left.size > 0:
            A[newidx+1:A.shape[0],0:newidx] = lower_left
        lower_right = A_old[newidx:A.shape[0],newidx:A.shape[1]]
        if lower_right.size > 0:
            A[newidx+1:A.shape[0],newidx+1:A.shape[1]] = lower_right
    lsml = LSML(tol=8e-7,regularization_lambda=reg_lambda,max_iter=1000,chatty=False,minstep=minstep) #lower than the default tolerance b/c learning this is quick
    X_active = X[:,active_subset]
    constraints = sample(take_amount=num_step_constraints,Xshape=X.shape)
    stepsize,num_violations,grads = lsml.fit(X_active,constraints,num_step_constraints,warm_start=A,verbose=False,l_prev=best_stepsize['step'],num_steps=80,wall_timeout=500,poisson_k=0.5)
    best_stepsize['step'] = stepsize
    print(f"{abs(num_violations - num_step_constraints/2)} better than random with {num_step_constraints} constraints")
    grads = np.array(grads)
    if (grads.shape[0] > 1):
        coord_variances = np.var(grads,axis=0)
        dosql(f"INSERT INTO greedy_variances VALUES('{args.nickname}',{len(active_subset)},'active',{np.mean(coord_variances)},{np.mean(np.abs(grads))})",hyperparams)
    test(hyperparams,lsml,sample,X_active,active_subset)
    return(lsml.M)

def metric_graft(hyperparams,X,y,querystr,A0,subset0,num_fields,reg_lambda,num_select_constraints,num_step_constraints,nickname,field_params,sample,entropy_const,include_entropy=False,candidate_data=None):
    '''
    Rather than doing such a step based thing, I might want to do one long fit, and then use the new absolute value column sum criteron.
    '''
    A = A0
    active_subset = subset0
    batchsize = 3
    for t in range(int(num_fields / batchsize)):
        if len(active_subset) >= num_fields:
            break
        selection = metric_selection(hyperparams,X,y,A,active_subset,reg_lambda,num_select_constraints,field_params,sample,entropy_const,batchsize=batchsize,include_entropy=include_entropy,candidate_data=candidate_data)
        for selected in selection:
            active_subset.append(selected)
            active_subset.sort()
            assert(len(active_subset) == len(np.unique(active_subset))) # duplicates should not be possible; if this gets triggered there is a bug.
            A = metric_step(hyperparams,A,X,y,np.array(active_subset),selected,reg_lambda,sample,num_step_constraints)
        try:
            dosql(querystr(active_subset,A,best_stepsize['select'],best_stepsize['step']),hyperparams)
            print("Saved to sql")
        except:
            print("*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!")
            print("WARNING: DATABASE SAVING FAILED, PICKLING AND CONTINUING")
            print("*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!")
            print("The failed statement is",querystr(active_subset,A,best_stepsize['select'],best_stepsize['step']))
            pickle.dump((A,active_subset),open(hyperparams.root(f'cache/As_{t}'),'wb'))
            print("Saved to pickle (remember to put it in db later)")
    return A

def visfields(nickname:str,splitid:int,hyperparams):
    csv = open('rfield_data.csv','w')
    nickname = 'test'
    num_fields = 86 + 90
    classes = readsql(f"SELECT category FROM splitcats WHERE dataset = 'COCO' AND splitid = {splitid} AND seen = 1",hyperparams)['category'].values
    num_classes = classes.size + 1
    params = readsql(f"SELECT * FROM split_field_candidates WHERE nickname = '{nickname}'",hyperparams) #pixnickname
    which = readsql(f"SELECT * FROM metrics WHERE num_fields = {num_fields} AND nickname = '{nickname}'",hyperparams)
    metric = eval(which['metric'].ix[0])
    corresp = readsql(f"SELECT * FROM subset_correspondence WHERE nickname = '{nickname}'"
    subset = eval(which['active_subset'].ix[0])
    csv.write('angle,offset_y,offset_x,category,magnitude,id,nickname\n')
    for i,field in enumerate(subset):
        magnitude = np.sum(np.abs(metric[i]))
        full_id = corresp[corresp['subidx'] == field]['fullidx'].values[0]
        if full_id % num_classes == classes.size:
            category = 'None'
        else:
            category = classes[full_id % num_classes]
        p = params[params['id'] == (full_id // num_classes)]
        csv.write(f"{p['angle'].values[0]},{p['offset_y'].values[0]},{p['offset_x'].values[0]},{category},{magnitude},{full_id},{nickname}\n")
    
if args.action == "generate_fields":
    if not os.path.exists(hyperparams.root("fields")):
        subprocess.call(["mkdir",hyperparams.root("fields")])
    #np.savetxt(params.root('fields/grid_field_candidates.npy'), grid_field_candidates(args.densenum))
    offsets, angles = split_field_candidates(args.densenum,args,hyperparams)
    np.savetxt(hyperparams.root(f'fields/split_offset_candidates_{hyperparams.pixnickname}.npy'),offsets)
    np.savetxt(hyperparams.root(f'fields/split_angle_candidates_{hyperparams.pixnickname}.npy'),angles)
elif args.action == "mkdense":
    mkdense(hyperparams,args)
elif args.action == "learn_metric":
    learn_metric(hyperparams,args)
elif args.action == "learn_classify":
    classify_learnf(hyperparams,args)
elif args.action == "visualize":
    visfields(args.nickname,args.splitid,hyperparams)
elif args.action == "superpix":
    dataproc.mksuperpixels(args.dataset,args.splitid,args.numfuse,args.trial,hyperparams.pixnickname,tstep=args.tstep)
else:
    print("Unknown action")
