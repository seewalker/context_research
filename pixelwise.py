'''
Alex Seewald 2016
aseewald@indiana.edu

A tensorflow implementation of Active Learning for Fine-Grained Localization
This contains an implementation of Fully Convolutional Networks For Semantic Segmentation.

The decision to store individual ground truth pixels in a relational database is in retrospect not the best space/time tradeoff,
but if you can spend like 800 GB and you make an index on the table, it's actually pretty reasonable.

The "_v" suffix means its a value resulting from evaluating the tensorflow variable of the same name without that suffix.
'''
import random
import sys
import pickle
import subprocess
import time
import itertools
import sqlite3
import os
import math
import signal
import argparse
import re
import numpy as np
import matplotlib as mpl
import multiprocessing as mp
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import deepdish as dd
import tensorflow as tf
import scipy.stats
from typing import List,Tuple,Optional,Callable,Dict
from functools import reduce
from scipy.misc import imresize
from skimage.transform import resize
from line_profiler import *
from utils import *
import datasets
import hyperparams as hp
import warnings
warnings.filterwarnings('ignore')
from mytypes import *

# Making this global to keep it the same for all visualization calls.
__author__ = "Alex Seewald"

sns.set(style="whitegrid")

def confusion(nickname:str,enforce_square=True):
    '''
    Visualization of confusion matrix.
    '''
    # put an index on 't' because it is high-cardinality and speeds this up.
    hyperparams = hp.pixelwise_hp[nickname]
    ts = readsql("select distinct(t) from fullyconv_cls",hyperparams)
    for t in ts['t'].values:
        print("making graph with t =",t)
        X = readsql(f"SELECT realcat,predcat,prob FROM fullyconv_cls WHERE nickname = '{nickname}' AND t = {t}",hyperparams)
        epsilon = X['prob'].min() / 1e4 # eliminate possibility of all-zero rows (which can't be normalized) 
        X = pd.pivot_table(X,index='realcat',columns='predcat',values='prob',fill_value=epsilon) # make like a table for heatmap, replacing nan with zero.
        if enforce_square:
            shared = set(X.columns) & set(X.index) #intersection.
            X = X.drop(set(X.index) - shared)
            for cat in set(X.columns) - shared:
                del(X[cat])
        X = X.div(X.sum(axis=1),axis=0)
        # find predcats which are not existing, and add them with all-zeros.
        sns.heatmap(X)
        plt.yticks(rotation=0)
        maybe_mkdir('results/confusion')
        oname = f'results/confusion/{nickname}_{t}.png'
        print("saving to ",oname)
        plt.savefig(oname)
        plt.show()
        plt.close()

def corr_vis(hyperparams,nickname):
    num_tstep_bins = 4
    Y = readsql(f"SELECT * FROM dbg_cat_active_obs WHERE nickname = '{nickname}' order by tstep",hyperparams)
    X = readsql(f"SELECT * FROM dbg_size_active_corr WHERE nickname = '{nickname}' order by tstep",hyperparams)
    Y['category'] = Y['category'].apply(lambda x:x.replace('"',''))
    cat_tsteps = readsql(f"select tstep from dbg_cat_active_obs WHERE nickname = '{nickname}' order by tstep",hyperparams)
    coco = readsql("SELECT pixcount,category FROM pixcounts WHERE dataset = 'COCO'",hyperparams)
    seps = np.zeros(num_tstep_bins)
    num_obs = len(cat_tsteps)
    for i in range(num_tstep_bins):
        seps[i] = cat_tsteps.iloc[int(float(i)/num_tstep_bins * num_obs)]
    dfs = []
    for i in range(num_tstep_bins-1):
        df = Y[Y['tstep']>=seps[i]]
        dfs.append(df[df['tstep'] < seps[i+1]])
    fig,axes=plt.subplots(ncols=len(dfs))
    fig.suptitle("Corr(M,Class Pixel Area)")
    for i,df in enumerate(dfs):
        dfp = df.groupby('category').mean()
        dfp['category'] = dfp.index
        joined = dfp.merge(coco,on='category',how='left')
        dfp = joined.sort_values(by='m')
        axes[i].scatter(np.log(joined['pixcount']),joined['m'])
        cor,p = scipy.stats.pearsonr(joined['pixcount'],joined['m'])
        axes[i].set_title('corr = ('+str(round(cor,3)) + ',' + str(round(p,3)) + ')')
    plt.savefig('cat_corr.png')
    plt.close()
    Xs = X.groupby('tstep').mean()
    Xs['tstep'] = Xs.index
    plt.scatter(Xs['tstep'],Xs['m'])  #m is a mislabeling.
    plt.gca().set_title('corr = ('+str(round(cor,3)) + ',' + str(round(p,3)) + ')')
    cor,p = scipy.stats.pearsonr(Xs['tstep'],Xs['m'])
    plt.suptitle("Corr(M,Object Size) vs Training Timestep")
    plt.savefig("size_corr.png")
    
def joint_accuracy(hyperparams,nickname):
    X = readsql(f"SELECT * FROM fullyconv_joint WHERE nickname = '{nickname}'",hyperparams)
    for phase,df in X.groupby("phase"):
        df = df.sort_values('t')
        plt.suptitle('phase='+phase)
        plt.plot(df['t'],df['posaccuracy'],c='r') 
        plt.plot(df['t'],df['negaccuracy'],c='b') 
        plt.plot(df['t'],df['metaaccuracy'],c='k')
        plt.savefig(phase + ".png")
        plt.close()

def accuracy():
    X = readsql("SELECT * from fullyconv",hyperparams)
    Y = readsql("SELECT * from fullyconv",hyperparams)
    X['accuracy'] = X['posaccuracy']
    X['foreground?'] = True
    Y['accuracy'] = Y['negaccuracy']
    Y['foreground?'] = False
    Z = pd.concat([X,Y])
    del(Z['posaccuracy'])
    del(Z['negaccuracy'])
    g = sns.FacetGrid(Z,col="nickname",hue="foreground?",legend_out=False)
    g.map(sns.pointplot,"t","accuracy",scale=0.5)
    g.fig.get_axes()[0].legend(loc='upper left')
    plt.savefig('accuracyfc.png')
    plt.show()
    
def allvis( ):
    # visualizing confusion matrices.
    maybe_mkdir('results/confusion')
    fname = '{sess_dir}/fcdistinct.hdf'
    if not os.path.exists(fname):
        whichc = readsql("SELECT distinct(nickname,t) FROM fullyconv_cls",hyperparams)
        whichc.to_hdf(fname,'root')
    else:
        whichc = pickle.load(open(fname,'rb'))
    for _,row in whichc.iterrows(): # the distinct function ended up storing this as text, so i have to parse to unpack.
        nickname,t = whichc.iloc[0]['row'][1:-1].split(',')
        confusion(nickname,int(t))
    accuracy()

def signal_handler(signal,frame):
    print("Exiting prematurely, remember to manually load tsv data, like the confusion data")
    sys.exit(0)

def nearest_round(vals:np.ndarray,domain:int) -> np.ndarray:
    for i in range(vals.shape[0]):
        vals[i] = domain[np.argmin(np.abs(domain - vals[i]))]
        #for j in range(vals.shape[1]):
            #vals[i,j] = domain[np.argmin(np.abs(domain - vals[i,j]))]
    return vals

def read_dense_outer(q,hyperparams,allnames:List[str],batchsize:int,num_classes:int,split:pd.DataFrame,splitid=None,all_data_avail=False,dataset="COCO",anticipate_missing=False,qmax=20,saving=True,loading=True,cache_t="postproc",synchronous=False,chatty=False,img_s=224) -> Callable:
    '''
    q - a queue to put the data once we've got it.
    qmax - if over this number of tuples in the queue, stop inserting new stuff and wait.
    allnames - list of imagenames to randomly sample from.
    batchsize - number of data to ultimately produce.
    split - pandas dataframe expressing known and unknown classes. 
    synchronous - if False, run indefinitely and put the output tuples onto the queue. if False, run once and return with the output tuple.
    partition - pool|extra|all
    num_classes - depending on dataset,split, and whether to include none.

    Outputs tuples : (imgout,outgt,prop_gt_bg,names)
    where imgout is the image, outgt is the pixelwise ground truth, prop_gt is the proportion of pixels whose class label is not None.

    Reading this function, you may notice that pixel ground truth are stored individually per pixel in a relational database, which may seem wasteful.
    There is indeed a sacrifice of space and I/O time for doing things a simple way, but I can spend a few hundred gigabytes and indexes on the tables
    make things pretty fast anyway.

    if none is included:
        intcat in (0,len(split['known'])) are real classes.  intcat = len(split['known'])+1 is the 'none' class.
    '''
    try:
        if os.path.exists(misname):
            missing = np.squeeze(pd.read_csv(misname).values)
            missing = missing if missing.shape != () else [] #because squeeze makes things 0d sometimes.
        else: missing = []
    except: missing = []
    if dataset == 'COCO':
        missing = [os.path.join(hyperparams.root('train_images'),decode_imgname(m)+".jpg") for m in missing]
        update = list(set(allnames) - set(missing))
        allnames = update
    else:
        allnames = list(set(allnames) - set(missing))
    is_pool = np.random.binomial(1,0.5,len(allnames)).astype(np.bool)
    pool_names = list(np.array(allnames)[is_pool])
    is_extra = np.logical_not(is_pool)
    extra_names = list(np.array(allnames)[is_extra])
    # now, partition allnames into pool_names and extra_names for the sake of active learning.
    def batch_names( ):
        if partition == 'all':
            return random.sample(allnames,batchsize)
        elif partition == 'pool':
            return random.sample(pool_names,batchsize)
        elif partition == 'extra':
            return random.sample(extra_names,batchsize)
        else:
            assert(False),f"unknown partition = {partition}"
    def from_partial_sql( ):
        mt0 = time.time()
        existing_names = np.unique(data['imgname'])
        names = fastlist + existing_names.tolist() #names gets re-assigned according to what is there.
        numres = existing_names.size
        ishit = [] + numres * [1] + ((batchsize - len(fastlist)) - numres) * [0]
        numslow = np.unique(data['imgname']).size
        while numslow < (batchsize - len(fastlist)):
            difference = (batchsize - len(fastlist)) - numslow #the number we want to get.
            if dataset == 'COCO':
                fillin_names = [encode_imgname(os.path.split(x)[1]) for x in random.sample(allnames,round(1.9 * difference))]
            else:
                fillin_names = random.sample(allnames,round(2.1 * difference))
            fillname_list = ','.join([f"'{fname}'" for fname in fillin_names])
            newdata = readsql(f"SELECT * FROM {relname} WHERE imgname IN ({fillname_list}) AND category IN ({catlist})",hyperparams,whichdb="postgres")
            unique_new = list(newdata['imgname'].unique())
            if len(unique_new) > difference:
                if chatty: print("Overshot it by {len(unique_new) - difference}")
                unique_new = unique_new[0:difference]
                newdata = newdata[newdata['imgname'].isin(unique_new)]
            data = data.append(newdata)
            for fname in fillin_names:
                hit = fname in unique_new
                ishit.append(hit)
                if hit:
                    numslow += 1
                    if chatty: sys.stdout.write('*')
                    names.append(fname)
                else:
                    mymissing.append(fname)
                    if chatty: sys.stdout.write('.')
                sys.stdout.flush()
                if numslow == (batchsize - len(fastlist)):
                    break
        if chatty: sys.stdout.write('\n')
        if chatty: print(f"proportion of times imgname is in the database: {np.mean(ishit)}, time spent adding missing={time.time() - mt0}")

    def from_full_sql( ):
        data = readsql(f"SELECT * FROM {relname} R WHERE imgname IN ({slowlist}) AND category IN ({catlist})".format(relname,slowlist,catlist),hyperparams,whichdb="postgres")
        # to get numpy indexing working properly, cast these as ints becuase they are ints anyway.
        data['y'] = data['y'].astype(int)
        data['x'] = data['x'].astype(int)
    def read_dense(partition:str) -> Optional[Tuple[np.ndarray,np.ndarray,float,List[str]]]:
        while True:
            qsize = q.qsize()
            if chatty: print(f"qsize={qsize}")
            if qsize > qmax: # if queue is filling up, sleep a bit.
                time.sleep(5)
                if chatty:
                    sys.stdout.write('z')
                    sys.stdout.flush()
                continue
            if anticipate_missing: assert(allnames is not None)
            names = batch_names()
            fg,allpix = 0,0
            if dataset == 'COCO':
                if not anticipate_missing: names = [decode_imgname(name) + ".jpg" for name in names]
                unoptimized_read = False
            else:
                unoptimized_read = True
            dirname = os.path.split(names[0])[0]
            if dataset == 'COCO': #using special imagename encoding.
                cachef = {imgname : "{splitid},{encode_imgname(os.path.split(imgname)[1])}" for imgname in names}
                isfast = {imgname : os.path.exists(hyperparams.root("postproc/") + cachef[imgname]) for imgname in names}
                fastlist = [encode_imgname(os.path.split(imgname)[1]) for imgname in names if isfast[imgname]]
                whichslow = [imgname for imgname in names if not isfast[imgname]]
                slowlist = ','.join([f"'{encode_imgname(os.path.split(imgname)[1])}'" for imgname in whichslow])
            else:
                cachef = {imgname : "{splitid},{os.path.split(imgname)[1]}" for imgname in names}
                isfast = {imgname : os.path.exists(hyperparams.root("postproc/") + cachef[imgname]) for imgname in names}
                fastlist = [os.path.split(imgname)[1] for imgname in names if isfast[imgname]]
                whichslow = [imgname for imgname in names if not isfast[imgname]]
                slowlist = ','.join([f"'{os.path.split(imgname)[1]}'" for imgname in whichslow])
            if chatty: print(f"prop(fastlist)={len(fastlist) / len(names)}")
            query = f"SELECT category FROM splitcats WHERE splitid = {splitid} AND dataset = '{dataset}' AND seen = 1"
            catdf = np.squeeze(readsql(query,hyperparams,whichdb="postgres").values)
            assert(len(catdf) > 0), f"The category select query, {query}, is probably wrong."
            catlist = ','.join([f"'{category}'" for category in catdf])
            relname = "pixgt" if dataset == "COCO" else "pascal_pixgt"
            t0 = time.time()
            data = None
            if len(whichslow) > 0: #if whichslow is zero, all are in fastlist
                if chatty: print("Reading from database and writing to the cache")
                data = from_full_sql()
            mymissing = []
            if anticipate_missing and len(whichslow) > 0: #if whichslow is zero, all are in fastlist
                data = from_partion_sql()
            ta = time.time()
            assert(len(names) == batchsize), "loop adding names exited too soon"
            try:
                imgs = OrderedDict({name : (imread_wrap(name,hyperparams,tmp=unoptimized_read,mkdir=False),imread(name)) for name in names})
            except:
                suffix = ".jpg" if "jpg" not in names[0] else ""
                if dataset == "COCO":
                    try: #this occasionally fails for some reason.
                        imgs = OrderedDict([(name, (imread_wrap(os.path.join(dirname,decode_imgname(name + suffix)),hyperparams,tmp=unoptimized_read),imread(os.path.join(dirname,decode_imgname(name + suffix))))) for name in names])
                    except:
                        print("Failed to imread,continuing")
                        continue
                else:
                    imgs = OrderedDict([(name,(imread_wrap(os.path.join(dirname,name),tmp=unoptimized_read),imread(os.path.join(dirname,name)))) for name in names])
            # keep a record of missing data, so I can not waste time querying for it. If training and adding data at the same time, we will want to periodically delete this file.
            with open(misname,'a') as mif:
                for mis in mymissing:
                    mif.write(f"{mis}\n")
            mymissing = []
            t1 = time.time()
            sizes = {name : img[1].shape for (name,img) in imgs.items()}
            tmp = OrderedDict([(name,num_classes * np.ones((sizes[name][0],sizes[name][1]))) for name in set(names) - set(fastlist)])
            # whether or not using the none class, mark none with greatest int value in alphabet.
            gt = num_classes * np.ones((batchsize,img_s,img_s,num_classes))
            unknownError = False
            try:
                data.x = data.x.astype(int)
                data.y = data.y.astype(int)
            except:
                print("rare problem with dtype conversion")
                continue
            if data is not None: #'data' will be none if its all in the fastlist.
                for imgname,df in data.groupby('imgname'):
                    if dataset == 'COCO':
                        k = encode_imgname(imgname)
                    else:
                        k = os.path.join(dirname,imgname)
                    if k not in set(names) - set(fastlist):
                        continue
                    for category,dfp in df.groupby('category'):
                        if category not in np.squeeze(split['category'].values):
                            continue
                        if splitid != None:
                            intcat = split[split['category'] == category].index[0]
                        try: #how did this get so complicated?
                            tmp[k][dfp[['y','x']].values.T.tolist()] = intcat
                        except:
                            print(f"k={k},names={names},k in names = {k in names}")
                            try:
                                tmp[os.path.split(k)[1]][dfp[['y','x']].values.T.tolist()] = intcat
                            except:
                                unknownError = True
                                break
                    if unknownError: break
                if unknownError:
                    print("Weird error with tmp keys")
                    continue
            try:
                i = 0
                ordered_names = []
                for k in fastlist:
                    if chatty: sys.stdout.write('~')
                    gt[i] = onehot(pickle.load(open(hyperparams.root('postproc/' + str(splitid) + "," + k),'rb')),num_classes)
                    i += 1
                    ordered_names.append(k)
                for k in tmp.keys():
                    gtshaped = resize(tmp[k],(img_s,img_s),order=0)
                    mask = np.add.reduce(np.array([gtshaped == val for val in np.unique(tmp[k])]),0).astype(np.bool)
                    gtshaped[~mask] = nearest_round(gtshaped[~mask],np.unique(tmp[k]))
                    fg += np.count_nonzero(gtshaped - num_classes)
                    allpix += gtshaped.size
                    try:
                        if not os.path.exists(hyperparams.root('postproc/' + str(splitid) + "," + k)):
                            pickle.dump(gtshaped,open(hyperparams.root('postproc/' + str(splitid) + "," + k),'wb'))
                    except:
                        if chatty: print("failed to write to cache, continuing")
                    if i < batchsize: #otherwise, we just dump them out.
                        gt[i] = onehot(gtshaped,num_classes)
                        i += 1
                        ordered_names.append(k)
                imgout = np.array([imgs[k][0] for k in ordered_names])
                try:
                    assert(imgout.shape[0] == batchsize), f"imgout.shape={imgout.shape},batchsize={batchsize}"
                    # multiplying by num_classes to get proportion foreground because there is one-hot encoding.
                    prop_gt_bg = 1.0 - fg/allpix
                    outgt = gt.reshape(gt.shape[0],gt.shape[1] * gt.shape[2],gt.shape[3])
                    if (outgt.shape[0] > batchsize) or imgout.shape[0] > batchsize: #why would this possible happen? seems it did, so just handle it.
                        outgt = outgt[0:batchsize]
                        imgout = imgout[0:batchsize]
                    if chatty:
                        print(f"Proportion background: {prop_gt_bg}, threadid={mp.current_process()} took {t1 - t0} seconds to load db, prop time on query={(ta - t0)/(time.time() - t0)}")
                    assert(imgout.shape == (batchsize,img_s,img_s,3) and outgt.shape == (batchsize,img_s*img_s,num_classes)), "imgout.shape={} and outgt.shape={}".format(imgout.shape,outgt.shape,len(names))
                except:
                    print("Bad shape, going to next iteration")
                    continue
            except:
                if chatty: print("some problem with gathering gt and getting it in right shape.")
                continue
            out = (imgout,outgt,prop_gt_bg,names)
            if synchronous:
                if chatty: print("returning a batch")
                return out
            else:
                if chatty: print("put a batch on the queue")
                q.put(out)
    return read_dense

def outer_vis(hyperparams,dataset:str,split:pd.DataFrame,num_classes:int,splitid:int) -> Tuple[Callable,Callable,Callable]:
    '''
    A closure-returning function for visualizing things. We use a closure to keep bound variables related to colors consistent.
    '''
    classnames = list(np.squeeze(split['category'].values)) + ['None']
    fname = hyperparams.root(f'cache/{dataset}_colors_{splitid}')
    if not os.path.exists(fname): #make consistent colors for all figures by pickling it.
        colors = [(1,1,1)] + [(random.random(),random.random(),random.random()) for i in range(len(classnames))]
        randmap = mpl.colors.LinearSegmentedColormap.from_list('new_map', colors, N=len(classnames))
        pickle.dump( (colors,randmap), open(fname,'wb'))
    else:
        colors,randmap = pickle.load(open(fname,'rb'))
    def visualize_net(out:np.ndarray,img:np.ndarray):
        fig,axes = plt.subplots(2)
        plt.gcf().set_size_inches(36,36)
        axes[0].imshow(img)
        axout = axes[1].matshow(out,cmap=randmap)
        for ax in axes: ax.grid('off')
        formatter = plt.FuncFormatter(lambda val,loc: classnames[val])
        plt.colorbar(axout,ticks=range(len(classnames)),format=formatter)
    def visualize_compare(out:np.ndarray,numfuse:int,img:np.ndarray):
        fig,axes = plt.subplots(numfuse+3)
        fig.set_size_inches(36,36)
        ax = 0
        for k,v in out.items():
            axes[ax].matshow(out[k],cmap=randmap)
            axes[ax].set_title(k)
            ax += 1
        axes[-1].imshow(img)
        axes[-1].set_title("image")
        for ax in axes: ax.grid('off')
    def visualize(rawimg:np.ndarray,active_mask:np.ndarray,gt:np.ndarray,imgout:np.ndarray,imgname:str,t:int,splitid:int,numfuse:int,title:str):
        argmax_prob = imgout.shape[-1] == (num_classes)
        fig,axes = plt.subplots(nrows=2,ncols=3)
        fig.set_size_inches(30,10)
        fig.suptitle(title)
        axes[0][0].imshow(rawimg)
        axin = axes[0][1].matshow(gt,cmap=randmap,vmin=0,vmax=len(classnames))
        if argmax_prob:
            axout = axes[1][0].matshow(np.max(imgout,axis=2),cmap=randmap,vmin=0,vmax=len(classnames))
            axes[1][1].matshow(np.max(imgout,axis=2) / np.sum(imgout,axis=2))
        else:
            axout = axes[1][0].matshow(imgout,cmap=randmap,vmin=0,vmax=len(classnames))
        formatter = plt.FuncFormatter(lambda val,loc: classnames[val])
        fig.colorbar(axout,ticks=range(len(classnames)),format=formatter)
        maybe_mkdir(f'{sess_dir}/predvis/')
        axes[0][2].matshow(np.squeeze(active_mask))
        oname = hyperparams.root(f'{sess_dir}/predvis/{t}_{os.path.split(imgname)[1]}_{splitid}_{numfuse}_{title}.png')
        plt.savefig(oname)
        plt.close()
        return oname
    return visualize,visualize_net,visualize_compare

def update_bgscale(prop_pred_bg:float,prop_gt_bg:float,bgscale:float,adjust_rate=0.05,max_bgscale=0.09) -> float:
    '''
    When treating all classes equally in the loss function, the model is biased towards classifying pixels as background.
    The number returned by this function is the weight (should be less than 1) 

    prop_pred_bg - proportion of pixels predicted as background.
    prop_gt_bg - proportion of pixels labeled as background in ground truth.
    bgscale - current bgscale, to be updated.
    '''
    # when this term gets very large things blow up. That happened accidentally, but might as well put "min" in there to avoid anything horrible.
    return min(math.exp(adjust_rate * (prop_gt_bg - prop_pred_bg)) * bgscale,max_bgscale)

def create_tables(hyperparams):
    '''
    Keep track of training statistics in relational databases.
    '''
    dosql("CREATE TABLE IF NOT EXISTS fullyconv(nickname TEXT, trial INT, t INT,name TEXT,walltime DATE,loss_amount FLOAT,samples INT,posaccuracy FLOAT,negaccuracy FLOAT,numfuse INT)",hyperparams,whichdb="postgres")
    dosql("CREATE TABLE IF NOT EXISTS fullyconv_joint(nickname TEXT, trial INT, t INT,walltime DATE,loss_amount FLOAT,samples INT,posaccuracy FLOAT,negaccuracy FLOAT,metaaccuracy FLOAT,numfuse INT)",hyperparams,whichdb="postgres")
    dosql("CREATE TABLE IF NOT EXISTS fullyconv_cls(nickname TEXT, trial INT, t INT,name TEXT,realcat TEXT,predcat TEXT, prob FLOAT, numfuse INT)",hyperparams,whichdb="postgres")
    dosql("CREATE TABLE IF NOT EXISTS splitcats(splitid INT,seen INT, category TEXT)",hyperparams,whichdb="postgres")
    dosql("CREATE TABLE IF NOT EXISTS fuseconst(nickname TEXT, t INT, layer TEXT, const FLOAT)",hyperparams)
    print("Created tables")

def hdf_restore(weights:weight_t,biases:weight_t,modeldir:str,t:int,sess:tf.Session,where:Callable,prefix=''):
    '''
    A function for resuming training having saved the parameters as a python dictionary of arrays serialized into hdf5.

    weights - a dictionary of tensorflow variables.
    biases - a dictionary of tensorflow variables.
    sess - the tensorflow session we want to restore into.
    t - the timestep within training which we restore from.
    '''
    fname = modeldir + "/" + prefix + str(t) + ".hdf"
    if not os.path.exists(fname):
        return False
    npy_w,npy_b = dd.io.load(fname)
    for k in weights.keys():
        if k not in npy_w.keys():
            assert(where not in [train,joint_train])
            # random init.
            print(f"Warning, key {k} not in checkpoint: randomly initializing")
            shape = [x.value for x in weights[k].get_shape()]
            sess.run(weights[k].assign(tf.random_normal(shape,stddev=0.01)))
        else:
            sess.run(weights[k].assign(npy_w[k]))
            assert(np.array_equal(sess.run(weights[k]),npy_w[k]))
    for k in biases.keys():
        if k not in npy_b.keys():
            assert(where not in [train,joint_train])
            print(f"Warning, key {k} not in checkpoint: randomly initializing")
            # random init.
            shape = [x.value for x in biases[k].get_shape()]
            sess.run(biases[k].assign(tf.random_normal(shape,stddev=0.01)))
        else:
            sess.run(biases[k].assign(npy_b[k]))
            assert(np.array_equal(sess.run(biases[k]),npy_b[k]))
 
def setup(sess:tf.Session,hyperparams,args,split:pd.DataFrame,placeholders,num_readers=1,train=True,arch='vgg',opt_t='classify',where=None,img_s=224):
    '''
    Setup that needs to be done by train and test, so abstract it as a function.
    '''
    trestart,nickname,numfuse,dataset,splitid,batchsize,all_data_avail,synchronous = args.trestart,args.nickname,hyperparams.numfuse,args.dataset,args.splitid,args.batchsize,args.all_data_avail,args.synchronous
    anticipate_missing = args.anticipate_missing
    _X,_pix,_dropout,_bgscale,_active_mask = placeholders
    num_classes = len(split)
    if dataset == 'COCO':
        traindir,valdir = hyperparams.root('train_images'),hyperparams.root('val_images')
        const_imgnames = [ ] # need to pick out some for illustration's sake.
    else:
        traindir,valdir = '/data_b/aseewald/data/VOC2008/JPEGImages','/data_b/aseewald/data/VOC2008/JPEGImages'
        const_imgnames = [ ]
    # determine which 'trial' is happening.
    trial = readsql(f"SELECT max(trial) FROM fullyconv WHERE nickname = '{nickname}'",hyperparams)
    if args.from_scratch:
        print("Doing new trial.")
        if trial is None or len(trial) == 0 or trial.values[0][0] == None: trial = 0
        else: trial = trial['max'].ix[0] + 1
    else:
        assert(isinstance(args.trial,int))
        print("Starting is an int, so restarting at {args.trial}th trial")
        trial = args.trial
    if all_data_avail:
        train_names = os.listdir(traindir)
        val_names = os.listdir(valdir)
        print("Training with all data.")
    else:
        # CREATE MATERIALIZED VIEW 
        relname = "coco" if dataset == "COCO" else "pascal"
        if args.anticipate_missing: #we will guess and check.
            if dataset == 'COCO':
                train_names = os.listdir(hyperparams.root('train_images'))
                val_names = os.listdir(hyperparams.root('val_images'))
            else:
                raise NotImplementedError
            assert(not all_data_avail) #there's no reason for that to ever be the case.
        else: # we have precomputed the view of present images.
            train_names = np.squeeze(readsql(f"SELECT imgname FROM {trainview}",hyperparams,whichdb="postgres").values).tolist()
        print(f"Training with {len(train_names) / len(os.listdir(traindir))} of the data")
    if dataset == 'pascal':
        val_names = train_names
        num_epochs = 50 #do more training epochs because there is less data.
    else:
        val_names = [os.path.join(valdir,name) for name in val_names]
        num_epochs = 4
    # make them absolute paths.
    train_names = [os.path.join(traindir,name) for name in train_names]
    alphas = {}
    if opt_t != 'classify':
        alphas['class'],alphas['uncertainty'] = {},{}
        if numfuse == 0:
            alphas['class']['upsample5']  = tf.Variable(1.0,dtype=tf.float32,name='class-scale',trainable=True)
            alphas['uncertainty']['upsample5'] =  tf.Variable(1.0,dtype=tf.float32,name='uncertainty-scale',trainable=True)
        else:
            scale_data = readsql(f"SELECT * FROM fuseconst WHERE nickname = '{nickname}' AND t = {trestart}",hyperparams)
            if len(scale_data) == 0:
                if numfuse == 1: #needs to be normalized, so one degree of freedom here.
                    alphas['class']['upsample5'] = tf.Variable(0.5,dtype=tf.float32,name="class-scale5",trainable=True)
                    alphas['uncertainty']['upsample5'] = tf.Variable(0.5,dtype=tf.float32,name="uncertainty-scale5",trainable=True)
                elif numfuse == 2:
                    alphas['class']['upsample5'] = tf.Variable(0.3333,dtype=tf.float32,name="class-scale5",trainable=True)
                    alphas['uncertainty']['upsample5'] = tf.Variable(0.3333,dtype=tf.float32,name="uncertainty-scale5",trainable=True)
                    alphas['class']['upsample4'] = tf.Variable(0.3333,dtype=tf.float32,name="class-scale4",trainable=True)
                    alphas['uncertainty']['upsample4'] = tf.Variable(0.3333,dtype=tf.float32,name="uncertainty-scale4",trainable=True)
                else:
                    return False
                print("No scale data saved, so starting with equal weights")
            else:
                for sd in scale_data.iterrows():
                    alphas[sd['layer']] = sd['const']
                assert(len(scales.keys()) == numfuse),"not all the scale data saved, delete incomplete data from fuseconst and start with uniform data"
    else:
        if numfuse == 0:
            alphas['upsample5'] = 1.0
        else:
            scale_data = readsql(f"SELECT * FROM fuseconst WHERE nickname = '{nickname}' AND t = {trestart}",hyperparams)
            if len(scale_data) == 0:
                if numfuse == 1: #needs to be normalized, so one degree of freedom here.
                    alphas['upsample5'] = tf.Variable(0.5,dtype=tf.float32,name="scale5",trainable=True)
                elif numfuse == 2:
                    alphas['upsample5'] = tf.Variable(0.3333,dtype=tf.float32,name="scale5",trainable=True)
                    alphas['upsample4'] = tf.Variable(0.3333,dtype=tf.float32,name="scale4",trainable=True)
                else:
                    return False
                print("No scale data saved, so starting with equal weights")
            else:
                for sd in scale_data.iterrows():
                    alphas[sd['layer']] = sd['const']
                assert(len(scales.keys()) == numfuse),"not all the scale data saved, delete incomplete data from fuseconst and start with uniform data"
    if not synchronous:
        queue = {}
        test_queue = mp.Queue()
        queue['test'] = test_queue
        read_dense_test = read_dense_outer(test_queue,hyperparams,val_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,synchronous=synchronous)
        proc_test = mp.Process(target=read_dense_test,args=('all',))
        proc_test.start()
        if where in [joint_train]:
            pool_queue = mp.Queue()
            extra_queue = mp.Queue()
            read_dense_pool = read_dense_outer(pool_queue,hyperparams,train_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,synchronous=synchronous)
            read_dense_extra = read_dense_outer(extra_queue,hyperparams,train_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,synchronous=synchronous)
            proc_pool = mp.Process(target=read_dense_pool,args=('pool',))
            proc_extra = mp.Process(target=read_dense_extra,args=('extra',))
            proc_pool.start()
            proc_extra.start()
            queue['pool'] = pool_queue
            queue['extra'] = extra_queue
    else:
        queue = mp.Queue()
        read_dense = read_dense_outer(queue,hyperparams,train_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,synchronous=synchronous)   
        read_dense_test = read_dense_outer(queue,hyperparams,val_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,synchronous=synchronous)
        queue = (read_dense,read_dense_test)
    print("Before the get")
    parameters = initialize(hyperparams,num_classes,numfuse,opt_t)
    if synchronous:
        dfst = read_dense('pool')
    else:
        dfst = queue['pool'].get()
    print("Did a get successfully")
    debug_info = {'feed' : {_X : dfst[0],_pix : dfst[1],_bgscale : 0.05, _dropout : 1.0, _active_mask : np.ones((batchsize,img_s,img_s))},'sess' : sess}
    saver = tf.train.Saver(max_to_keep=50)
    if hyperparams.resolution_t == 'pixel': 
        mkops = mkops_pixelwise
    elif hyperparams.resolution_t == 'detection':
        mkops = mkops_ssd
    is_accurate,loss,optimizer,pixelwise_preds,inimgs,masks,outdict,lossfg,lossreg = mkops(_X,_pix,parameters,_dropout,num_classes,batchsize,numfuse,_bgscale,hyperparams,di=debug_info,arch=arch,opt_t=opt_t,alphas=alphas)
    sess.run(tf.initialize_all_variables()) # run it again, now that AdamOptimizer created some new variables. No evidence there is a problem with doing it twice.
    if not args.from_scratch and os.path.exists(hyperparams.root(modeldir)) and len(os.listdir(hyperparams.root(modeldir))) > 0:
        assert(trestart is not None), "if hdf, you must provide trestart"
        if trestart == -1:
            checkpoints = []
            for x in os.listdir(modeldir):
                m = re.match('pool-(\d+).hdf',x)
                if m:
                    checkpoints.append(int(m.group(1)))
            assert(len(checkpoints) > 0)
            trestart = max(checkpoints)
            if hyperparams.opt_t == 'joint' and hyperparams.sparsity_model[0] == 'afterwards':
                extra_checkpoints,baseline_checkpoints = [], []
                for x in os.listdir(modeldir):
                    m = re.match('extra-(\d+).hdf',x)
                    if m:
                        extra_checkpoints.append(int(m.group(1)))
                    m = re.match('extra-baseline-(\d+).hdf',x)
                    if m:
                        baseline_checkpoints.append(int(m.group(1)))
                checkpoints = [int(x.split('.')[0]) for x in os.listdir(modeldir) if re.match('\d+.hdf',x)]
                if len(extra_checkpoints) > 0:
                    sparse_trestart = max(extra_checkpoints)
                else:
                    sparse_trestart = 0
                if len(baseline_checkpoints) > 0:
                    baseline_trestart = max(baseline_checkpoints)
                else:
                    baseline_trestart = 0
                trestart = (trestart,sparse_trestart,baseline_trestart)
        hdf_restore(parameters[0],parameters[1],modeldir,trestart,sess,where)
        print("Sucessfully restored from iteration",trestart)
    else:
        print("Starting from scratch")
        if hyperparams.opt_t == 'joint' and hyperparams.sparsity_model[0] == 'afterwards':
            trestart = 0,0,0
        else:
            trestart = 0
    return queue,parameters,is_accurate,loss,optimizer,pixelwise_preds,inimgs,masks,outdict,lossfg,lossreg,saver,train_names,val_names,trial,trestart

def checkpoint(hyperparams,parameters,sess,num_main_iters,predcounts,pc_cols,bg_hist,nickname,modeldir,copy_name=None,prefix=''):
    tsa= time.time()
    # keys and values get saved in the same order, so this dependence on ordering works.
    w_keys,b_keys = list(parameters[0].keys()),list(parameters[1].keys())
    w_out,b_out = sess.run(list(parameters[0].values())), sess.run(list(parameters[1].values()))
    weight_snapshot = OrderedDict({k : w_out[w_keys.index(k)] for k in parameters[0].keys()})
    bias_snapshot = OrderedDict({k : b_out[b_keys.index(k)] for k in parameters[1].keys()})
    oname = modeldir + "/" + str(num_main_iters) + ".hdf"
    dd.io.save(oname,(weight_snapshot,bias_snapshot))
    if prefix != '':
        ln_name = modeldir + "/" + prefix + "-" + str(num_main_iters) + ".hdf"
        if os.path.exists(ln_name):
            subprocess.call(["unlink",ln_name])
        subprocess.call(["ln","-s",oname,ln_name])
    if copy_name is not None:
        ln_name =  modeldir + "/" + copy_name + ".hdf"
        if os.path.exists(ln_name):
            subprocess.call(["unlink",ln_name])
        subprocess.call(["ln","-s",oname,ln_name])
    # saving tf style.
    print(f"Saving at num_main_iters={num_main_iters}")
    pc = pd.DataFrame(predcounts,columns=pc_cols)
    pc.to_hdf(f'{args.cache_dir}/predcounts.hdf','root')
    pickle.dump(bg_hist,open(f'{args.cache_dir}/bghist.pkl','wb'))
    #saver.save(sess,modeldir + "/" + "model",global_step=num_main_iters)
    print(f"Done saving, which took {time.time() - tsa} seconds")

def log_confusion(lock,batchsize,nickname,gt,num_classes,gtprop,cats,trial,batchid,pixprobs,uncertainty_pixelwise_preds_v,net_uncertainty,numfuse,cls_tsv,img_s=224):
    try:
        gtshaped = gt.reshape((batchsize,img_s,img_s,num_classes))
        gtmeans = np.mean(gtprop,axis=0)
        # so that when we take min, we don't count the zero-entries ( )
        gtmin = np.min(gtmeans[gtmeans > 0])
        samples = 0
        tups = []
        dim_r = list(range(img_s))
        for intcat in range(num_classes):
            gtc = gtmeans[intcat]
            sample_prob = 0.25 * (gtmin / max(gtc,gtmin))
            num_pix_per_dim = int(math.ceil(img_s * math.sqrt(sample_prob)))
            catname = cats[intcat]
            for b in range(batchsize):
                ys,xs = random.sample(dim_r,num_pix_per_dim),random.sample(dim_r,num_pix_per_dim)
                samples += len(ys) * len(xs)
                for py in ys:
                    for px in xs:
                        tups.append((nickname,trial,batchid,cats[np.argmax(gtshaped[b,py,px])],catname,pixprobs[b,py,px,intcat],uncertainty_pixelwise_preds_v[b,py,px],net_uncertainty[b,py,px,0],numfuse))
        lock.acquire()
        for tup in tups:
            cls_tsv.write('\t'.join(map(str,tup)) + '\n')
        lock.release()
        print(f"wrote {samples} samples")
    except:
        print("Failed to write confusion data")


def joint_train_body(hyperparams,args,sess,nickname,trial,splitid,modeldir,queue,num_main_iters,gtprop,parameters,outdict,pixelwise_preds,uncertainty_pixelwise_preds,is_accurate,diff,inimgs,active_mask,lossfg,loss,lossreg,uncertainty_loss,optimizer,active_optimizer,sparse_optimizer,batchid,visualize,_X,_pix,_dropout,_bgscale,bgscale,batchsize,num_classes,predcounts,meta_hist,bg_hist,pc_cols,cats,gtpname,phase='pool',img_s=224):
    cls_tsv = open(f'{sess_dir}/cls-cache.tsv','a')
    log_ps = []
    log_lock = mp.Lock()
    if phase == 'extra':
        data_k = 'extra'
    elif phase in ['extra-baseline','pool']:
        data_k = 'pool'
    if args.synchronous:
        read_dense,read_dense_test = queue
        X,gt,prop_gt_bg,batchnames = read_dense(data_k)
    else:
        X,gt,prop_gt_bg,batchnames = queue[data_k].get()
    gtprop.append(np.mean(gt,axis=(0,1)))
    if batchid % 10 == 0:
        pickle.dump(gtprop,open(gtpname,'wb'))
    sys.stdout.write(',got\n')
    # when at visstep, do optimization of both.
    pos_accurate,neg_accurate,meta_accurate = is_accurate
    if (batchid % args.visstep == 0) or (batchid % args.infreq_visstep == 1):
        feed = {_X : X, _pix : gt, _dropout : 0.4, _bgscale : bgscale}
        pixelwise_preds_v,uncertainty_pixelwise_preds_v,diff_v,imgsin,mask_vals,loss_fg_v,loss_v,loss_reg_v,uncertainty_loss_v,_,_ = sess.run([pixelwise_preds,uncertainty_pixelwise_preds,diff,inimgs,active_mask,lossfg,loss,lossreg,uncertainty_loss,optimizer,active_optimizer], feed_dict=feed)
        num_main_iters += 1
        pred = pixelwise_preds_v = np.squeeze(pixelwise_preds_v)
        try:
            proc = mp.Process(target=active_corr,args=(mask_vals,X,gt,batchnames,num_main_iters,hyperparams))
            proc.start()
        except:
            print("Failed to start active_corr process")
        print(f"loss={loss_v},loss_fg={loss_fg_v},loss_reg={loss_reg_v/loss_v},loss_prop_fg={loss_fg_v/loss_v},loss_uncertainty={uncertainty_loss_v}")
        print(f"avg(mask_vals)={np.mean(mask_vals)}")
    if (batchid % args.visstep == 0):
        try:
            tn = time.time()
            numrand = min(batchsize,2)
            for i in random.sample(list(range(len(batchnames))),numrand):
                visualize[0](X[i],mask_vals[i],imgsin[i],pixelwise_preds_v[i],batchnames[i],batchid,splitid,hyperparams.numfuse,"fused")
            print(f"Frequent Visualization took {time.time() - tn} seconds")
        except: print("Something wrong with visualiation.")
    elif (batchid % args.infreq_visstep == 1):
        print("Starting infrequent visualization")
        feed = {_X : X, _pix : gt, _dropout : 0.3, _bgscale : bgscale}
        try:
            numrand = min(batchsize,2)
            which = random.sample(list(range(len(batchnames))),numrand)
            net_keys = [x for x in outdict.keys() if 'upsample' in x]
            active_keys = [x for x in outdict.keys() if 'active' in x]
            for k in net_keys:
                out_k = np.squeeze(sess.run(outdict[k],feed)).reshape(batchsize,img_s,img_s,num_classes)
                for i in which:
                    visualize[0](X[i],mask_vals[i],imgsin[i],out_k[i],batchnames[i],batchid,splitid,hyperparams.numfuse,k)
            for k in active_keys:
                out_k = np.squeeze(sess.run(outdict[k],feed)).reshape(batchsize,img_s,img_s)
                for i in which:
                    visualize[0](X[i],mask_vals[i],imgsin[i],out_k[i],batchnames[i],batchid,splitid,hyperparams.numfuse,k)
            print("finished infrequent visualization")
        except: print("Something wrong with infrequent visualization")
    else: #just running the optimizer according to the schedule.
        feed = {_X : X, _pix : gt, _dropout : 0.3, _bgscale : bgscale}
        t0 = time.time()
        if hyperparams.schedule_t == ('alternating',1,1):
            pixelwise_preds_v,outfull,loss_v,loss_fg_v,loss_reg_v,_,_ = sess.run([pixelwise_preds,outdict['upsample5'],loss,lossfg,lossreg,optimizer,active_optimizer], feed_dict=feed)
            num_main_iters += 1
        elif hyperparams.schedule_t[0] == 'alternating':
            episode_iter = batchid % (alter_main + alter_active)
            if episode_iter < alter_main:
                print("main iter")
                if phase in ['pool','extra-baseline']:
                    pixelwise_preds_v,outfull,loss_v,loss_fg_v,loss_reg_v,_ = sess.run([pixelwise_preds,outdict['upsample5'],loss,lossfg,lossreg,optimizer], feed_dict=feed)
                elif phase == 'extra':
                    discrete_mask_v = sess.run(discrete_mask,feed)
                    feed[_active_mask] = discrete_mask_v
                    pixelwise_preds_v,outfull,loss_v,loss_fg_v,loss_reg_v,_ = sess.run([pixelwise_preds,outdict['upsample5'],loss,lossfg,lossreg,sparse_optimizer], feed_dict=feed)
                    # sparse training.
                num_main_iters += 1
            else:
                print("active iter")
                pixelwise_preds_v,outfull,loss_v,loss_fg_v,loss_reg_v,_ = sess.run([pixelwise_preds,outdict['upsample5'],loss,lossfg,lossreg,active_optimizer], feed_dict=feed)
        elif hyperparams.schedule_t[0] == 'converging':
            if train_state == 'main':
                if phase in ['pool','extra-baseline']:
                    pixelwise_preds_v,outfull,loss_v,loss_fg_v,loss_reg_v,_ = sess.run([pixelwise_preds,outdict['upsample5'],loss,lossfg,lossreg,optimizer], feed_dict=feed)
                elif phase == 'extra':
                    discrete_mask_v = sess.run(discrete_mask,feed)
                    pixelwise_preds_v,outfull,loss_v,loss_fg_v,loss_reg_v,_ = sess.run([pixelwise_preds,outdict['upsample5'],loss,lossfg,lossreg,sparse_optimizer], feed_dict=feed)
                if done:
                    train_state = 'active'
                num_main_iters += 1
            else:
                pixelwise_preds_v,outfull,loss_v,loss_fg_v,loss_reg_v,_,_ = sess.run([pixelwise_preds,outdict['upsample5'],loss,lossfg,lossreg,active_optimizer], feed_dict=feed)
                if done:
                    train_state = 'main'
        pixprobs = normalize_unscaled_logits(outfull).reshape((batchsize,img_s,img_s,-1))
        minprob,maxprob = np.min(pixprobs),np.max(pixprobs)
        print(f"loss_v={loss_v},lossfg_prop={loss_fg_v/loss_v},lossreg_prop={loss_reg_v/loss_v},minprob={minprob},maxprob={maxprob}")
        if batchid % args.biasstep == 0:
            for intcat in range(num_classes):
                minc,maxc = np.min(pixprobs[:,:,:,intcat]),np.max(pixprobs[:,:,:,intcat])
                cat = cats[intcat] if intcat < num_classes else 'None'
                print("cat={cat},min={minc},max={maxc}")
                plt.title(cat + ' distribution')
                plt.hist(pixprobs[:,:,:].flatten())
                outn = hyperparams.root(f"results/{nickname}")
                if not os.path.exists(outn): subprocess.call(["mkdir",outn])
                plt.savefig(outn + "/" + str(batchid) + "_" + cat)
                plt.close()
        pred = pixelwise_preds_v
        prop_pred_bg = np.count_nonzero(pred == num_classes) / pred.size
        histogram = np.bincount(pred.flatten())
        predcount = np.concatenate((histogram,np.zeros(num_classes - len(histogram)),[batchid]))
        predcounts.append(predcount)
        print("prediction frequency: ",list(zip(pc_cols,predcount)))
        bgscale = update_bgscale(prop_pred_bg,prop_gt_bg,bgscale)
        if bgscale > 1: print("Warning: bgscale is getting big and it is getting weird.")
        else: print(f"Proportion predicted bg: {prop_pred_bg} ,New bgscale: {bgscale}")
        bg_hist.append(bgscale)
        print(f"Optimizer took {time.time() - t0} seconds")
    # TEST
    if (batchid % args.valstep == 0):
        for i in range(args.num_test_batches):
            # testing on the training set here.
            if args.synchronous:
                X,gt,prop_gt_bg,_ = read_dense_test('all')
            else:
                X,gt,prop_gt_bg,_ = queue['test'].get()
            feed_acc = {_X : X, _pix : gt, _dropout : 1.0, _bgscale : bgscale}
            if tf.__version__.split('.')[0] == '1':
                outfull,uncertainty_pixelwise_preds_v,diff_v,net_uncertainty,posaccuracy,negaccuracy,metaaccuracy,loss_v = sess.run([outdict['net'],uncertainty_pixelwise_preds,diff,outdict['net_active'],pos_accurate,neg_accurate,meta_accurate,loss],feed_dict=feed_acc)
            elif tf.__version__.split('.')[0] == '0':
                outfull,uncertainty_pixelwise_preds_v,net_uncertainty,posaccuracy,negaccuracy,metaaccuracy,loss_v,summary = sess.run([outdict['net'],uncertainty_pixelwise_preds,outdict['net_active'],pos_accurate,neg_accurate,meta_accurate,loss,merged],feed_dict=feed_acc)
                train_writer.add_summary(summary,batchid)
            ploss_v = -1 * (diff_v - uncertainty_pixelwise_preds_v)
            avg_uncertainty,avg_diff,avg_ploss = np.mean(uncertainty_pixelwise_preds_v),np.mean(diff_v),np.mean(ploss_v)
            print(f"average uncertainty: {avg_uncertainty}, average meta correctness: {avg_diff}, average real loss: {avg_ploss}")
            meta_hist.append({'timestep' : num_main_iters,'average uncertainty' : avg_uncertainty,'average meta correctness' : avg_diff,'avg real loss' : avg_ploss,'phase' : phase})
            pixprobs = normalize_unscaled_logits(outfull).reshape((batchsize,img_s,img_s,-1))
            if hyperparams.active_network in hyperparams.uncertainty_types:
                pacc,nacc,macc= np.mean(posaccuracy),np.mean(negaccuracy),np.mean(metaaccuracy)
                print(f"posaccuracy={pacc},negaccuracy={nacc},metaaccuracy={macc}")
                q_acc = lambda x:f"INSERT INTO fullyconv_joint VALUES('{nickname}',{trial},{batchid},'{phase}',{x},{loss_v},{len(X)},{pacc},{nacc},{macc},{hyperparams.numfuse})"
                dosql(xplatformtime(q_acc),hyperparams,whichdb="postgres")
            #log_confusion(log_lock,batchsize,nickname,pred,gt,num_classes,gtprop,cats,trial,batchid,pixprobs,uncertainty_pixelwise_preds_v,net_uncertainty,hyperparams.numfuse,cls_tsv)
            try:
                log_ps.append(mp.Process(target=log_confusion,args=(log_lock,batchsize,nickname,gt,num_classes,gtprop,cats,trial,batchid,pixprobs,uncertainty_pixelwise_preds_v,net_uncertainty,hyperparams.numfuse,cls_tsv)))
                log_ps[-1].start()
            except:
                log_ps = []
                continue
    print("Waiting for logging threads")
    for log_p in log_ps:
        log_p.join()
    print("All logging threads joined.")
    # checkpointing.
    if num_main_iters % args.savestep == (args.savestep - 1):
        checkpoint(hyperparams,parameters,sess,num_main_iters,predcounts,pc_cols,bg_hist,nickname,modeldir,prefix=phase)
    cls_tsv.close()
    return num_main_iters,gtprop,predcounts,bgscale,meta_hist

def joint_train(args:argparse.Namespace,hyperparams:hp.PixelwiseHyperParams,arch='vggnet',device='GPU',img_s=224,bgscale=0.05):
    '''
    Trains the main model and the other model used for 
    '''
    global misname
    opt_t = 'joint'
    nickname,dataset,batchsize,splitid,numfuse = args.nickname,args.dataset,args.batchsize,args.splitid,hyperparams.numfuse
    misname = f'{args.cache_dir}/missing.pkl'
    if device == "GPU":
        devstr = '/gpu:0'
    elif device == "CPU":
        devstr = '/cpu:0'
    # this is shared across nicknames, so keep it at the dataset level.
    maybe_mkdir(hyperparams.root('gtprop'))
    gtpname = hyperparams.root('gtprop/{}.pkl'.format(splitid))
    if not os.path.exists(gtpname):
        gtprop = []
    else:
        gtprop = pickle.load(open(gtpname,'rb'))
    predcounts = []
    bg_hist,meta_hist = [],[]
    loss_history = { }
    convergence_iter = 0
    create_tables(hyperparams)
    split = readsql(f"SELECT * FROM splitcats WHERE dataset = '{dataset}' AND splitid = {splitid} AND seen = 1",hyperparams,whichdb="postgres")
    pc_cols = np.concatenate((split['category'].values,['None','timestep']))
    num_classes = len(split)
    cats = ['' for cat in split['category']]
    for category in split['category'].values:
        intcat = split[split['category'] == category].index[0]
        cats[intcat] = category
    if args.include_none:
        num_classes += 1
        cats.append('None')
    visualize = outer_vis(hyperparams,dataset,split,num_classes,splitid=splitid)
    with tf.device(devstr):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            # Using a concrete batchsize because deconvolution had a problem with a ? dimension.
            _X = tf.placeholder(tf.float32,[batchsize,img_s,img_s,3])
            _pix = tf.placeholder(tf.float32,[batchsize,img_s * img_s,num_classes])
            _dropout = tf.placeholder(tf.float32,shape=())
            _bgscale = tf.placeholder(tf.float32,shape=())
            _active_mask = tf.placeholder(tf.float32,[batchsize,img_s,img_s])
            if hyperparams.resolution_t == 'pixel':
                active_mask = np.ones((batchsize,img_s,img_s))
            placeholders = (_X,_pix,_dropout,_bgscale,_active_mask)
            # trestart is both an argument and a return value because it is conditionally updated.
            queue,parameters,is_accurate,loss,optimizer,pixelwise_preds,inimgs,masks,outdict,lossfg,lossreg,saver,train_names,val_names,trial,trestart = setup(sess,hyperparams,args,split,placeholders,train=True,arch=arch,opt_t=opt_t,where=joint_train)
            if hyperparams.schedule_t[0] == 'alternating':
                alter_main,alter_active = hyperparams.schedule_t[1],hyperparams.schedule_t[2]
                trestart,sparse_trestart,baseline_trestart = trestart
            loss,uncertainty_loss = loss
            # how to unpack masks?
            optimizer,active_optimizer,sparse_optimizer = optimizer
            active_mask,discrete_mask = masks
            # uncertainty_pixelwise_preds is the difference between predicted uncertainty and true correctness.
            pixelwise_preds,uncertainty_pixelwise_preds,diff = pixelwise_preds
            pos_accurate,neg_accurate,meta_accurate = is_accurate
            if tf.__version__.split('.')[0] == '0':
                print("Not using summaries")
            elif tf.__version__.split('.')[0] == '0':
                train_writer = tf.train.SummaryWriter(logdir + '/train', sess.graph)
                merged = tf.merge_all_summaries()
            num_epochs = 50 if dataset == 'pascal' else 1.0 #do more training epochs because there is so little data.
            num_batches = int(num_epochs * len(train_names) // batchsize)
            num_main_iters = trestart
            print("About to begin training jointly")
            num_pool_batches = int(hyperparams.pool_iter_prop * num_batches)
            num_extra_batches = int((1-hyperparams.pool_iter_prop) * num_batches)
            for batchid in range(trestart,num_pool_batches):
                if os.path.exists(modeldir + "/afterpool.hdf"): # this means we're done with this section of training.
                    break
                epoch = (batchid * batchsize) / len(train_names)
                sys.stdout.write(f"pool datetime={time.asctime()},batchid={batchid},epoch={epoch}")
                num_main_iters,gtprop,predcounts,bgscale,meta_hist = joint_train_body(hyperparams,args,sess,nickname,trial,splitid,modeldir,queue,num_main_iters,gtprop,parameters,outdict,pixelwise_preds,uncertainty_pixelwise_preds,is_accurate,diff,inimgs,active_mask,lossfg,loss,lossreg,uncertainty_loss,optimizer,active_optimizer,sparse_optimizer,batchid,visualize,_X,_pix,_dropout,_bgscale,bgscale,batchsize,num_classes,predcounts,meta_hist,bg_hist,pc_cols,cats,gtpname,phase='pool')
            # end of pool training loop, do a checkpoint.
            t_afterpool = num_main_iters
            checkpoint(hyperparams,parameters,sess,num_main_iters,predcounts,pc_cols,bg_hist,nickname,modeldir,copy_name='afterpool',prefix='pool')
            if hyperparams.sparsity_model[0] == 'afterwards':
                print("Continuing training in sparse mode.")
                for batchid in range(sparse_trestart,num_extra_batches):
                    epoch = (batchid * batchsize) / len(train_names)
                    sys.stdout.write(f"sparse extra datetime={time.asctime()},batchid={batchid},epoch={epoch}")
                    num_main_iters,gtprop,predcounts,bgscale,meta_hist = joint_train_body(hyperparams,args,sess,nickname,trial,splitid,modeldir,queue,num_main_iters,gtprop,parameters,outdict,pixelwise_preds,uncertainty_pixelwise_preds,is_accurate,diff,inimgs,active_mask,lossfg,loss,lossreg,uncertainty_loss,optimizer,active_optimizer,sparse_optimizer,batchid,visualize,_X,_pix,_dropout,_bgscale,bgscale,batchsize,num_classes,predcounts,meta_hist,bg_hist,pc_cols,cats,gtpname,phase='extra')
                    # sparse training.
                hdf_restore(parameters[0],parameters[1],modeldir,t_afterpool,sess,where=joint_train)
                # reset to post-pool checkpoint.
                for batchid in range(baseline_trestart,num_extra_batches): #dense train with same number of iters.
                    epoch = (batchid * batchsize) / len(train_names)
                    sys.stdout.write(f"baseline extra datetime={time.asctime()},batchid={batchid},epoch={epoch}")
                    num_main_iters,gtprop,predcounts,bgscale,meta_hist = joint_train_body(hyperparams,args,sess,nickname,trial,splitid,modeldir,queue,num_main_iters,gtprop,parameters,outdict,pixelwise_preds,uncertainty_pixelwise_preds,is_accurate,diff,inimgs,active_mask,lossfg,loss,lossreg,uncertainty_loss,optimizer,active_optimizer,sparse_optimizer,batchid,visualize,_X,_pix,_dropout,_bgscale,bgscale,batchsize,num_classes,predcounts,meta_hist,bg_hist,pc_cols,cats,gtpname,phase='extra-baseline')
                    # dense training.
            else:
                print("Done training.")
            print("Done training, starting testing from checkpoints")
            test_all(hyperparams,args,sess,nickname,trial,splitid,modeldir,queue,num_main_iters,gtprop,parameters,outdict,pixelwise_preds,uncertainty_pixelwise_preds,is_accurate,diff,inimgs,active_mask,lossfg,loss,lossreg,uncertainty_loss,optimizer,active_optimizer,sparse_optimizer,batchid,visualize,_X,_pix,_dropout,_bgscale,bgscale,batchsize,num_classes,predcounts,meta_hist,bg_hist,pc_cols,cats,gtpname) #an infinite loop.

def train(hyperparams,args,bgscale=0.05,device="GPU",img_s=224):
    
    '''
    numfuse - number of convolutional layers to upsample and add together to get a "fused" pixelwise prediction.
    timeout - number of seconds after which to interrupt training. Set by default to 36 hours.
    all_data_avail - if True, don't check whether data exists. 
    num_readers - The database reads happen in another process communicating with this function over a pipe. this argument is the number of such processes.
    val_step - Run testing of performance with this period.
    vis_step - Visualize classifications with this period.
    infreq_visstep - Additional visualizations with this period.
    biasstep - Analyze biases of the network with this period.

    Other arguments are assumed self-explanatory or equal in name to something explained elsewhere.
    '''
    global misname
    signal.signal(signal.SIGINT,signal_handler)
    misname = f'{args.cache_dir}/missing_{args.nickname}_{args.splitid}.pkl'
    predcounts = []
    batchsize,nickname,dataset = args.batchsize,args.nickname,args.dataset
    # should be moved to hyperparams if I play with it more.
    splitid,numfuse = args.splitid,hyperparams.numfuse
    if dataset == 'pascal':
        anticipate_missing = False
    walltime_0 = time.time()
    create_tables(hyperparams)
    cls_tsv = open(f'fc-cls-cache_{nickname}.tsv','a')
    dosql("CREATE TABLE IF NOT EXISTS fullyconv_settings(nickname TEXT, pkl TEXT)",hyperparams)
    setname = f'{sess_dir}/settings.pkl'
    dosql(f"INSERT INTO fullyconv_settings VALUES ('{nickname}','{setname}')",hyperparams)
    # this should change. 
    if not os.path.exists(setname):
        pickle.dump(hyperparams,open(setname,'wb'))
    bg_hist = []
    if len(readsql("SELECT * FROM splitcats",hyperparams,whichdb="postgres")) == 0:
        addcat(split)
    split = readsql(f"SELECT * FROM splitcats WHERE dataset = '{dataset}' AND splitid = {splitid} AND seen = 1",hyperparams,whichdb="postgres")
    num_classes = len(split)
    visualize = outer_vis(hyperparams,dataset,split,num_classes,splitid=splitid)
    cats = ['' for cat in split['category']]
    for category in split['category'].values:
        intcat = split[split['category'] == category].index[0]
        cats[intcat] = category
    if args.include_none:
        num_classes += 1 # including none.
        cats.append('None')
    assert(num_classes == len(cats))
    if not os.path.exists(hyperparams.root('gtprop')):
        subprocess.call(["mkdir",hyperparams.root('gtprop')])
    gtpname = hyperparams.root('gtprop/{}.pkl'.format(splitid))
    if not os.path.exists(gtpname):
        gtprop = []
    else:
        gtprop = pickle.load(open(gtpname,'rb'))
    orderfile = f'{args.cache_dir}/verify_split_order.pkl'
    if not os.path.exists():
        pickle.dump(cats,open(orderfile,'wb'))
    else:
        assert(cats == pickle.load(open(orderfile,'rb'))),"ordering is different on different runs. This should be impossible."
    pc_cols = np.concatenate((split['category'].values,['None','timestep']))
    if device == "GPU":
        devstr = '/gpu:0'
    elif device == "CPU":
        devstr = '/cpu:0'
    num_epochs = 50 if dataset == 'pascal' else 8 #do more training epochs because there is so little data.
    with tf.device(devstr):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            # Using a concrete batchsize because deconvolution had a problem with a ? dimension.
            _X = tf.placeholder(tf.float32,[batchsize,img_s,img_s,3])
            _pix = tf.placeholder(tf.float32,[batchsize,img_s * img_s,num_classes])
            _dropout = tf.placeholder(tf.float32,shape=())
            _bgscale = tf.placeholder(tf.float32,shape=())
            placeholders = (_X,_pix,_dropout,_bgscale)
            # trestart is both an argument and a return value because it is conditionally updated.
            queue,parameters,is_accurate,loss,optimizer,pixelwise_preds,inimgs,masks,outdict,lossfg,lossreg,saver,train_names,val_names,trial,trestart = setup(sess,hyperparams,args,split,placeholders,train=True,arch=args.arch)
            maybe_mkdir(modeldir)
            num_batches = int(num_epochs * len(train_names) // batchsize)
            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(logdir + '/train', sess.graph)
            pos_accurate,neg_accurate = is_accurate
            print("About to begin training")
            for batchid in range(trestart,num_batches):
                epoch = (batchid * batchsize) / len(train_names)
                sys.stdout.write(f"datetime={time.asctime()},batchid={batchid},epoch={epoch}")
                if args.synchronous:
                    X,gt,prop_gt_bg,batchnames = read_dense(queue,hyperparams,train_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,synchronous=args.synchronous)
                else:
                    X,gt,prop_gt_bg,batchnames = queue.get()
                gtprop.append(np.mean(gt,axis=(0,1)))
                if batchid % 10 == 0:
                    pickle.dump(gtprop,open(gtpname,'wb'))
                sys.stdout.write(',got\n')
                if (batchid % args.visstep == 0):
                    feed = {_X : X, _pix : gt, _dropout : 0.4, _bgscale : bgscale}
                    pixelwise_preds_v,imgsin,loss_fg_v,loss_v,loss_reg_v,_ = sess.run([pixelwise_preds,inimgs,lossfg,loss,lossreg,optimizer], feed_dict=feed)
                    pixelwise_preds_v = np.squeeze(pixelwise_preds_v)
                    pred = pixelwise_preds_v
                    print(f"loss={loss_v},loss_fg={loss_fg_v},loss_reg={loss_reg_v/loss_v},loss_prop_fg={loss_fg_v/loss_v}")
                    try:
                        numrand = min(batchsize,2)
                        for i in random.sample(list(range(len(batchnames))),numrand):
                            tn = time.time()
                            oname = visualize[0](X[i],imgsin[i],pixelwise_preds_v[i],batchnames[i],batchid,splitid,numfuse,"fused")
                            print(f"Saving {oname} took {time.time() - tn} seconds")
                    except: print("Something wrong with visualiation.")
                elif (batchid % args.infreq_visstep == 1):
                    print("Starting infrequent visualization")
                    feed = {_X : X, _pix : gt, _dropout : 0.3, _bgscale : bgscale}
                    try:
                        for k in outdict:
                            pixelwise_preds_v = sess.run(outdict[k],feed)
                            pixelwise_preds_v = np.squeeze(pixelwise_preds_v).reshape((batchsize,img_s,img_s,num_classes))
                            pred = pixelwise_preds_v
                            for i in random.sample(list(range(len(batchnames))),numrand):
                                visualize[0](X[i],imgsin[i],pixelwise_preds_v[i],batchnames[i],batchid,splitid,numfuse,"fused")
                        print("finished infrequent visualization")
                    except: print("Something wrong with infrequent visualization")
                else: #just running the optimizer.
                    feed = {_X : X, _pix : gt, _dropout : 0.3, _bgscale : bgscale}
                    t0 = time.time()
                    pred,outfull,loss_v,loss_fg_v,loss_reg_v,_ = sess.run([pixelwise_preds,outdict['upsample5'],loss,lossfg,lossreg,optimizer], feed_dict=feed)
                    pixprobs = normalize_unscaled_logits(outfull)
                    minprob,maxprob = np.min(pixprobs),np.max(pixprobs)
                    print(f"loss_v={loss_v},lossfg_prop={loss_fg_v/loss_v},lossreg_prop={loss_reg_v/loss_v},minprob={minprob},maxprob={maxprob}")
                    if batchid % biasstep == 0:
                        for intcat in range(num_classes):
                            minc,maxc = np.min(pixprobs[:,:,:,intcat]),np.max(pixprobs[:,:,:,intcat])
                            cat = cats[intcat] if intcat < num_classes else 'None'
                            print(f"cat={cat},min={minc},max={maxc}")
                            plt.title(cat + ' distribution')
                            plt.hist(pixprobs[:,:,:].flatten())
                            plt.savefig(f'{t_sess_dir(batchid)}/catdistr.jpg')
                            plt.close()
                    prop_pred_bg = np.count_nonzero(pred == num_classes) / pred.size
                    histogram = np.bincount(pred.flatten())
                    predcount = np.concatenate((histogram,np.zeros(num_classes - len(histogram)),[batchid]))
                    predcounts.append(predcount)
                    print("prediction frequency: ",list(zip(pc_cols,predcount)))
                    bgscale = update_bgscale(prop_pred_bg,prop_gt_bg,bgscale)
                    if bgscale > 1: print("Warning: bgscale is getting big and it is getting weird.")
                    else: print(f"Proportion predicted bg: {prop_pred_bg} ,New bgscale: {bgscale}")
                    bg_hist.append(bgscale)
                    print(f"Optimizer took {time.time() - t0} seconds")
                # TEST
                if (batchid % args.valstep == 0):
                    for i in range(args.num_test_batches):
                        if args.synchronous:
                            X,gt,prop_gt_bg,_ = read_dense(queue,hyperparams,train_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,synchronous=args.synchronous)
                        else:
                            X,gt,prop_gt_bg,_ = queue.get()
                        feed_acc = {_X : X, _pix : gt, _dropout : 1.0, _bgscale : bgscale}
                        outfull,posaccuracy,negaccuracy,loss_v,summary = sess.run([outdict['net'],pos_accurate,neg_accurate,loss,merged],feed_dict=feed_acc)
                        train_writer.add_summary(summary,batchid)
                        posaccuracy,negaccuracy = np.mean(posaccuracy),np.mean(negaccuracy)
                        pixprobs = normalize_unscaled_logits(outfull)
                        print(f"posaccuracy={posaccuracy},negaccuracy={negaccuracy}")
                        q_acc = lambda x:f"INSERT INTO fullyconv VALUES('{nickname}',{trial},{batchid},{x},{loss_v},{len(X)},{posaccuracy},{negaccuracy})"
                        dosql(xplatformtime(q_acc),hyperparams,whichdb="postgres")
                if batchid % savestep == (savestep - 1):
                    # saving hdf style
                    tsa= time.time()
                    # keys and values get saved in the same order, so this dependence on ordering works.
                    w_keys,b_keys = list(parameters[0].keys()),list(parameters[1].keys())
                    w_out,b_out = sess.run(list(parameters[0].values())), sess.run(list(parameters[1].values()))
                    weight_snapshot = OrderedDict({k : w_out[w_keys.index(k)] for k in parameters[0].keys()})
                    bias_snapshot = OrderedDict({k : b_out[b_keys.index(k)] for k in parameters[1].keys()})
                    dd.io.save(f'{modeldir}/{str(batchid)}.hdf',(weight_snapshot,bias_snapshot))
                    # saving tf style.
                    print(f"Saving at batchid={batchid}")
                    pc = pd.DataFrame(predcounts,columns=pc_cols)
                    pc.to_hdf(f'{sess_dir}/predcounts.hdf','root')
                    mc = pd.DataFrame(meta_hist)
                    mc.to_hdf(f'{sess_dir}/metahist.hdf','root')
                    pickle.dump(bg_hist,open(f'{args.cache_dir}/bghist.pkl','wb'))
                    saver.save(sess,f'{modeldir}/model',global_step=batchid)
                    print(f"Done saving, which took {time.time() - tsa} seconds")

def initialize(hyperparams:hp.PixelwiseHyperParams,num_classes:int,numfuse:int,opt_t:str,pretrained='vggnet.npy',hack_suffix=None,img_s=224):
    '''
    Goal: eventually replace pretrained with earlier conv weights learned on COCO pixelwise.
    '''
    try:
        vgg = np.load(hyperparams.root('cnn/npymodels/{}'.format(pretrained)),encoding='bytes').item()
    except:
        vgg = np.load(pretrained,encoding='bytes').item()
    take = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3']
    weights = {}
    for layer in take:
        weights[layer] = vgg[layer]
    weights['myconv5_1'] = [0.01 * np.random.randn(3,3,512,512), 0.001 * np.random.randn(512)]
    weights['myconv5_2'] = [0.01 * np.random.randn(3,3,512,512), 0.001 * np.random.randn(512)]
    weights['upsample5'] = [0.01 * np.random.randn(img_s//7,img_s//7,num_classes,num_classes), 0.0001 * np.random.randn(num_classes)]
    weights['narrow5'] = [0.01 * np.random.randn(1,1,512,num_classes), 0.0001 * np.random.randn(num_classes)]
    if numfuse >= 1:
        weights['upsample4'] = [0.01 * np.random.randn(img_s//14,img_s//14,num_classes,num_classes), 0.0001 * np.random.randn(num_classes)]
        weights['narrow4'] = [0.01 * np.random.randn(1,1,512,num_classes), 0.0001 * np.random.randn(num_classes)]
    if numfuse >= 2:
        weights['upsample3'] = [0.01 * np.random.randn(img_s//28,img_s//28,num_classes,num_classes), 0.0001 * np.random.randn(num_classes)]
        weights['narrow3'] = [0.01 * np.random.randn(3,3,256,num_classes), 0.01 * np.random.randn(num_classes)]
    if opt_t != 'classify':
        # need uncertainty parameters here.
        if hyperparams.ensemble_num is not None:
            pass 
        else:
            weights['active5'] = [0.01 * np.random.randn(img_s//7,img_s//7,1,hyperparams.active_dim[0]), 0.0001 * np.random.randn(1)]
            weights['narrow5_est'] = [0.01 * np.random.randn(1,1,512,hyperparams.active_dim[0]), 0.0001 * np.random.randn(hyperparams.active_dim[0])]
            if numfuse >= 1:
                weights['active4'] = [0.01 * np.random.randn(img_s//14,img_s//14,1,hyperparams.active_dim[0]), 0.0001 * np.random.randn(1)]
                weights['narrow4_est'] = [0.01 * np.random.randn(1,1,512,hyperparams.active_dim[0]), 0.0001 * np.random.randn(hyperparams.active_dim[0])]
            if numfuse >= 2:
                weights['active3'] = [0.01 * np.random.randn(img_s//28,img_s//28,1,hyperparams.active_dim[0]), 0.0001 * np.random.randn(1)]
                weights['narrow3_est'] = [0.01 * np.random.randn(1,1,256,hyperparams.active_dim[0]), 0.0001 * np.random.randn(hyperparams.active_dim[0])]
    return(totensors(weights,trainable=True,xavier={k : False for k in weights.keys()},hack_suffix=hack_suffix))

def archof(arch:str,opt_t:str,hyperparams) -> Callable:
    if opt_t == 'classify':
        if arch == 'vgg':
            return arch_vgg
        elif arch == 'resnet':
            return arch_resnet
        else:
            assert(False), "unknown arch"
    elif opt_t == ['ensemble-joint','ensemble-uncertainty']:
        return arch_vgg_ensemble
    elif opt_t in ['joint','uncertainty']:
        if hyperparams.active_network in hyperparams.uncertainty_types:
            return uncertainty_vgg
        elif hyperparams.active_network == 'emoc':
            return emoc_vgg
    else:
        assert(False), "unknown opt_t"

def run(hyperparams,topk=6,arch='vgg',dropout_iters=1,dropout=1.0,img_s=224):
    i = 0
    imgnames = [os.path.join(imgdir,x ) for x in os.listdir(imgdir)]
    dataset = 'COCO'
    split = pd.read_sql("SELECT * FROM splitcats WHERE dataset = '{dataset}' AND splitid = {splitid} AND seen = 1",sqlite3.connect('splitcats.db'))
    num_classes = len(split)
    classnames = np.array(list(np.squeeze(split['category'].values)) + ['None'])
    archfn = archof(arch)
    if not os.path.exists(hyperparams.root("cache")):
        subprocess.call(["mkdir",hyperparams.root("cache")])
    _,visualize_net,visualize_compare = outer_vis(dataset,split,num_classes,splitid=splitid)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        parameters = initialize(hyperparams,num_classes,numfuse,opt_t)
        try:
            fs = [x for x in os.listdir(hyperparams.root(modeldir)) if "hdf" in x and "best.hdf" not in x]
            existing_ts = [int(os.path.split(x)[1].split('.')[0]) for x in fs]
            hdf_restore(parameters[0],parameters[1],modeldir,np.max(existing_ts),sess,run)
        except:
            try:
                hdf_restore(parameters[0],parameters[1],os.getcwd(),7279,sess,run)
            except:
                print("A trained model does not exist, contact Alex at aseewald@indiana.edu to determine the issue")
                sys.exit(1)
        # Using a concrete batchsize because deconvolution had a problem with a ? dimension.
        _X = tf.placeholder(tf.float32,[batchsize,img_s,img_s,3])
        _dropout = tf.placeholder(tf.float32,shape=())
        outs = archfn(hyperparams,_X,parameters[0],parameters[1],_dropout,num_classes,batchsize,numfuse)
        out_trials = []
        dropout_var = None
        while i < len(imgnames):
            amount = min(len(imgnames)-i,batchsize)
            X = [imread_wrap(imgnames[i+j]) for j in range(amount)]
            feed = {_X : X, _dropout : dropout}
            for t in range(dropout_iters):
                out,mask = sess.run([outs,dropout_var],feed)
                out_trials.append({k : normalize_unscaled_logits(v) for k,v in out.items()})
            for k in out_trials[0].keys():
                vs = [x[k] for x in out_trials]
                #out[k] = None
            # average them to get 'out'
            for j in range(amount):
                net = np.argmax(out['net'][j].reshape((img_s,img_s,num_classes)),axis=2)
                tail = os.path.split(imgnames[i+j])[1]
                visualize_net(net,X[j])
                plt.savefig(os.path.join(sess_dir,f"net_{tail}.jpg"))
                which = np.flipud(np.argsort(np.sum(out['net'][j],axis=0)))[0:topk]
                k_cat = classnames[which]
                print(f"Top k categories in image={tail} are {k_cat}")
            for j in range(amount):
                visualize_compare({k : np.argmax(out[k][j].reshape((img_s,img_s,num_classes)),axis=2) for k in out.keys()},numfuse,X[j])
                plt.savefig(os.path.join(sess_dir,f"comparison_{os.path.split(imgnames[i+j])[1]}.jpg"))
            plt.close("all")
    
def test_all(hyperparams,args,sess,nickname,trial,splitid,modeldir,queue,num_main_iters,gtprop,parameters,outdict,pixelwise_preds,uncertainty_pixelwise_preds,is_accurate,diff,inimgs,active_mask,lossfg,loss,lossreg,uncertainty_loss,optimizer,active_optimizer,sparse_optimizer,batchid,visualize,_X,_pix,_dropout,_bgscale,bgscale,batchsize,num_classes,predcounts,meta_hist,bg_hist,pc_cols,cats,gtpname,phase='pool',img_s=224):
    '''
    Test the various checkpoints indefinitely, adding visualizations and to the sql tables and to the confusion matrix.
    '''
    log_lock = mp.Lock()
    pos_accurate,neg_accurate,meta_accurate = is_accurate
    cls_tsv = open(f'{sess_dir}/cls_stats.tsv','a')
    while True:
        mc = pd.DataFrame(meta_hist)
        mc.to_hdf(f'{args.cache_dir}/meta_hist.hdf','root')
        checkpoints = []
        if hyperparams.sparsity_model[0] == 'afterwards':
            phase = random.choice(['baseline-extra','extra','pool'])
            for x in os.listdir(modeldir):
                m = re.match(phase+"-(\d+).hdf",x)
                if m is not None:
                    checkpoints.append(int(m.group(1)))
            num_main_iters = random.choice(checkpoints)
            hdf_restore(parameters[0],parameters[1],modeldir,num_main_iters,sess,where=test_all)
        else:
            for x in os.listdir(modeldir):
                m = re.match('(\d+).hdf',x)
                if m is not None:
                    checkpoints.append(int(m.group(1)))
            num_main_iters = random.choice(checkpoints)
            hdf_restore(parameters[0],parameters[1],modeldir,num_main_iters,sess,where=test_all)
        for i in range(args.num_test_batches):
            log_ps = []
            if args.synchronous:
                X,gt,prop_gt_bg,_ = read_dense_test('all')
            else:
                X,gt,prop_gt_bg,_ = queue['test'].get()
            # LOG ACCURACY AND CONFUSION AND ACTIVE_CORR
            feed_acc = {_X : X, _pix : gt, _dropout : 1.0, _bgscale : bgscale}
            if tf.__version__.split('.')[0] == '1':
                outfull,uncertainty_pixelwise_preds_v,diff_v,net_uncertainty,posaccuracy,negaccuracy,metaaccuracy,loss_v = sess.run([outdict['net'],uncertainty_pixelwise_preds,diff,outdict['net_active'],pos_accurate,neg_accurate,meta_accurate,loss],feed_dict=feed_acc)
            elif tf.__version__.split('.')[0] == '0':
                outfull,uncertainty_pixelwise_preds_v,net_uncertainty,posaccuracy,negaccuracy,metaaccuracy,loss_v,summary = sess.run([outdict['net'],uncertainty_pixelwise_preds,outdict['net_active'],pos_accurate,neg_accurate,meta_accurate,loss,merged],feed_dict=feed_acc)
                train_writer.add_summary(summary,batchid)
            ploss_v = -1 * (diff_v - uncertainty_pixelwise_preds_v)
            avg_uncertainty,avg_diff,avg_ploss = np.mean(uncertainty_pixelwise_preds_v),np.mean(diff_v),np.mean(ploss_v)
            print(f"average uncertainty: {avg_uncertainty}, average meta correctness: {avg_diff}, average real loss: {avg_ploss}")
            meta_hist.append({'timestep' : num_main_iters,'average uncertainty' : avg_uncertainty,'average meta correctness' : avg_diff,'avg real loss' : avg_ploss,'phase' : phase})
            pixprobs = normalize_unscaled_logits(outfull).reshape((batchsize,img_s,img_s,-1))
            if hyperparams.active_network in hyperparams.uncertainty_types:
                pacc,nacc,macc= np.mean(posaccuracy),np.mean(negaccuracy),np.mean(metaaccuracy)
                print(f"posaccuracy={pacc},negaccuracy={nacc},metaaccuracy={macc}")
                q_acc = lambda x:f"INSERT INTO fullyconv_joint VALUES('{nickname}',{trial},{batchid},'{phase}',{x},{loss_v},{len(X)},{pacc},{nacc},{macc},{hyperparams.numfuse})"
                dosql(xplatformtime(q_acc),hyperparams,whichdb="postgres")
                #log_confusion(log_lock,batchsize,nickname,pred,gt,num_classes,gtprop,cats,trial,batchid,pixprobs,uncertainty_pixelwise_preds_v,net_uncertainty,hyperparams.numfuse,cls_tsv)
                try:
                    log_ps.append(mp.Process(target=log_confusion,args=(log_lock,batchsize,nickname,gt,num_classes,gtprop,cats,trial,batchid,pixprobs,uncertainty_pixelwise_preds_v,net_uncertainty,hyperparams.numfuse,cls_tsv)))
                    log_ps[-1].start()
                except:
                    print("failed to start logging confusion")
            feed = {_X : X, _pix : gt, _dropout : 0.4, _bgscale : bgscale}
            pixelwise_preds_v,uncertainty_pixelwise_preds_v,diff_v,imgsin,mask_vals,loss_fg_v,loss_v,loss_reg_v,uncertainty_loss_v,_,_ = sess.run([pixelwise_preds,uncertainty_pixelwise_preds,diff,inimgs,active_mask,lossfg,loss,lossreg,uncertainty_loss,optimizer,active_optimizer], feed_dict=feed)
            num_main_iters += 1
            pred = pixelwise_preds_v = np.squeeze(pixelwise_preds_v)
            try:
                proc = mp.Process(target=active_corr,args=(mask_vals,X,gt,batchnames,num_main_iters,hyperparams))
                proc.start()
            except:
                print("Failed to start active_corr process")
            print(f"loss={loss_v},loss_fg={loss_fg_v},loss_reg={loss_reg_v/loss_v},loss_prop_fg={loss_fg_v/loss_v},loss_uncertainty={uncertainty_loss_v}")
            print("avg(mask_vals)={np.mean(mask_vals)}")
            # DO VISUALIZATIONS.
            print("Starting infrequent visualization")
            feed = {_X : X, _pix : gt, _dropout : 0.3, _bgscale : bgscale}
            try:
                numrand = min(batchsize,2)
                which = random.sample(list(range(len(batchnames))),numrand)
                net_keys = [x for x in outdict.keys() if 'upsample' in x]
                active_keys = [x for x in outdict.keys() if 'active' in x]
                for k in net_keys:
                    out_k = np.squeeze(sess.run(outdict[k],feed)).reshape(batchsize,img_s,img_s,num_classes)
                    for i in which:
                        visualize[0](X[i],mask_vals[i],imgsin[i],out_k[i],batchnames[i],num_main_iters,splitid,hyperparams.numfuse,k)
                for k in active_keys:
                    out_k = np.squeeze(sess.run(outdict[k],feed)).reshape(batchsize,img_s,img_s)
                    for i in which:
                        visualize[0](X[i],mask_vals[i],imgsin[i],out_k[i],batchnames[i],num_main_iters,splitid,hyperparams.numfuse,k)
            except:
                print("IN test_all visualization failed.")

def arch_resnet(X,weights,biases,dropout,num_classes,batchsize,numfuse,alphas=None,di=None):
    pass

def arch_vgg(hyperparams,X,weights,biases,dropout,num_classes,batchsize,numfuse,alphas=None,di=None,img_s=224):
    '''
    The beginning of the architecture is a VGGnet.
    X - (?,img_s,img_s,3)
    weights - a dictionary containing tensorflow variables. It is defined in the initialize function.
    biases - similar, but for biases.
    num_classes - 
    batchsize - here, treated as a constant.
    numfuse - 
    '''
    # use normalized but parameterized scales.
    scales = {}
    if numfuse == 0:
        scales['upsample5'] = 1.0
    elif numfuse == 1:
        if alphas is not None:
            scales['upsample5'] = alphas['upsample5']
            scales['upsample4'] = 1 - scales['upsample5']
        else:
            scales['upsample5'],scales['upsample4'] = 0.5,0.5
    elif numfuse == 2:
        if alphas is not None:
            scales['upsample5'] = alphas['upsample5']
            scales['upsample4'] = alphas['upsample4']
            scales['upsample3'] = 1 - (scales['upsample4'] + scales['upsample5'])
        else:
            scales['upsample5'],scales['upsample3'],scales['upsample4'] = 0.333,0.333,0.333
    conv1_1 = conv2d('conv1_1', X, weights['conv1_1'], biases['conv1_1'])
    conv1_2 = conv2d('conv1_2', conv1_1, weights['conv1_2'], biases['conv1_2'])
    pool1 = max_pool('pool1', conv1_2, k=2)
    norm1 = lrn('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, dropout)
    conv2_1 = conv2d('conv2_1', norm1, weights['conv2_1'], biases['conv2_1'])
    conv2_2 = conv2d('conv2_2', conv2_1, weights['conv2_2'], biases['conv2_2'])
    pool2 = max_pool('pool2', conv2_2, k=2)
    norm2 = lrn('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, dropout)
    conv3_1 = conv2d('conv3_1', norm2, weights['conv3_1'], biases['conv3_1'])
    conv3_2 = conv2d('conv3_2', conv3_1, weights['conv3_2'], biases['conv3_2'])
    conv3_3 = conv2d('conv3_3', conv3_2, weights['conv3_3'], biases['conv3_3'])
    pool3 = max_pool('pool3', conv3_3, k=2)
    norm3 = lrn('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, dropout)
    conv4_1 = conv2d('conv4_1', norm3, weights['conv4_1'], biases['conv4_1'])
    conv4_2 = conv2d('conv4_2', conv4_1, weights['conv4_2'], biases['conv4_2'])
    conv4_3 = conv2d('conv4_3', conv4_2, weights['conv4_3'], biases['conv4_3'])
    pool4 = max_pool('pool4', conv4_3, k=2)
    norm4 = lrn('norm4', pool4, lsize=4)
    norm4 = tf.nn.dropout(norm4, dropout)
    conv5_1 = conv2d('myconv5_1', norm4, weights['myconv5_1'], biases['myconv5_1'])
    conv5_2 = conv2d('myconv5_2', conv5_1, weights['myconv5_2'], biases['myconv5_2'])
    pool5 = max_pool('pool5', conv5_2,k=2)
    norm5 = lrn('norm5', pool5, lsize=4)
    norm5 = tf.nn.dropout(norm5, dropout)
    # 28x28x512 -> 28x28xnum_classes
    narrow5 = conv2d('narrow5',norm5,weights['narrow5'],biases['narrow5'])
    upsampled5 = tf.nn.conv2d_transpose(narrow5,weights['upsample5'],[batchsize,img_s,img_s,num_classes],[1,img_s/7,img_s/7,1],name='upsample5',padding='SAME') + tf.reshape(biases['upsample5'],[1,1,1,num_classes])
    if tf.__version__.split('.')[0] == '1':
        print("Not using summaries")
    elif tf.__version__.split('.')[0] == '0':
        tf.histogram_summary('outbias',biases['upsample5'])
        tf.histogram_summary('upsample_W',weights['upsample5'])
    net = scales['upsample5'] * tf.reshape(upsampled5,[batchsize,img_s * img_s,num_classes])
    outdict = {'upsample5' : upsampled5}
    if numfuse >= 1:
        narrow4 = conv2d('narrow4',norm4,weights['narrow4'],biases['narrow4'])
        upsampled4 = tf.nn.conv2d_transpose(narrow4,weights['upsample4'],[batchsize,img_s,img_s,num_classes],[1,img_s/14,img_s/14,1],name='upsample4',padding='SAME') + tf.reshape(biases['upsample4'],[1,1,1,num_classes])
        outdict['upsample4'] = upsampled4
        net += scales['upsample4'] * tf.reshape(upsampled4,[batchsize,img_s * img_s,num_classes])
    if numfuse == 2:
        narrow3 = conv2d('narrow3',norm3,weights['narrow3'],biases['narrow3'])
        upsampled3 = tf.nn.conv2d_transpose(narrow3,weights['upsample3'],[batchsize,img_s,img_s,num_classes],[1,img_s/28,img_s/28,1],name='upsample3',padding='SAME') + tf.reshape(biases['upsample3'],[1,1,1,num_classes])
        outdict['upsample3'] = upsampled3
        net += scales['upsample3'] * tf.reshape(upsampled3,[batchsize,img_s * img_s,num_classes])
    elif numfuse > 2:
        raise NotImplementedError
    outdict['net'] = net
    return outdict

def pixelwise_variance(maps:List[tf.Tensor]) -> tf.Tensor:
    '''

    '''
    T = tf.pack(maps)
    mu = tf.reduce_mean(T,0)
    vsq = tf.square(T - mu)
    return vsq * vsq

def variance_active_fit( ):
    '''

    '''
        
    
def arch_vgg_ensemble(Xs,weights,biases,dropout,num_classes,batchsize,numfuse,alphas=None,di=None,img_s=224):
    ensemble_num = len(Xs)
    outs = []
    for X in Xs:
        outs.append(arch_vgg(X,weights,biases,dropout,num_classes,batchsize,numfuse,alphas=None,di=None,img_s=224)['net'])
    var = pixelwise_variance(outs)
    return variance_
    
def vgg_head(X,weights,biases,dropout):
    conv1_1 = conv2d('conv1_1', X, weights['conv1_1'], biases['conv1_1'])
    conv1_2 = conv2d('conv1_2', conv1_1, weights['conv1_2'], biases['conv1_2'])
    pool1 = max_pool('pool1', conv1_2, k=2)
    norm1 = lrn('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, dropout)
    conv2_1 = conv2d('conv2_1', norm1, weights['conv2_1'], biases['conv2_1'])
    conv2_2 = conv2d('conv2_2', conv2_1, weights['conv2_2'], biases['conv2_2'])
    pool2 = max_pool('pool2', conv2_2, k=2)
    norm2 = lrn('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, dropout)
    conv3_1 = conv2d('conv3_1', norm2, weights['conv3_1'], biases['conv3_1'])
    conv3_2 = conv2d('conv3_2', conv3_1, weights['conv3_2'], biases['conv3_2'])
    conv3_3 = conv2d('conv3_3', conv3_2, weights['conv3_3'], biases['conv3_3'])
    pool3 = max_pool('pool3', conv3_3, k=2)
    norm3 = lrn('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, dropout)
    conv4_1 = conv2d('conv4_1', norm3, weights['conv4_1'], biases['conv4_1'])
    conv4_2 = conv2d('conv4_2', conv4_1, weights['conv4_2'], biases['conv4_2'])
    conv4_3 = conv2d('conv4_3', conv4_2, weights['conv4_3'], biases['conv4_3'])
    pool4 = max_pool('pool4', conv4_3, k=2)
    norm4 = lrn('norm4', pool4, lsize=4)
    norm4 = tf.nn.dropout(norm4, dropout)
    conv5_1 = conv2d('myconv5_1', norm4, weights['myconv5_1'], biases['myconv5_1'])
    conv5_2 = conv2d('myconv5_2', conv5_1, weights['myconv5_2'], biases['myconv5_2'])
    pool5 = max_pool('pool5', conv5_2,k=2)
    norm5 = lrn('norm5', pool5, lsize=4)
    norm5 = tf.nn.dropout(norm5, dropout)
    return norm3,norm4,norm5

def channel_scales(hyperparams,alphas):
    class_scales,active_scales = {},{}
    if hyperparams.numfuse == 0:
        scales['upsample5'] = 1.0
    elif hyperparams.numfuse == 1:
        if alphas is not None:
            class_scales['upsample5'] = alphas['class']['upsample5']
            class_scales['upsample4'] = 1 - class_scales['upsample5']
            active_scales['upsample5'] = alphas['uncertainty']['upsample5']
            active_scales['upsample4'] = 1 - active_scales['upsample5']
        else:
            class_scales['upsample5'],class_scales['upsample4'] = 0.5,0.5
            active_scales['upsample5'],active_scales['upsample4'] = 0.5,0.5
    elif hyperparams.numfuse == 2:
        if alphas is not None:
            class_scales['upsample5'] = alphas['class']['upsample5']
            class_scales['upsample4'] = alphas['class']['upsample4']
            class_scales['upsample3'] = 1 - (class_scales['upsample4'] + class_scales['upsample5'])
            active_scales['upsample5'] = alphas['uncertainty']['upsample5']
            active_scales['upsample4'] = alphas['uncertainty']['upsample4']
            active_scales['upsample3'] = 1 - (active_scales['upsample4'] + active_scales['upsample5'])
        else:
            class_scales['upsample5'],class_scales['upsample3'],class_scales['upsample4'] = 0.333,0.333,0.333
            active_scales['upsample5'],active_scales['upsample3'],active_scales['upsample4'] = 0.333,0.333,0.333
    return class_scales,active_scales

def emoc_vgg( ):
    class_scales,active_scales = channel_scales()
    norm3,norm4,norm5 = vgg_head(X,weights,biases,dropout)
    if hyperparams.stopgrad:
        norm5_con = tf.stop_gradient(norm5)
        norm4_con = tf.stop_gradient(norm4)
        norm3_con = tf.stop_gradient(norm3)
    else:
        norm5_con = norm5 
        norm4_con = norm4 
        norm3_con = norm3 
    # 28x28x512 -> 28x28xnum_classes
    narrow5 = conv2d('narrow5',norm5,weights['narrow5'],biases['narrow5'])
    narrow5_est = conv2d('narrow5_est',norm5_con,weights['narrow5_est'],biases['narrow5_est'])
    up5_shape = [1,img_s/7,img_s/7,1]
    up4_shape = [1,img_s/14,img_s/14,1]
    up3_shape = [1,img_s/28,img_s/28,1]
    upsampled5 = tf.nn.conv2d_transpose(narrow5,weights['upsample5'],[batchsize,img_s,img_s,num_classes],up5_shape,name='upsample5',padding='SAME') + tf.reshape(biases['upsample5'],[1,1,1,num_classes])
    active = active5 = active_scales['upsample5'] * (tf.nn.conv2d_transpose(narrow5_est,weights['active5'],[batchsize,img_s,img_s,1],up5_shape,name='active5',padding='SAME') + tf.reshape(biases['active5'],[1,1,1,1]))
    if tf.__version__.split('.')[0] == '1':
        print("not using summaries.")
    elif tf.__version__.split('.')[0] == '0':
        tf.histogram_summary('outbias',biases['upsample5'])
        tf.histogram_summary('upsample_W',weights['upsample5'])
    net = class_scales['upsample5'] * tf.reshape(upsampled5,[batchsize,img_s * img_s,num_classes])
    outdict = {'upsample5' : upsampled5,'active5' : active5}
    if numfuse >= 1:
        narrow4 = conv2d('narrow4',norm4,weights['narrow4'],biases['narrow4'])
        narrow4_est = conv2d('narrow4_est',norm4_con,weights['narrow4_est'],biases['narrow4_est'])
        upsampled4 = tf.nn.conv2d_transpose(narrow4,weights['upsample4'],[batchsize,img_s,img_s,num_classes],up4_shape,name='upsample4',padding='SAME') + tf.reshape(biases['upsample4'],[1,1,1,num_classes])
        active4 = tf.nn.conv2d_transpose(narrow4_est,weights['active4'],[batchsize,img_s,img_s,1],up4_shape,name='active4',padding='SAME') + tf.reshape(biases['active4'],[1,1,1,1])
        outdict['upsample4'],outdict['active4'] = upsampled4,active4
        net += class_scales['upsample4'] * tf.reshape(upsampled4,[batchsize,img_s * img_s,num_classes])
        outdict['active4'] = active4
        active += active_scales['upsample4'] * active4
    if numfuse == 2:
        narrow3 = conv2d('narrow3',norm3,weights['narrow3'],biases['narrow3'])
        narrow3_est = conv2d('narrow3_est',norm3_con,weights['narrow3_est'],biases['narrow3_est'])
        upsampled3 = tf.nn.conv2d_transpose(narrow3,weights['upsample3'],[batchsize,img_s,img_s,num_classes],up3_shape,name='upsample3',padding='SAME') + tf.reshape(biases['upsample3'],[1,1,1,num_classes])
        active3 = tf.nn.conv2d_transpose(narrow3_est,weights['active3'],[batchsize,img_s,img_s,1],up3_shape,name='active3',padding='SAME') + tf.reshape(biases['active3'],[1,1,1,1])
        outdict['active3'] = active3
        outdict['upsample3'] = upsampled3
        active += active_scales['upsample3'] * active3
        net += class_scales['upsample3'] * tf.reshape(upsampled3,[batchsize,img_s * img_s,num_classes])
    elif numfuse > 2:
        raise NotImplementedError
    outdict['net'] = net
    # net_active not decided yet.
    outdict['net_active'] = None
    return outdict

 
def uncertainty_vgg(hyperparams,X,weights,biases,dropout,num_classes,batchsize,numfuse,alphas=None,di=None,img_s=224):
    '''
    The beginning of the architecture is a VGGnet.
    X - (?,img_s,img_s,3)
    weights - a dictionary containing tensorflow variables. It is defined in the initialize function.
    biases - similar, but for biases.
    num_classes - depending on dataset,split,and whether to include none.
    batchsize - here, treated as a constant.
    numfuse - 
    '''
    # use normalized but parameterized scales.
    class_scales,active_scales = channel_scales(hyperparams,alphas)
    norm3,norm4,norm5 = vgg_head(X,weights,biases,dropout)
    if hyperparams.stopgrad:
        norm5_con = tf.stop_gradient(norm5)
        norm4_con = tf.stop_gradient(norm4)
        norm3_con = tf.stop_gradient(norm3)
    else:
        norm5_con = norm5 
        norm4_con = norm4 
        norm3_con = norm3 
    # 28x28x512 -> 28x28xnum_classes
    narrow5 = conv2d('narrow5',norm5,weights['narrow5'],biases['narrow5'])
    narrow5_est = conv2d('narrow5_est',norm5_con,weights['narrow5_est'],biases['narrow5_est'])
    up5_shape = [1,img_s/7,img_s/7,1]
    up4_shape = [1,img_s/14,img_s/14,1]
    up3_shape = [1,img_s/28,img_s/28,1]
    upsampled5 = tf.nn.conv2d_transpose(narrow5,weights['upsample5'],[batchsize,img_s,img_s,num_classes],up5_shape,name='upsample5',padding='SAME') + tf.reshape(biases['upsample5'],[1,1,1,num_classes])
    active = active5 = active_scales['upsample5'] * (tf.nn.conv2d_transpose(narrow5_est,weights['active5'],[batchsize,img_s,img_s,1],up5_shape,name='active5',padding='SAME') + tf.reshape(biases['active5'],[1,1,1,1]))
    if tf.__version__.split('.')[0] == '1':
        print("not using summaries.")
    elif tf.__version__.split('.')[0] == '0':
        tf.histogram_summary('outbias',biases['upsample5'])
        tf.histogram_summary('upsample_W',weights['upsample5'])
    net = class_scales['upsample5'] * tf.reshape(upsampled5,[batchsize,img_s * img_s,num_classes])
    outdict = {'upsample5' : upsampled5,'active5' : active5}
    if numfuse >= 1:
        narrow4 = conv2d('narrow4',norm4,weights['narrow4'],biases['narrow4'])
        narrow4_est = conv2d('narrow4_est',norm4_con,weights['narrow4_est'],biases['narrow4_est'])
        upsampled4 = tf.nn.conv2d_transpose(narrow4,weights['upsample4'],[batchsize,img_s,img_s,num_classes],up4_shape,name='upsample4',padding='SAME') + tf.reshape(biases['upsample4'],[1,1,1,num_classes])
        active4 = tf.nn.conv2d_transpose(narrow4_est,weights['active4'],[batchsize,img_s,img_s,1],up4_shape,name='active4',padding='SAME') + tf.reshape(biases['active4'],[1,1,1,1])
        outdict['upsample4'],outdict['active4'] = upsampled4,active4
        net += class_scales['upsample4'] * tf.reshape(upsampled4,[batchsize,img_s * img_s,num_classes])
        outdict['active4'] = active4
        active += active_scales['upsample4'] * active4
    if numfuse == 2:
        narrow3 = conv2d('narrow3',norm3,weights['narrow3'],biases['narrow3'])
        narrow3_est = conv2d('narrow3_est',norm3_con,weights['narrow3_est'],biases['narrow3_est'])
        upsampled3 = tf.nn.conv2d_transpose(narrow3,weights['upsample3'],[batchsize,img_s,img_s,num_classes],up3_shape,name='upsample3',padding='SAME') + tf.reshape(biases['upsample3'],[1,1,1,num_classes])
        active3 = tf.nn.conv2d_transpose(narrow3_est,weights['active3'],[batchsize,img_s,img_s,1],up3_shape,name='active3',padding='SAME') + tf.reshape(biases['active3'],[1,1,1,1])
        outdict['active3'] = active3
        outdict['upsample3'] = upsampled3
        active += active_scales['upsample3'] * active3
        net += class_scales['upsample3'] * tf.reshape(upsampled3,[batchsize,img_s * img_s,num_classes])
    elif numfuse > 2:
        raise NotImplementedError
    outdict['net'] = net
    outdict['net_active'] = active
    return outdict

def mkops_common(X,Xgt,parameters,dropout,num_classes,batchsize,numfuse,bg_scale,hyperparams,alphas=None,di=None,reg_scale=0,arch='vgg',uncertainty=False,img_s=224,reg_const=0.005):
    if hyperparams.opt_t in ['ensemble_uncertainty','ensemble_joint']: #working with ensemble, need a few identical copies of the data.
        assert(type(X) == list and type(X[0]) == tf.Tensor)
    else:
        assert(type(X) == tf.Tensor)
    active_epsilon = 1e-8
    weights,biases = parameters
    archfn = archof(arch,hyperparams.opt_t,hyperparams)
    outdict = archfn(hyperparams,X,weights,biases,dropout,num_classes,batchsize,numfuse,alphas=alphas,di=di)
    gtexists = tf.less(tf.argmax(Xgt,2),num_classes) # 1 as the num_classes position indicates non-existing ground truth (0,num_classes-1) are the classes.
    fg_gt = tf.boolean_mask(Xgt,gtexists)
    bg_gt = tf.boolean_mask(Xgt,tf.logical_not(gtexists))
    fg_out = tf.boolean_mask(outdict['net'],gtexists)
    bg_out = tf.boolean_mask(outdict['net'],tf.logical_not(gtexists))
    # the plan is to define this computational graph just once with active_mask placeholder in case using sparsity.
    # if always, we re-set the active mask to what the model says
    active_mask = tf.nn.relu(outdict['net_active']) + active_epsilon
    active_mask = (active_mask / tf.reduce_sum(active_mask)) * (batchsize * hyperparams.budget * img_s * img_s)
    discrete_mask = tf.less(active_mask,tf.random_uniform(shape=active_mask.get_shape()))
    pos_is_accurate = tf.equal(tf.argmax(fg_out,1),tf.argmax(fg_gt,1))
    neg_is_accurate = tf.equal(tf.argmax(bg_out,1),tf.argmax(bg_gt,1))
    bias_const = 500 
    loss_reg = tf.nn.l2_loss(weights['upsample5']) 
    loss_reg += bias_const * tf.nn.l2_loss(biases['upsample5'])
    loss_reg = reg_const * loss_reg
    if hyperparams.numfuse >= 1:
        loss_reg += tf.nn.l2_loss(weights['upsample4']) 
        loss_reg += bias_const * tf.nn.l2_loss(biases['upsample4'])
    if hyperparams.numfuse >= 2:
        loss_reg += tf.nn.l2_loss(weights['upsample3'])
        loss_reg += bias_const * tf.nn.l2_loss(biases['upsample3'])
    return (pos_is_accurate,neg_is_accurate),fg_gt,bg_gt,fg_out,bg_out,(active_mask,discrete_mask),outdict,loss_reg,gtexists

def distance_from_edge(img_s=224,hist_bins=32):
    r = 0.9*img_s
    dist_bins = np.linspace(-1.0*r,r,hist_bins)
    dist_bins = np.floor(dist_bins).astype(np.int)
    angled_dist_bins = 1.0/math.sqrt(2.0) * dist_bins
    angled_dist_bins = np.floor(angled_dist_bins).astype(np.int)
    zeros = np.zeros(dist_bins.size,dtype=np.int)
    assert(zeros.size == dist_bins.size == angled_dist_bins.size)
    paths = {'right' : np.dstack((dist_bins,zeros)),
             'left' : np.dstack((-1 * dist_bins,zeros)),
             'up' : np.dstack((zeros,dist_bins)),
             'down' : np.dstack((zeros,-1*dist_bins)),
             'right-up' : np.dstack((angled_dist_bins,angled_dist_bins)),
             'left-up' : np.dstack((-1 * angled_dist_bins,angled_dist_bins)),
             'left-down' : np.dstack((-1.0 * angled_dist_bins,-1.0 * angled_dist_bins)),
             'right-down' : np.dstack((angled_dist_bins,-1*angled_dist_bins))}
    for k,v in paths.items():
        paths[k] = np.squeeze(v)
    def d(pixs,y,x):
        min_bin = hist_bins + 1 # will be re-assigned no matter what.
        for k,v in paths.items():
            coords = [y,x] + v
            for i,coord in enumerate(coords):
                try:
                    pixs.ix[coord]
                except: # runs when coord not in the index.
                    if i < min_bin: 
                        min_bin = i
                    break
        return min_bin
    return d
     
def dense_within(imgname,box,hyperparams,img_s=224):
    '''
    Rough assumption : any pixel of main category in box belongs to the object (this may be incorrect in certain cases, but overall okay and hard to work without).
    '''
    name = os.path.split(imgname)[1]
    pixs = readsql(f"select * from pixgt where imgname = '{encode_imgname(name)}' AND y > {box['miny']} AND y < {box['maxy']} AND x > {box['minx']} AND x < {box['maxx']} AND category <> 'None'",hyperparams)
    img_shape = readsql(f"SELECT * FROM imgsize where imgname = '{os.path.splitext(name)[0]}'",hyperparams)
    scale_y,scale_x = img_s/img_shape['height'].ix[0],img_s/img_shape['width'].ix[0]
    pixs['y'] = pixs['y'] * scale_y
    pixs['x'] = pixs['x'] * scale_x
    pixs['y'],pixs['x'] = pixs['y'].apply(round),pixs['x'].apply(round)
    min_y,max_y = pixs['y'].min(),pixs['y'].max()
    min_x,max_x = pixs['x'].min(),pixs['x'].max()
    mean_y,mean_x = int(round((max_y+min_y)/2)),int(round((max_x+min_x)/2))
    pixs_ix = pixs.set_index(['y','x'])
    # assumption pixels near middle have chosen category.
    index = pixs_ix.index.tolist()
    if (mean_y,mean_x) in index:
        cat = pixs_ix.ix[(mean_y,mean_x)].iloc[0]['category']
    elif (mean_y+1,mean_x) in index:
        cat = pixs_ix.ix[(mean_y+1,mean_x)].iloc[0]['category']
    elif (mean_y-1,mean_x) in index:
        cat = pixs_ix.ix[(mean_y-1,mean_x)].iloc[0]['category']
    elif (mean_y,mean_x+1) in index:
        cat = pixs_ix.ix[(mean_y,mean_x+1)].iloc[0]['category']
    elif (mean_y,mean_x-1) in index:
        cat = pixs_ix.ix[(mean_y,mean_x-1)].iloc[0]['category']
    else:
        return None,None,None,False
    pixs = pixs[pixs['category'] == cat].drop_duplicates()
    pts = pixs[['y','x']].sample(min(len(pixs),max(int(0.001 * len(pixs)),50)))
    return pixs,pixs.set_index(['y','x']),cat,pts,True

def active_corr(discrete_mask_v,X,gt,imgnames,tstep,hyperparams,img_s=224,imgdir='/data/aseewald/COCO/train_images'):
    '''
    Tracking correlation of various things, such as size of objects and frequency of object class and closeness to boundary, with sampling probabilities.
    '''
    #
    try:
        t0 = time.time()
        discrete_mask_v = np.squeeze(discrete_mask_v)
        object_size_obs = []
        class_id_obs = []
        distances_obs = []
        d = distance_from_edge()
        for i,gt_img in enumerate(gt):
            if random.random() < 0.7: continue
            try:
                imgname = imgdir + '/' + decode_imgname(imgnames[i]) + ".jpg"
                gt_img = gt_img.reshape((img_s,img_s,gt_img.shape[1]))
                boxes = readsql(f"SELECT * FROM perfect_bbox WHERE imgname = '{imgname}'",hyperparams)
                if len(boxes) == 0:
                    print("Warning zero boxes in active_corr for imgname = ",imgname)
                    continue
                for idx,box in boxes.iterrows():
                    try:
                        pixs,pixs_ix,cat,pts,ok = dense_within(imgname,box,hyperparams)
                    except:
                        continue
                    if ok:
                        area = len(pixs)
                        for pt in pts.values:
                            sample_p = discrete_mask_v[i,pt[0],pt[1]]
                            di = d(pixs_ix,pt[0],pt[1])
                            if di > 0:
                                distances_obs.append((d,sample_p))
                        ys,xs = pixs[['y','x']].values.T
                        # now fixing rounding errors.
                        ys[ys>223] = 223
                        xs[xs>223] = 223
                        avg_sample_p = np.mean(discrete_mask_v[i][ys,xs])
                        class_id_obs.append((cat,avg_sample_p))
                        object_size_obs.append((area,avg_sample_p))
            except:
                continue
        if len(class_id_obs) > 0:
            df_cls = pd.DataFrame(class_id_obs,columns=['category','M'])
            for cat,df in df_cls.groupby('category'):
                dosql(f"INSERT INTO cat_active_obs VALUES('{cat}',{tstep},{df['M'].mean()})",hyperparams)
        if len(object_size_obs) > 0:
            df_size = pd.DataFrame(object_size_obs,columns=['size','M'])
            size_corr = scipy.stats.pearsonr(df_size['size'],df_size['M'])[0]
            dosql(f"INSERT INTO size_active_corr VALUES({tstep},{size_corr})",hyperparams)
        if len(distances_obs) > 0:
            df_dedge = pd.DataFrame(distances_obs,columns=['distance','M'])
            dedge_corr = scipy.stats.pearsonr(df_dedge['distance'],df_size['M'])[0]
            dosql(f"INSERT INTO dedge_active_corr VALUES({tstep},{dedge_corr})",hyperparams)
    except:
        print("active_corr failed")
    print(f"active corr thread done, took {time.time()-t0} seconds")

# These different mkops allow for different regions which may be sampled.
def mkops_ssd(X,Xgt,parameters,dropout,num_classes,batchsize,numfuse,bg_scale,hyperparams,alphas=None,di=None,reg_scale=0,arch='vgg',opt_t='classify',uncertainty=False,img_s=224):
    archfn = archof(arch,opt_t,hyperparams)
    is_accurate,fg_out,bg_out,masks,outdict,loss_reg,gtexists = mkops_common(X,Xgt,parameters,dropout,num_classes,batchsize,numfuse,bg_scale,hyperparams)
    detections = ssd(outdict)
    # average the active_mask within regions.
    return is_accurate,loss,(opt,active_opt),(pred,diff),tf.reshape(tf.argmax(Xgt,2),vis_shape),masks,outdict,loss_fg,loss_reg

def mkops_pixelwise(X,Xgt,parameters,dropout,num_classes,batchsize,numfuse,bg_scale,hyperparams,alphas=None,di=None,reg_scale=0,arch='vgg',opt_t='classify',uncertainty=False,img_s=224):
    is_accurate,fg_gt,bg_gt,fg_out,bg_out,masks,outdict,loss_reg,gtexists = mkops_common(X,Xgt,parameters,dropout,num_classes,batchsize,numfuse,bg_scale,hyperparams)
    (active_mask,discrete_mask) = masks
    # I have the masks but am not using them yet.
    (pos_is_accurate,neg_is_accurate) = is_accurate
    loss_fg = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fg_out,labels=fg_gt))
    loss = loss_fg + (bg_scale * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=bg_out,labels=bg_gt)))
    loss += loss_reg
    # these active variables incorperate the discrete mask.
    flat_discrete_mask = tf.reshape(discrete_mask,gtexists.get_shape())
    fg_active_mask = tf.logical_and(gtexists,flat_discrete_mask)
    bg_active_mask = tf.logical_and(tf.logical_not(gtexists),flat_discrete_mask)
    fg_active_out = tf.boolean_mask(outdict['net'],fg_active_mask)
    fg_active_gt = tf.boolean_mask(Xgt,fg_active_mask)
    bg_active_out = tf.boolean_mask(outdict['net'],bg_active_mask)
    bg_active_gt = tf.boolean_mask(Xgt,bg_active_mask)
    sparse_loss_fg = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fg_active_out,labels=fg_active_gt))
    sparse_loss_bg = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=bg_active_out,labels=bg_active_gt))
    sparse_loss = sparse_loss_fg + bg_scale * sparse_loss_bg
    opt = tf.train.AdamOptimizer(learning_rate=hyperparams.lr,name='main_optimizer').minimize(loss)
    sparse_opt = tf.train.AdamOptimizer(learning_rate=hyperparams.lr,name='sparse_optimizer').minimize(sparse_loss)
    vis_shape = [batchsize,img_s,img_s]
    pred = tf.reshape(tf.argmax(outdict['net'],2),vis_shape)
    archfn = archof(arch,opt_t,hyperparams)
    active_pred = tf.squeeze(outdict['net_active'])
    diff = active_pred - tf.reshape(tf.reduce_sum(tf.abs(Xgt - outdict['net']),2),vis_shape)
    uncertainty_loss = tf.nn.l2_loss(diff)
    active_opt = tf.train.AdamOptimizer(learning_rate=hyperparams.lr,name='active_optimizer').minimize(uncertainty_loss)
    meta_is_accurate = tf.less(diff*diff,0.25) # compare squared distance to some threshold.
    is_accurate = (is_accurate[0],is_accurate[1],meta_is_accurate)
    return is_accurate,(loss,uncertainty_loss),(opt,active_opt,sparse_opt),(pred,active_pred,diff),tf.reshape(tf.argmax(Xgt,2),vis_shape),masks,outdict,loss_fg,loss_reg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',help="classify|joint|uncertainty|joint-ensemble|uncertainty-ensemble")
    parser.add_argument('dataset',help="COCO|")
    parser.add_argument('splitid',type=int,help="which classes are known, see breakdown in datasets.py")
    parser.add_argument('nickname',help="name which models and results will be stored with.")
    parser.add_argument('--rtdir',default="./runtime_stats/greedy")
    parser.add_argument('--cache_dir',default="./cache/greedy")
    parser.add_argument('--model_root',default="./modeldir/greedy")
    parser.add_argument('--log_root',default="./logdir/greedy")
    parser.add_argument('-include_none',default=False,action='store_true',help="None is included as a category.")
    parser.add_argument('-synchronous',default=False,action='store_true')
    parser.add_argument('-arch',default="vgg",help="vgg|")
    parser.add_argument('-batchsize',default=24,type=int)
    parser.add_argument('-trial',default=0,type=int)
    parser.add_argument('-from_scratch',default=0,type=int,help="whether to train from scratch or from checkpoint.")
    parser.add_argument('-trestart',default=-1,type=int,help="timestep to restore training at. -1 meaning at max")
    parser.add_argument('-anticipate_missing',default=True,action='store_false')
    parser.add_argument('-biased_upsampling',default=0)
    parser.add_argument('-device',default="GPU",help="GPU|CPU")
    parser.add_argument('-restart',default=True,type=bool)
    parser.add_argument('-all_data_avail',default=False,action='store_true')
    parser.add_argument('-imgdir',default=None)
    parser.add_argument('-dropout',default=1.0,type=float,help="When action=run, the dropout to use.")
    parser.add_argument('-dropout_iters',default=5,type=float,help="When action=run, the dropout to use.")
    parser.add_argument('-valstep',default=40,type=int,help="period for testing the model.")
    parser.add_argument('-visstep',default=20,type=int,help="period for visualizing results.")
    parser.add_argument('-savestep',default=100,type=int,help="period for saving parameters.")
    parser.add_argument('-infreq_visstep',default=20,type=int)
    parser.add_argument('-biasstep',default=40,type=int,help=" ")
    parser.add_argument('-num_test_batches',default=4,type=int)
    parser.add_argument('-savemethod',default="hdf",type=str)
    args = parser.parse_args()
    hyper = hp.pixelwise_hp[args.nickname]
    sess_id = f'{args.nickname}_{args.dataset}_{args.splitid}_{args.trial}'
    sess_dir = f'{args.rtdir}/{sess_id}'
    t_sess_dir = lambda t: f'{sess_dir}/t/{t}'
    if args.action == 'train':
        train(hyper,args)
    elif args.action == 'uncertainty':
        active_train(args,hyper)
    elif args.action == 'joint':
        joint_train(args,hyper)
    elif args.action == 'test':
        test(args,hyper)
    elif args.action == 'run':
        run(args,hyper)
    elif args.action == 'visualize':
        joint_accuracy(hyper,args.nickname)
        corr_vis(hyper,args.nickname)
