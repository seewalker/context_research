'''
Alex Seewald 2016.
aseewald@indiana.edu

A tensorflow implementation of Learning Multi-Glimpse Attention To Represent Context.

This should be run with python >=3.6 because it uses type annotations to keep track of what things are. Custom type definitions are imported from mytypes.py

This program has an argparse-style command line interface specified in arch_common. Running this program with the '--help' flag will describe the way to use it.

Ways results are stored:
    - in the SQL database specified in the hyperparams object, when table-like.
    - in the args.rtdir. this is meant to be a local, fast directory.
    - in the args.cachedir.
    - 
'''
import tensorflow as tf
import numpy as np
import sqlite3
import pandas as pd
import math
import uuid
from math import sqrt
import time
import psycopg2
import random
import sys
import pickle
import socket
import logging
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict
import subprocess
import deepdish as dd
from skimage.io import imread,imsave
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from skimage import img_as_float
from scipy.misc import imresize
from skimage.color import gray2rgb
from scipy.stats import pearsonr
from colorama import Fore, Back, Style
from tqdm import tqdm
from typing import Dict,Tuple,List
import qualitative
from utils import *
from arch_common import *
from arch_draw import *
from arch_block import *
from arch_visualize import *
from arch_patch import *
import constants
import hyperparams as hp
from mytypes import *

__author__ = "Alex Seewald"

args,hyperparams = arch_args()

# a constant list of images to run things on, for sake of 1to1 comparison.
illust_imgs = ["303389_tv_7643.jpg","304456_tv_6414.jpg","307954_tv_596478.jpg","306866_bicycle_344166.jpg","306219_bicycle_471269.jpg","300499_bicycle_82671.jpg"]

fullimg_group = ["DRAW","block_intensity", "block_blur","patches","above_below","expand"] # the set of ctxops which need the full image as well.
# At this point, all hyperparameters are decided upon.

# GLOBAL VARIABLES
chatty = True
# bad global variable for now.
sigmanet_layers = 'conv'
cname = 'clusterspec.pkl'
# hopefully someday this can be tf.float16
PRECISION = tf.float32

# uniquely identifies training session as string suffix.
sess_id = f'{args.train_dataset}_{args.nickname}_{args.splitid}_{args.trial}'
sess_dir = f'{args.rtdir}/arch/{sess_id}'
modeldir = f'{args.model_root}/arch/{sess_id}'
logdir = f'{args.log_root}/arch/{sess_id}'
# uniquely identifies training session at a particular time as string suffix.
t_sess_dir = lambda t: f'{sess_dir}/t/{t}'
for d in {sess_dir,modeldir,logdir,f'{sess_dir}/t'}:
    maybe_mkdir(d)

# Set up logging globally, so that all functions can use it.
logFormatter = logging.Formatter("%(asctime)s %(message)s")
rootLogger = logging.getLogger()
fileHandler = logging.FileHandler(f'{logdir}/log.txt')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.DEBUG)


#for sake of determinism.
tf.set_random_seed(1)

def create_tables():
    stmts = [
    "CREATE TABLE IF NOT EXISTS salsigma_corr(nickname TEXT,splitid INT, tstep INT, trial INT, corr FLOAT, pval FLOAT,ctxop TEXT)",
    "CREATE TABLE IF NOT EXISTS clsvecs(timestep INT, walltime DATE,nickname TEXT, splitid TEXT,vec TEXT, gt TEXT)",
    "CREATE TABLE IF NOT EXISTS attentionvals (nickname TEXT, timestep INT, type TEXT, filtid INT, summaryimgname TEXT, raw FLOAT, raw_rel FLOAT, readable FLOAT, readable_rel FLOAT)",
    "CREATE TABLE IF NOT EXISTS nickname_lookup(nickname TEXT,pickle_payload TEXT,PRIMARY KEY (nickname))",
    "CREATE TABLE IF NOT EXISTS loss(timestep INT, walltime DATE,seen INT, loss_type TEXT, loss_amount FLOAT, samples INT,nickname TEXT,split TEXT, FOREIGN KEY (nickname) REFERENCES nickname_lookup(nickname))",
    "CREATE TABLE IF NOT EXISTS correlation(timestep INT, walltime DATE,corrwith TEXT,seen INT, channel INT, sigmacorr FLOAT, pval FLOAT, samples INT,nickname TEXT,split TEXT, FOREIGN KEY (nickname) REFERENCES nickname_lookup(nickname))",
    # features due to trained networks.
    "CREATE TABLE IF NOT EXISTS arch_ctx_reprs(timestep INT, walltime DATE,imgname TEXT,isperfect INT,iseven INT,canid INT,repr TEXT,split TEXT,category TEXT, nickname TEXT,FOREIGN KEY (nickname) REFERENCES nickname_lookup(nickname))",
    # features due to baselines.
    "CREATE TABLE IF NOT EXISTS raw_ctx_reprs(nickname TEXT,imgname TEXT, canid INT,repr TEXT,category TEXT, type TEXT, FOREIGN KEY (nickname) REFERENCES nickname_lookup(nickname))",
    "CREATE TABLE IF NOT EXISTS pairloss(timestep Int, walltime DATE, seen INT, loss_amount FLOAT, samples INT, nickname TEXT, split TEXT, cat_a TEXT, cat_b TEXT)"
]
    for stmt in stmts:
        dosql(stmt,hyperparams)

# need to fix this.
def possibly_restart_quant() -> Tuple[np.ndarray,np.ndarray]:
    '''

    '''
    try:
        sal_hists = dd.io.load(f'{sess_dir}/saliency.hdf')
    except:
        sal_hists = []
        logging.warn("Failed to load previous saliency data.")
    try:
        closest_hists = dd.io.load('{sess_dir}/closest-quant.hdf')
    except:
        logging.warn("Failed to load previous closest object data.")
        closest_hists = []
    try:
        iou_hists = pickle.load('{sess_dir}/quant.hdf')
    except:
        logging.warn("Failed to load previous intersection over union data.")
        iou_hists = []
    return sal_hists,closest_hists,iou_hists

def data_multiplier(hyperparams):
    if hyperparams.ctxop == "above_below":
        return 2
    elif hyperparams.ctxop == "patches":
        return 3
    elif hyperparams.ctxop == "DRAW3D":
        return hyperparams.frame_budget
    elif hyperparams.ctxop in {"DRAW",'block_blur','block_intensity',None}: #none means vanilla.
        return 1
    else:
        assert(False), f"you are using ctxop={hyperparams.ctxop} which wasn't expected"
    
def train(args:argparse.Namespace,hyperparams,savemethod="tf",on_curriculum=[],check_init=False,restart_num=None,dataset='COCO',img_s=224):
    '''
    '''
    splitid,batchsize,rtdir = args.splitid,args.batchsize,args.rtdir
    if args.split_type == "random": split = hyperparams.possible_splits[splitid]
    elif args.split_type == "clusterbased": split = coco_clust[splitid]
    else:
        logging.warn("Unknown split_type, see command line help.")
        return False
    create_tables()
    maybe_mkdir(sess_dir)
    try:    
        pickle.dump((hyperparams,args),open(f'{sess_dir}/nickname_hyperparams.pkl','wb'))
    except:
        logging.warn("Failed to write nickname hyperparameter log.")
    nclasses,nickname = len(split['known']),args.nickname
    try:
        val_candidates = pd.read_hdf(hyperparams.root("objectness/objness_chosen.hdf"),'root')
    except:
        logging.warn("Not going to have detected candidates, which isn't that important for training just for auxilliary stats.")
    maybe_mkdir(modeldir)
    summary = bool(args.summary)
    bias_history = [] # append dictionaries here and eventually make it a data frame.
    # loss_amounts will store (euclidean_loss,contrastive_loss,classification_loss).
    corr_history,loss_amounts = [], []
    saldistrs, prop_black = [], []
    # whether or not to artifically center depends on whether model can incorperate bbox info.
    needs_centering = hyperparams.ctxop in ['block_blur','block_intensity']
    if not args.cpu_only:
        if args.gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        else:
            assert('CUDA_VISIBLE_DEVICES' in os.environ.keys()), "CUDA_VISIBLE_DEVICES needs to be set as an environment variable."
    session = tf.Session()
    devices = ['/cpu:0'] if args.cpu_only else ['/gpu:'+str(i) for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]
    with session as sess:
        md = data_multiplier(hyperparams)
        # 3 channels for color.
        X_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        Xp_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="Xp")
        Xfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        Xpfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        Xcfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        Xcpfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        # y is whether they are equal.
        y_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize],name="eq")
        ya_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize,nclasses],name="ya")
        yb_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize,nclasses],name="yb")
        bbox_loc = tf.placeholder(dtype=PRECISION,shape=[md*batchsize,4],name="bbox_loc")
        bboxp_loc = tf.placeholder(dtype=PRECISION,shape=[md*batchsize,4],name="bboxp_loc")
        dropout = tf.placeholder(tf.float32,name="dropout")
        parameters = initialize(hyperparams.M,args,hyperparams,nclasses,initialization=hyperparams.initialization)
        sess.run(tf.initialize_all_variables())
        if hyperparams.isvanilla:
            loss,euclid,optimizer,out1,out2 = netops_vanilla(hyperparams,parameters,X_placeholder,Xp_placeholder,y_placeholder,ya_placeholder,yb_placeholder,dropout)
        else:
            debug_X = np.repeat(imread_wrap(hyperparams.root("debug/test.jpg"),hyperparams,tmp=True).reshape((1,img_s,img_s,3)),batchsize,0)
            debug_Xp = copy.deepcopy(debug_X)
            # transpose some channels to get some difference when debugging.
            debug_Xp[:,:,:,0],debug_Xp[:,:,:,1] = debug_X[:,:,:,1],debug_X[:,:,:,0]
            debug_Xfull = np.repeat(imread_wrap(hyperparams.root("debug/test.jpg"),hyperparams,tmp=True).reshape((1,img_s,img_s,3)),batchsize,0)
            debug_Xpfull = copy.deepcopy(debug_Xfull)
            debug_Xpfull[:,:,:,0],debug_Xpfull[:,:,:,1] = debug_Xfull[:,:,:,1],debug_Xfull[:,:,:,0]
            debug_items = {'session' : sess,'feed' : {X_placeholder : debug_X, dropout : 1.0,
                           Xfull_placeholder : debug_Xfull,
                           Xp_placeholder : debug_Xp,
                           Xpfull_placeholder : debug_Xpfull,
                           Xcfull_placeholder : debug_Xfull, #not realistic, this is just for non-empty debug info.
                           Xcpfull_placeholder : debug_Xpfull,
                           y_placeholder: np.ones(batchsize,dtype=np.float32),
                           ya_placeholder: onehot(np.ones(batchsize),nclasses),
                           yb_placeholder: onehot(np.ones(batchsize),nclasses),
                           dropout: 1.0, bboxp_loc : np.array(batchsize * [[30,200,10,100]]),
                           bbox_loc : np.array(batchsize * [[30,200,10,100]])}}
            loss_tup,euclid,optimizer,attention,pre,pre_full,post,out1,out2,filt,gvs = netops(parameters,X_placeholder,Xp_placeholder,Xfull_placeholder,Xpfull_placeholder,Xcfull_placeholder,Xcpfull_placeholder,bbox_loc,bboxp_loc,y_placeholder,ya_placeholder,yb_placeholder,dropout,batchsize,hyperparams,args,debug_items=debug_items,devices=devices)
            # depending on loss_t some of these might be undefined.
            loss,loss_contrastive,loss_classify = loss_tup
            if hyperparams.ctxop == 'DRAW':
                attentionvals,attention_boxes = attention
                o1,o2 = sess.run([out1['similarity'],out2['similarity']],debug_items['feed'])
                logging.info(f"Difference on identical o1-o2 = {np.linalg.norm(o2-o1) / np.product(o1.shape)}, vs norm(o1)={np.linalg.norm(o1)}")
            else:
                attentionvals = None
            if hyperparams.ctxop in ['block_intensity','block_blur']:
                sigmas1out,sigmas2 = attention
        saver = tf.train.Saver(max_to_keep=12)
        ckpt = tf.train.get_checkpoint_state(modeldir)
        if (not args.from_scratch) and os.path.exists(modeldir):
            if savemethod == "tf":
                t0 = int(ckpt.model_checkpoint_path.split("-")[-1])
                saver.restore(sess,ckpt.model_checkpoint_path)
                logging.info(f"RESTARTING FROM step={t0}")
                # In case there are uninitialized variables...
            elif savemethod == "hdf":
                if restart_num == None:
                    logging.info("Loading from max")
                    avail = [int(x.split(".")[0]) for x in os.listdir(modeldir)]
                    assert(len(avail) > 0), "Not from scratch and no checkpoint to continue from."
                    t0 = max(avail)
                else:
                    t0 = restart_num
                restartname = modeldir + "/" + str(t0) + ".hdf"
                try:
                    npy_w,npy_b = dd.io.load(restartname)
                except:
                    logging.warn(f"Failed to load from hdf {restartname}")
                    sys.exit(1)
                global_step = tf.Variable(t0)
                sess.run(tf.initialize_all_variables())
                for k in parameters[0].keys():
                    sess.run(parameters[0][k].assign(npy_w[k]))
                for k in parameters[1].keys():
                    sess.run(parameters[1][k].assign(npy_b[k]))
                logging.info(f"Successfully restarted from parameters at {t0}")
            sal_hists,closest_hists,iou_hists = possibly_restart_quant(nickname,splitid,args.trial)
        else:
            t0 = 0
            sal_hists,closest_hists,iou_hists = [], [],[]
            global_step = tf.Variable(t0)
            sess.run(tf.initialize_all_variables())
            if not os.path.exists(f'{logdir}/graph.pbtxt'):
                logging.info("Writing graph")
                #tf.train.write_graph(sess.graph,logdir,'graph.pbtxt')
                logging.info("Done writing graph")
        if check_init:
            logging.info("Checking if any uninitialized variables")
            uninitialized_vars = determine_uninitialized(sess)
            if len(uninitialized_vars) > 0:
                sess.run(tf.intialize_variables(uninitialized_vars))
                logging.info("Initialized previously uninitialized:", [v.name for v in uninitialized_vars])
            else:
                logging.info("NO UNINITIALIZED VARS")
            assert(checknan(sess,parameters))
        if summary:
            try:
                logging.info("MAKING SUMMARY OPS")
                tf.scalar_summary("Loss",loss)
                for k in parameters[0].keys():
                    tf.histogram_summary("w_" + k,parameters[0][k])
                for k in parameters[1].keys():
                    tf.histogram_summary("b_" + k,parameters[1][k])
                summary_op = tf.merge_all_summaries() #at this point all summaries have been declared, the histogram summaries here and some in netops.
                writer = tf.train.SummaryWriter(logdir,sess.graph,flush_secs=360)
            except:
                logging.warn("Problems with summary ops, to be expected with tensorflow > 1 because everything got changed and it's not that important.")
            #writer.add_graph()
        if socket.gethostname() == hp.mainhost: existing_samples = len(os.listdir(hyperparams.root('train_patches')))
        else: existing_samples = 6e5 #an approximation, it takes long to do such large listdir over sshfs.
        num_samples = args.num_epochs * existing_samples
        sanitystep,summarystep,teststep,imgstep = 500 // sqrt(batchsize),400 // sqrt(batchsize),400 // sqrt(batchsize),450 // sqrt(batchsize)
        savestep,bias_savestep,quantify_step = 300,50,10
        num_batches = int(num_samples / batchsize)
        if not hyperparams.isvanilla:
            sanitystep *= 2
            summarystep *= 2
            teststep *= 2
        prop_zero,prev_repr = {'sampled' : [],'const' : [],'compare' : []},None
        eq_hist,grads,diffs,gt = [],[],[],[]
        if hyperparams.dataset_t == 'image':
            sample_fn = sample_img(batchsize,splitid,hyperparams,variety="train_normal",full_img=(hyperparams.ctxop in fullimg_group))
            try:
                sample_fn_sal = sample_img(1,splitid,hyperparams,include_saliency=True,val_candidates=val_candidates,full_img=(hyperparams.ctxop in fullimg_group))
            except:
                logging.warn("Saliency batches not available.")
                sample_fn_sal = None
        elif hyperparams.dataset_t == 'video':
            sample_fn = sample_video(batchsize,splitid,hyperparams)
            sample_fn_sal = None
        logging.info("ABOUT TO START TRAINING")
        logging.info("Lines will be printed of the form:\ngreater equal accuracy by {}. last10:{},last100:{}")
        logging.info("Greater equal accuracy is accuracy on equal pairs compared to not equal pairs. last10 and last100 are averages of this value over the previous 10 or 100 iterations.")
        logging.info("Lines will be printed of the form:\no1sum={},o2-o1={},corr100={},corr20={}")
        logging.info("To determine whether outputs simply approach 0 (as was previously seen),measure output vector magnitude and difference magnitude as no1sum,o2-o1 respectively.")
        logging.info("corr100 and corr20 are correlation between distance and whether ground truth are equal. Naturally, if things are going well, this is a strong negative correlation.")

        # this gets run just once, it is the random sample data that will be tracked over time.
        compare_file = f'{args.cachedir}/compare_data_{splitid}_{hyperparams.ctxop in fullimg_group}.hdf'
        if hyperparams.ctxop in fullimg_group:
            X_const,Xfull_const,bboxs_const,y_const,imgnames_const = sample_fn(specified=illust_imgs)
            Xp_const,Xpfull_const,bboxsp_const,yp_const,imgnamesp_const = sample_fn(tomatch=y_const)
            Xcfull_const,alt_bboxs_const = center_transform(Xfull_const,imgnames_const,bboxs_const)
            Xcpfull_const,alt_bboxsp_const = center_transform(Xpfull_const,imgnamesp_const,bboxsp_const)
            equal_const = (y_const == yp_const).astype(np.float32)
            feed_const = {X_placeholder : X_const, Xp_placeholder: Xp_const, y_placeholder : equal_const,Xcfull_placeholder : Xcfull_const, Xcpfull_placeholder : Xcpfull_const, dropout : 1.0,Xfull_placeholder : Xfull_const, Xpfull_placeholder : Xpfull_const, bbox_loc : bboxs_const, bboxp_loc : bboxsp_const}
            feed_const = refine_feed(hyperparams,feed_const,X_placeholder,Xp_placeholder,Xfull_placeholder,Xpfull_placeholder,Xcfull_placeholder,Xcpfull_placeholder,X_const,Xp_const,Xfull_const,Xpfull_const,Xcfull_const,Xcpfull_const,y_placeholder,equal_const,ya_placeholder,yb_placeholder,None,None,bboxs_const,bboxsp_const)
            if not os.path.exists(compare_file):
                X_compare,Xfull_compare,bboxs_compare,y_compare,imgnames_compare = sample_fn(specified=illust_imgs)
                Xp_compare,Xpfull_compare,bboxsp_compare,yp_compare,imgnamesp_compare = sample_fn(tomatch=y_const)
                Xcfull_compare,alt_bboxs_compare = center_transform(Xfull_const,imgnames_const,bboxs_const)
                Xcpfull_compare,alt_bboxsp_compare = center_transform(Xpfull_const,imgnamesp_const,bboxsp_const)
                equal_compare = (y_compare == yp_compare).astype(np.float32)
                tup_compare = X_compare,Xfull_compare,bboxs_compare,y_compare,imgnames_compare,Xp_compare,Xpfull_compare,bboxsp_compare,yp_compare,imgnamesp_compare,Xcfull_compare,alt_bboxs_compare,Xcpfull_compare,alt_bboxsp_compare,equal_compare 
                try:
                    dd.io.save(compare_file,tup_compare)
                except:
                    assert False, "Failed to save compare data."
            else:
                X_compare,Xfull_compare,bboxs_compare,y_compare,imgnames_compare,Xp_compare,Xpfull_compare,bboxsp_compare,yp_compare,imgnamesp_compare,Xcfull_compare,alt_bboxs_compare,Xcpfull_compare,alt_bboxsp_compare,equal_compare = dd.io.load(compare_file)
            feed_compare = {X_placeholder : X_compare, Xp_placeholder: Xp_compare, y_placeholder : equal_compare,Xcfull_placeholder : Xcfull_compare, Xcpfull_placeholder : Xcpfull_compare, dropout : 1.0,Xfull_placeholder : Xfull_compare, Xpfull_placeholder : Xpfull_compare, bbox_loc : bboxs_compare, bboxp_loc : bboxsp_compare}
            feed_compare = refine_feed(hyperparams,feed_compare,X_placeholder,Xp_placeholder,Xfull_placeholder,Xpfull_placeholder,Xcfull_placeholder,Xcpfull_placeholder,X_compare,Xp_compare,Xfull_compare,Xpfull_compare,Xcfull_compare,Xcpfull_compare,y_placeholder,equal_compare,ya_placeholder,yb_placeholder,None,None,bboxs_compare,bboxsp_compare)
        else:
            if not os.path.exists(compare_file):
                X_compare,bboxs_compare,y_compare,imgnames_compare = sample_fn(specified=illust_imgs)
                Xp_compare,bboxsp_compare,yp_compare,imgnamesp_compare = sample_fn(tomatch=y_const)
                equal_compare = (y_compare == yp_compare).astype(np.float32)
                tup_compare = X_compare,bboxs_compare,y_compare,imgnames_compare,Xp_compare,bboxsp_compare,yp_compare,imgnamesp_compare,equal_compare
                try:
                    dd.io.save(compare_file,tup_compare)
                except:
                    assert False, "Failed to save compare data."
            else:
                X_compare,Xfull_compare,bboxs_compare,y_compare,imgnames_compare,Xp_compare,Xpfull_compare,bboxsp_compare,yp_compare,imgnamesp_compare,Xcfull_compare,alt_bboxs_compare,Xcpfull_compare,alt_bboxsp_compare,equal_compare = dd.io.load(compare_file)
            X_const,y_const,imgnames_const = sample_fn()
            Xp_const,yp_const,imgnamesp_const = sample_fn(tomatch=y_const)
            equal_const = (y_const == yp_const).astype(np.float32)
            feed_const = {X_placeholder : X_const, Xp_placeholder: Xp_const, y_placeholder : equal_const,dropout : 1.0}
            bboxs_const,Xfull_const = None,None # need the name bound to something for pre_post call.
        if hyperparams.ctxop in ["patches","above_below"]:
            del(feed_const[bbox_loc])
            del(feed_const[bboxp_loc])
        # check if I need to initialize loss_amounts with previous history
        if t0 > 0:
            try:
                loss_history = pd.read_hdf("{sess_dir}/loss_history.hdf","root").values
                for lh in loss_history:
                    loss_amounts.append(tuple(lh.tolist()))
            except:
                logging.warning("Couldn't read previous loss_history.")
        for t in range(t0,(num_batches-t0)):
            if hyperparams.ctxop in fullimg_group:
                # format of bboxs here: ymin,ymax,xmin,xmax
                X,Xfull,bboxs,y,imgnames = sample_fn()
                Xp,Xpfull,bboxsp,yp,imgnamesp = sample_fn(tomatch=y)
                Xcfull,alt_bboxs = center_transform(Xfull,imgnames,bboxs)
                Xcpfull,alt_bboxsp = center_transform(Xpfull,imgnames,bboxsp)
            else:
                X,y,imgnames = sample_fn()
                Xp,yp,imgnamesp = sample_fn(tomatch=y)
            yint,ypint = [split['known'].index(yi) for yi in y], [split['known'].index(yi) for yi in yp] #convert from
            equal = (y == yp).astype(np.float32)
            logging.info(f"***************t={t},equal={str(equal)}***************")
            if hyperparams.ctxop in fullimg_group:
                feed = {X_placeholder : X, Xp_placeholder: Xp, y_placeholder : equal,dropout : hyperparams.dropout, ya_placeholder : onehot(yint,nclasses), yb_placeholder : onehot(ypint,nclasses),Xfull_placeholder : Xfull, Xpfull_placeholder : Xpfull, Xcfull_placeholder : Xcfull, Xcpfull_placeholder : Xcpfull, bbox_loc : bboxs, bboxp_loc : bboxsp}
                feed = refine_feed(hyperparams,feed,X_placeholder,Xp_placeholder,Xfull_placeholder,Xpfull_placeholder,Xcfull_placeholder,Xcpfull_placeholder,X,Xp,Xfull,Xpfull,Xcfull,Xcpfull,y_placeholder,equal,ya_placeholder,yb_placeholder,onehot(yint,nclasses),onehot(ypint,nclasses),bboxs,bboxsp)
            else:
                feed = {X_placeholder : X, Xp_placeholder: Xp, y_placeholder : equal,dropout : hyperparams.dropout, ya_placeholder : onehot(yint,nclasses), yb_placeholder : onehot(ypint,nclasses)}
            if hyperparams.ctxop in ["patches","above_below"]:
                del(feed[bbox_loc])
                del(feed[bboxp_loc])
            if (t % quantify_step == 0):
                if hyperparams.ctxop == "DRAW":
                    qout = quantify_patches(hyperparams,X,Xfull,imgnames,bboxs,post,nickname,splitid)
                    sal_hists = sal_hists + qout['saliency']
                    closest_hists = closest_hists + qout['closest']
                    iou_hists = iou_hists + qout['iou']
                    dd.io.save(f'{sess_dir}/saliency.hdf',sal_hists)
                    dd.io.save(f'{sess_dir}/closest.hdf',closest_hists)
                    dd.io.save(f'{sess_dir}/iou.hdf',iou_hists)
                elif hyperparams.ctxop in ['block_intensity','block_blur']:
                    try:
                        props = block_properties(np.array(sess.run(sigmas1out[0],feed)),Xfull,splitid,hyperparams.M)
                        corr,pval = props['saliency_corr']
                        dosql(f"INSERT INTO salsigma_corr VALUES('{nickname}',{splitid},{t},{args.trial},{corr},{pval},'{hyperparams.ctxop}')",hyperparams)
                        saldistrs.append(props['distr'])
                        pickle.dump(saldistrs,open(f'{t_sess_dir(t)}/salsigma.pkl','wb'))
                    except:
                        logging.warn("Failed on block properties")
                else:
                    logging.info("No visualization to do")
            if (t % imgstep == 0) and not hyperparams.isvanilla: 
                # run visualizations both on const data and whatever the batch happens to be.
                prop_zero['compare'].append(t,plot_qualitative(sess,filt,post,bboxs_compare,bboxs_compare,feed_compare,t,splitid,nickname,Xfull_compare,parameters,imgnames_compare,attentionvals,hyperparams,args.trial,t_sess_dir,hyperparams.include_center,data_t='compare'))
                prop_zero['const'].append(t,plot_qualitative(sess,filt,post,bboxs_const,bboxs_const,feed_const,t,splitid,nickname,Xfull_const,parameters,imgnames_const,attentionvals,hyperparams,args.trial,t_sess_dir,hyperparams.include_center,data_t='const'))
                prop_zero['sampled'].append(t,plot_qualitative(sess,filt,post,bboxs,bboxs,feed,t,splitid,nickname,Xfull,parameters,imgnames,attentionvals,hyperparams,args.trial,t_sess_dir,hyperparams.include_center,data_t='sampled'))
            if hyperparams.loss_t == "dual":
                o1,o2,o1class,o2class,loss_amount,loss_classify_amount,loss_contrastive_amount,euclid_amount,gvs = sess.run([out1['similarity'],out2['similarity'],out1['classify'],out2['classify'],loss,loss_classify,loss_contrastive,euclid,optimizer],feed_dict = feed)
                loss_amounts.append((loss_amount,loss_classify_amount,loss_contrastive_amount,np.mean(euclid_amount),t))
            elif hyperparams.loss_t == 'contrastive':
                o1,o2,loss_amount,euclid_amount,gvs = sess.run([out1['similarity'],out2['similarity'],loss,euclid,optimizer],feed_dict = feed)
                loss_amounts.append((loss_amount,np.mean(euclid_amount),t))
            if prev_repr is not None and np.array_equal(o1==0,prev_repr==0) and np.sum(o1==0) > 0:
                logging.critical("Fell into the always-zero-at-certain-location situation.")
            if np.array_equal(o1 == 0,o2 == 0) and np.sum(o1==0) > 0:
                logging.critical("Fell into the always-zero-at-certain-location situation.")
            prev_repr = o1
            if t % 50 == 1:
                # check for nan values.
                for k in parameters[0].keys():
                    nans = sess.run(tf.is_nan(parameters[0][k]))
                    if np.sum(nans) > 0:
                        logging.warn(f"Warning for weights, {np.mean(nans)} of {k} values are NaN")
                for k in parameters[1].keys():
                    nans = sess.run(tf.is_nan(parameters[1][k]))
                    if np.sum(nans) > 0:
                        logging.warn(f"Warning for biases, {np.mean(nans)} of {k} values are NaN")
            o1,o2 = np.squeeze(o1),np.squeeze(o2)
            accuracy_amount = euclid_amount < 0.25 #the halfway point.
            try:
                logging.info(f"nickname={nickname},t={t},loss={loss_amount},accuracy={np.mean(accuracy_amount)},zip(o1-o2,equal)={list(zip(equal,np.linalg.norm(o1 - o2,axis=(1,2))))},avg o1-o2={np.linalg.norm(o1-o2)/batchsize},propzero={np.where(o1==0)[2].size / o1.size}")
            except:
                logging.critical(f"Warning: at timestep {t} there was {np.sum(np.isnan(o1))} NAN output values in o1 and {np.sum(np.isnan(o2))} NAN output values in o2")
            if hyperparams.ctxop == 'DRAW' and t % 10 == 0 and hyperparams.numfilts > 1:
                atvs = {}
                for idx,av in enumerate(attentionvals):
                    if 'filtid' in av.keys(): #should only run once.
                        del(av['filtid'])
                    atvs[idx] = {k : sess.run(av[k],feed) for k in av.keys()}
                for fid,v in atvs.items():
                    vvar = {k : np.var(v[k]) for k in v.keys()}
                    vmean = {k : np.mean(v[k]) for k in v.keys()}
                    logging.info(f"filter={fid},means={vmean},readable_means={draw_rangealt(vmean,report_relative=True)},variances={vvar}")
            #Keep track of whether there is a tendency to overestimate or underestimate similarity.
            eq_acc = accuracy_amount[equal.astype(np.bool)]
            if random.random() < 0.5: #randomize use of > vs >= to avoid bias due to equal predictions.
                eq_hist.append(int(np.mean(eq_acc) > np.mean(accuracy_amount)))
            else:
                eq_hist.append(int(np.mean(eq_acc) >= np.mean(accuracy_amount)))
            tb,th = -1 * min(t,10),-1 * min(t,100)
            mu_eq,mu = np.mean(eq_acc),np.mean(accuracy_amount)
            if mu_eq > mu:
                logging.info(f"greater equal accuracy by {mu_eq - mu}. last10:{np.mean(eq_hist[-1:tb:-1])},last100:{np.mean(eq_hist[-1:th:-1])}")
            else:
                logging.info(f"lesser equal accuracy by {mu_eq - mu}. last10:{np.mean(eq_hist[-1:tb:-1])},last100:{np.mean(eq_hist[-1:th:-1])}")
            if len(o1.shape) == 3:
                o1,o2 = o1[0].flatten(),o2[0].flatten()
            elif len(o1.shape) == 2: #true if batchsize >1
                o1,o2 = o1[0],o2[0]
            try:
                diffs.append(euclidean(o2,o1))
                gt.append(equal[0])
            except:
                logging.critical(f"Warning: at timestep {t} there was are {np.sum(np.isnan(o1))} NAN output values in o1 and {np.sum(np.isnan(o2))} NAN output values in o2")
            if len(gt) > 3:
                slice20 = slice((-1 *min(len(diffs)-1,20)),-1)
                slice100 = slice((-1 *min(len(diffs)-1,100)),-1)
                try:
                    rawstr= f"o1sum={np.sum(np.abs(o1))},o2-o1={euclidean(o2,o1)},corr100={pearsonr(diffs[slice100],gt[slice100])},corr20={pearsonr(diffs[slice20],gt[slice20])}"
                except:
                    logging.critical(f"Warning: at timestep {t} there was are {np.sum(np.isnan(o1))} NAN output values in o1 and {np.sum(np.isnan(o2))} NAN output values in o2")
                if pearsonr(diffs[slice20],gt[slice20])[0] < 0:
                    logging.info(Fore.GREEN + rawstr)
                else:
                    logging.info(Fore.RED + rawstr)
            if summary and (t % summarystep == 0):
                try:
                    summary_str = sess.run(summary_op,feed_dict=feed)
                    writer.add_summary(summary_str,t)
                except:
                    logging.warning("Warning, failed to write summary, probably because of NANs.")
                    continue
            if hyperparams.ctxop == 'DRAW':
                for filtid in range(hyperparams.numfilts-1):
                    key = f'attention_{filtid}'
                    b = sess.run(parameters[1][key])
                    bias_history.append({'t' : t, 'filtid' : filtid,'g_x' : b[0],'g_y' : b[1], 'sigmasq' : b[2], 'stride' : b[3], 'intensity' : b[4]})
            elif hyperparams.ctxop in ['block_intensity','block_blur']:
                if sigmanet_layers in ['mlp_1','mlp_2']:
                    bnames = [x.name.split(":")[0] for x in tf.all_variables() if x.name[0:5] == "sigma"]
                    biases = sess.run(bnames)
                elif sigmanet_layers == 'conv':
                    biases = [] #there isn't really a notion of sigmasq biases here.
                elif sigmanet_layers == 'highlevel':
                    bnames = ['snet']
                    biases = [sess.run(parameters[1]['snet'])]
                for i,bias in enumerate(biases):
                    bias_history.append({'t' : t,'key' : bnames[i], 'nickname' : nickname,'bias' : bias})
            if t % bias_savestep == (bias_savestep - 1):
                pd.DataFrame(bias_history).to_hdf(f'{sess_dir}/attention_bias_history.hdf','root')
            # save the model. if distributed,
            if (t % savestep == savestep - 1) or t == 10:
                col_lab = ('net_loss','classify_loss','contrastive_loss','euclid_loss','t') if hyperparams.loss_t == 'dual' else ('net_loss','euclid_loss','t')
                pd.DataFrame(loss_amounts,columns=col_lab).to_hdf(f'{sess_dir}/loss_history.hdf','root')
                logging.info("Plotted loss and accuracy history")
                ta = time.time()
                logging.info("Beginning snapshot")
                if savemethod in ["tf","both"]:
                    saver.save(sess,f'{modeldir}/model',global_step=t)
                    logging.info(f"Made tensorflow snapshot, taking {time.time() - ta} seconds")
                if savemethod in ["hdf","both"]:
                    w_keys = list(parameters[0].keys())
                    b_keys = list(parameters[1].keys())
                    w_out = sess.run(list(parameters[0].values()))
                    b_out = sess.run(list(parameters[1].values()))
                    weight_snapshot = OrderedDict({k : w_out[w_keys.index(k)] for k in parameters[0].keys()})
                    logging.debug("Done with weight snapshot")
                    bias_snapshot = OrderedDict({k : b_out[b_keys.index(k)] for k in parameters[1].keys()})
                    logging.debug("Done with bias snapshot")
                    dd.io.save(f'{modeldir}/{str(t)}.hdf',(weight_snapshot,bias_snapshot))
                    logging.info(f"Made hdf5 snapshot, taking {time.time() - ta} seconds")
                    dd.io.save(f"{sess_dir}/propzero.hdf",prop_zero)

def test(splitid,nickname,args:argparse.Namespace,step,nclasses,split,numiterations):
    '''
    '''
    t0 = time.time()
    batchsize = 7
    with tf.Session() as sess:
        X_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize, 224,224,3],name="X")
        Xp_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize, 224,224,3],name="Xp")
        Xfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize, 224,224,3],name="X")
        Xpfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize, 224,224,3],name="Xp")
        y_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize],name="eq")
        ya_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize,nclasses],name="ya")
        yb_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize,nclasses],name="yb")
        bbox_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize,4],name="bbox")
        bboxp_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize,4],name="bboxp")
        dropout = tf.placeholder(tf.float32)
        t = step[1]
        weights,biases = dd.io.load(os.path.join(modeldir,step[0]))
        weights = {k : tf.Variable(v,dtype=PRECISION) for k,v in weights.items()}
        biases = {k : tf.Variable(v,dtype=PRECISION) for k,v in biases.items()}
        parameters = (weights,biases)
        if hyperparams.isvanilla:
            outs = netops_vanilla(parameters,X_placeholder,Xp_placeholder,y_placeholder,dropout)
            out1,out2 = outs[3], outs[4]
        else:
            outs = netops(parameters,X_placeholder,Xp_placeholder,Xfull_placeholder,Xpfull_placeholder,bbox_placeholder,bboxp_placeholder,y_placeholder,ya_placeholder,yb_placeholder,dropout,batchsize,hyperparams.relative_arch,args.distinct_decode,loss_t,devices=devices)
            out1,out2 = outs[6], outs[7]
        sess.run(tf.initialize_all_variables())
        loss = outs[0]
        seen_sample = sample_img(N=batchsize,variety="train_normal",splitid=splitid)
        unseen_sample = sample_img(N=batchsize,variety="unseen_normal",splitid=splitid)
        val_sample = sample_img(N=batchsize,variety="test",splitid=splitid) # the real candidates.
        for iteration in range(numiterations):
            if iteration % 10 == 0:
                logging.info(f"nickname={nickname},t={t},iter={iteration / numiterations}")
            isseen = random.choice([0,0,1,2])
            try:
                if isseen == 0:
                    X,Xfull,bboxs,y,imgnames = seen_sample()
                    Xp,Xpfull,bboxsp,yp,imgnamesp = seen_sample(tomatch=y)
                elif isseen == 1:
                    X,Xfull,bboxs,y,imgnames = unseen_sample()
                    Xp,Xpfull,bboxsp,yp,imgnamesp = unseen_sample(tomatch=y)
                elif isseen == 2:
                    X,Xfull,bboxs,y,imgnames = val_sample()
                    Xp,Xpfull,bboxsp,yp,imgnamesp = val_sample(tomatch=y)
            except:
                continue
            equal = (y == yp).astype(np.float32)
            if isseen == 0: #don't do classificiation on unseen classes.
                yint,ypint = [split['known'].index(yi) for yi in y], [split['known'].index(yi) for yi in yp] #convert from 
                feed = {X_placeholder : X, Xp_placeholder : Xp,dropout : hyperparams.dropout,y_placeholder:equal,ya_placeholder:onehot(yint,nclasses),yb_placeholder:onehot(ypint,nclasses)}
            else:
                yint,ypint = None,None #not needed.
                feed = {X_placeholder : X, Xp_placeholder : Xp,dropout : hyperparams.dropout,y_placeholder:equal,ya_placeholder:np.zeros((batchsize,nclasses)),yb_placeholder:np.zeros((batchsize,nclasses))}
            if hyperparams.loss_t == 'dual' and isseen == 0:
                o1sim,o2sim,o1class,o2class,loss_amount = sess.run([out1['similarity'],out2['similarity'],out1['classify'],out2['classify'],loss],feed_dict=feed)
                similarity_loss_amount,slt = pyloss({'similarity' : o1sim},{'similarity' : o2sim},y,yp,onehot(yint,nclasses),onehot(ypint,nclasses),"contrastive")
                classification_loss_amount,clt = pyloss({'classify' : o1class},{'classify' : o2class},y,yp,onehot(yint,nclasses),onehot(ypint,nclasses),"softmax")
                dosql(xplatformtime(lambda x:f"INSERT INTO loss VALUES({t},{x},{isseen},'dual',{loss_amount},{batchsize},'{nickname}','{splitid}')",hyperparams,only_log=True))
                dosql(xplatformtime(lambda x:f"INSERT INTO loss VALUES({t},{x},{isseen},'dual',{similarity_loss_amount},{batchsize},'{nickname}','{splitid}')",hyperparams,only_log=True))
                dosql(xplatformtime(lambda x:f"INSERT INTO loss VALUES({t},{x},{isseen},'dual',{classification_loss_amount},{batchsize},'{nickname}','{splitid}')",hyperparams,only_log=True))
                for lt in slt['contrastive']:
                    dosql(xplatformtime(lambda x: f"INSERT INTO pairloss VALUES ({t},{x},{isseen},{lt[2]},1,'{nickname}','{splitid}','{lt[0]}','{lt[1]}')",hyperparams,only_log=True))
                for lt in clt['classify']:
                    dosql(xplatformtime(lambda x: f"INSERT INTO clsvecs VALUES ({t},{x},'{nickname}','{str(splitid)}','{lt[0]}','{floatserial(lt[1],5)}')"),hyperparams,only_log=True)
                # do custom pyloss of classify and similarity.
            else:
                o1sim,o2sim = sess.run([out1['similarity'],out2['similarity']],feed_dict=feed)
                loss_amount,lt = pyloss({'similarity' : o1sim},{'similarity' : o2sim},y,yp,yint,ypint,"contrastive")
                dosql(xplatformtime(lambda x:f"INSERT INTO loss VALUES({t},{x},{isseen},'similarity',{loss_amount},{batchsize},'{nickname}','{splitid}')"),hyperparams,only_log=True)
            logging.info(f"seen={isseen},loss_amount={loss_amount}")
        logging.info(f"TESTING TOOK {time.time() - t0} SECONDS")

def even_cans(canmax,dataset,train_dataset,splitid,hyperparams,catlist=None):
    '''
    An issue with purity as an evaluation metric is unbalanced nature of real datasets.
    This function balances a dataframe by category.

    if catlist is specified, those exact categories will be taken.
    if splitid is specified, the unseen categories in that split will be taken.
    '''
    # making an even sampling of detected candidates
    allcans = None
    logging.info("Starting a lengthy db read with progress bar")
    conn = psycopg2.connect(**hyperparams.pg)
    if catlist is not None:
        for df in tqdm(readsql(f"SELECT * FROM candidate_bbox NATURAL JOIN ground_truth NATURAL JOIN splitcats WHERE dataset = '{dataset}' AND classname IN ({catlist})",hyperparams,chunksize=10000,lowlevel=False,conn=conn)):
            try: allcans.append(df)
            except: allcans = df
    else:
        for df in tqdm(readsql(f"SELECT * FROM candidate_bbox NATURAL JOIN ground_truth NATURAL JOIN splitcats WHERE dataset = '{dataset}' AND splitid = {splitid} AND seen = 0",hyperparams,chunksize=10000,lowlevel=False,conn=conn)):
            try: allcans.append(df)
            except: allcans = df
    conn.close()
    allcans = transfer_exclude(allcans,train_dataset,splitid,dataset,hyperparams)
    cat_freqs = allcans.groupby('classname').size()
    takemax = min(2 * min(cat_freqs),canmax // len(cat_freqs))
    logging.info(f"For detected candidates, taking {takemax} candidates from each category, making for {takemax * len(cat_freqs)} total candidates")
    for cat,df in allcans.groupby('classname'):
        take = df.sample(int(min(takemax,len(df))))
        # hack for not being able to concat on the first go. not actually anticipating failure.
        try:
            detectedeven = pd.concat([detectedeven,take])
        except:
            detectedeven = take
    # making an even sampling of perfect candidates
    if catlist is not None:
        perfect_cans = readsql(f"SELECT * FROM perfect_bbox WHERE category IN ({catlist}) AND dataset = '{dataset}'",hyperparams)
    else:
        perfect_cans = readsql(f"SELECT * FROM perfect_bbox AND dataset = '{dataset}'",hyperparams)
    if len(perfect_cans) == 0:
        return detectedeven,[] #this happens with pascal for now.
    cat_freqs = perfect_cans.groupby('category').size()
    takemax = min(2 * min(cat_freqs),canmax // len(cat_freqs))
    logging.info(f"For perfect candidates, taking {takemax} candidates from each category, making for {takemax * len(cat_freqs)} total candidates")
    for cat,df in perfect_cans.groupby('category'):
        take = df.sample(int(min(takemax,len(df))))
        # hack for not being able to concat on the first go. not actually anticipating failure.
        try:
            perfecteven = pd.concat([perfecteven,take])
        except:
            perfecteven = take
    return detectedeven,perfecteven

def baselines(canmax,initialization,cantype,args:argparse.Namespace,seen):
    '''
    This function's purpose is inserting tuples into raw_ctx_reprs.
    '''
    create_tables()
    batchsize = 10
    tsv = open('{sess_dir}/baselines','a')
    dataset,splitid = args.transfer_dataset,args.splitid
    outputs = None
    if dataset == args.train_dataset:
        catdf = np.squeeze(readsql(f"SELECT category FROM splitcats WHERE splitid = {args.splitid} AND dataset = '{dataset}' AND seen = 0",hyperparams).values)
    else:
        catdf = np.squeeze(readsql(f"SELECT distinct(category) FROM splitcats WHERE dataset = '{dataset}'",hyperparams).values)
    catlist = ','.join(["'{}'".format(category) for category in catdf])
    if cantype == "perfect_natfreq":
        can_names = random.sample(os.listdir(hyperparams.root('val_patches')),canmax)
        can_ids = [-1 for name in can_names] #-1 indicating no value
        cats = [name.split("_")[1] for name in can_names]
        can_names = [hyperparams.root(f"val_patches/{can_name}") for can_name in can_names]
        sampled_candidates = np.squeeze(np.dstack([can_names,can_ids,cats]))
    elif cantype == "detected_natfreq":
        if dataset != args.train_dataset:
            sampled_candidates = readsql(f"SELECT imgname,canid,classname FROM candidate_bbox NATURAL JOIN ground_truth WHERE dataset = '{dataset}'",hyperparams)
        else:
            sampled_candidates = readsql(f"SELECT imgname,canid,classname FROM candidate_bbox NATURAL JOIN ground_truth INNER JOIN splitcats ON classname = splitcats.category AND ground_truth.dataset = splitcats.dataset WHERE splitcats.dataset = '{dataset}' AND splitcats.seen = {seen}",hyperparams)
        if args.train_dataset != dataset:
            l0 = len(sampled_candidates)
            sampled_candidates = transfer_exclude(sampled_candidates,args.train_dataset,splitid,dataset)
            logging.info(f"Due to removing knowns after transfering, removed {len(sampled_candidates) - l0} samples")
        sampled_candidates['imgname'] = sampled_candidates['imgname'].apply(lambda x: hyperparams.root(f"val_candidateimgs/{x}"))
        sampled_candidates.reindex(np.random.permutation(sampled_candidates.index)) #shuffling candidates
        sampled_candidates = sampled_candidates.sample(min(len(sampled_candidates),canmax)).values
    elif cantype == "detected_even":
        sampled_candidates,_ = even_cans(canmax,dataset,args.train_dataset,splitid,hyperparams,catlist=catlist)
    elif cantype == "perfect_even":
        _,sampled_candidates = even_cans(canmax,dataset,args.train_dataset,splitid,hyperparams,catlist=catlist)
    count = 0
    with tf.Session( ) as sess:
        X_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize, 224,224,3],name="X")
        dropout = tf.placeholder(PRECISION)
        net_w,net_b = initialize(None,args,hyperparams,1000,initialization=initialization,only_fromnpy=True)
        if initialization == "embed":
            network = vanilla(X_placeholder,net_w,net_b,dropout,stop_fc7=True,output_keys=['fc7'])
        else:
            network = vanilla(X_placeholder,net_w,net_b,dropout,output_keys=['fc7'])
        sess.run(tf.initialize_all_variables())
        for candidates in chunks(sampled_candidates,batchsize):
            logging.info(initialization,cantype,count/len(sampled_candidates),seen)
            if cantype == "detected_even":
                imgnames,canids,categories = candidates['imgname'].values,candidates['canid'].values,candidates['classname'].values
            elif cantype == "perfect_even":
                imgnames,canids,categories = candidates['imgname'].values,batchsize * [-1],candidates['category'].values
            else:
                imgnames,canids,categories = np.array(candidates).T
            N = len(canids)
            try:
                if cantype in ['perfect_natfreq','perfect_even']:
                    X = [imread_wrap(name,hyperparams) for name in imgnames]
                else:
                    if dataset == 'COCO':
                        X = [imread_wrap(os.path.join(hyperparams.root("val_candidateimgs"),"_".join(map(str,[imgnames[i],0,hyperparams.candidate_method,canids[i]]))) + ".jpg",hyperparams) for i in range(N)]
                    elif dataset == 'pascal':
                        X = [imread_wrap(os.path.join(hyperparams.root("val_candidateimgs"),"_".join(map(str,[imgnames[i],hyperparams.candidate_method,canids[i]]))) + ".jpg",hyperparams) for i in range(N)]
                feed = {X_placeholder : np.array(X).reshape((N,224,224,3)), dropout : 1.0}
            except:
                logging.warn("failed to imread")
                continue
            try:
                output = sess.run(network,feed_dict=feed)
                if type(output) == dict:
                    k = 'fc7' if 'fc7' in output.keys() else 'similarity'
                    output = output[k]
                if outputs is None: outputs = np.zeros([canmax] + list(output.shape))
                for i,imgname in enumerate(imgnames):
                    vec = floatserial(output[i],12)
                    outputs[count] = output[i]
                    tup = (initialization,imgname,canids[i],vec,categories[i],cantype,splitid,dataset)
                    tsv.write('\t'.join(map(str,tup)) + '\n')
                    count += 1
                    #dosql(xplatformtime(lambda x:"INSERT INTO raw_ctx_reprs VALUES ('{}','{}',{},'{}','{}','{}')".format))
            except:
                logging.warn("Failed on an iteration")
        logging.info("All outputs are gathered and ready to do precision/recall")

def metric_visualize( ):
    '''
    When using learned distance metric between DRAW concatenated things.
    '''
    pass

def context_similarity(args:argparse.Namespace,hyperparams,savemethod="hdf",restart_num=None,keep_p=True,img_s=224):
    '''
    Takes a model and runs visualizations on a variety of data of similar context.
    '''
    batchsize = args.batchsize
    sample_fn = sample_img(N=args.batchsize,variety="train_normal",splitid=args.splitid,full_img=(hyperparams.ctxop in fullimg_group))
    with tf.Session() as sess:
        md = dataset_multiplier(hyperparams)
        X_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        Xp_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="Xp")
        Xfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        Xpfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        Xcfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        Xcpfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize, img_s,img_s,3],name="X")
        y_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize],name="eq")
        ya_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize,nclasses],name="ya")
        yb_placeholder = tf.placeholder(dtype=PRECISION, shape=[md*batchsize,nclasses],name="yb")
        bbox_loc = tf.placeholder(dtype=PRECISION,shape=[md*batchsize,4],name="bbox_loc")
        bboxp_loc = tf.placeholder(dtype=PRECISION,shape=[md*batchsize,4],name="bboxp_loc")
        dropout = tf.placeholder(PRECISION,name="dropout")
        parameters = initialize(hyperparams.M,args,hyperparams,nclasses,initialization=hyperparams.initialization)
        sess.run(tf.initialize_all_variables())
        if savemethod == "tf":
            t0 = int(ckpt.model_checkpoint_path.split("-")[-1])
            saver.restore(sess,ckpt.model_checkpoint_path)
            logging.info(f"RESTARTING FROM step={t0}")
            # In case there are uninitialized variables...
        elif savemethod == "hdf":
            if restart_num == None:
                t0 = max([int(x.split(".")[0]) for x in os.listdir(modeldir)])
            else:
                t0 = restart_num
            restartname = f'{modeldir}/{str(t0)}.hdf'
            try:
                npy_w,npy_b = dd.io.load(restartname)
            except:
                logging.warn(f"Failed to load from hdf {restartname}")
                sys.exit(1)
            global_step = tf.Variable(t0)
            sess.run(tf.initialize_all_variables())
            for k in parameters[0].keys():
                sess.run(parameters[0][k].assign(npy_w[k]))
            for k in parameters[1].keys():
                sess.run(parameters[1][k].assign(npy_b[k]))
            logging.info(f"Successfully restarted from parameters at {t0}")
        loss,euclid,optimizer,attention,pre,pre_full,post,out1,out2,filt,gvs = netops(parameters,X_placeholder,Xp_placeholder,Xfull_placeholder,Xpfull_placeholder,Xcfull_placeholder,Xcpfull_placeholder,bbox_loc,bboxp_loc,y_placeholder,ya_placeholder,yb_placeholder,dropout,batchsize,hyperparams,keep_p=keep_p,devices=devices)
        if keep_p:
            attention,attention_p = attention
            if hyperparams.ctxop == 'DRAW':
                attentionvals,attention_boxes = attention
                attentionvals_p,attention_boxes_p = attention_p
            pre,pre_p = pre
            pre_full,pre_full_p = pre_full
            post,post_p = post
            filt,filt_p = filt
        else:
            if hyperparams.ctxop == 'DRAW':
                attentionvals,attention_boxes = attention
        for t in range(args.niter):
            uuids = []
            for i in range(batchsize):
                uuids.append(str(uuid.uuid1()))
                time.sleep(0.01) #needed for uuids to be distinct.
            assert(len(uuids) == np.unique(uuids).size) #make sure no overlap, so labels are correct.
            X,Xfull,bboxs,y,imgnames,ctxs,_ = sample_fn(report_ctx=True)
            Xp,Xpfull,bboxsp,yp,imgnamesp,ctxsp,ok = sample_fn(tomatch=y,ctx=True,ctx_k=args.ctx_k,ctxlabels=ctxs,report_ctx=True)
            Xcfull,alt_bboxs = center_transform(Xfull,imgnames,bboxs)
            Xcpfull,alt_bboxsp = center_transform(Xpfull,imgnamesp,bboxsp)
            yint,ypint = [split['known'].index(yi) for yi in y], [split['known'].index(yi) for yi in yp] #convert from 
            equal = (y == yp).astype(np.float32)
            logging.info("***************t={t},equal={str(equal)},enough_ctx={str(ok)}***************")
            if hyperparams.ctxop in fullimg_group:
                feed = {X_placeholder : X, Xp_placeholder: Xp, y_placeholder : equal,dropout : 1.0, ya_placeholder : onehot(yint,nclasses), yb_placeholder : onehot(ypint,nclasses),Xfull_placeholder : Xfull, Xpfull_placeholder : Xpfull, Xcfull_placeholder : Xcfull, Xcpfull_placeholder : Xcpfull, bbox_loc : bboxs, bboxp_loc : bboxsp}
            else:
                feed = {X_placeholder : X, Xp_placeholder: Xp, y_placeholder : equal,dropout : 1.0, ya_placeholder : onehot(yint,nclasses), yb_placeholder : onehot(ypint,nclasses)}
            if hyperparams.ctxop in ["patches","above_below"]:
                del(feed[bbox_loc])
                del(feed[bboxp_loc])
            # need to get the uuid information in here somehow.
            logging.info("*************DOING NON P*************")
            plot_qualitative(sess,filt,post,bboxs,bboxs,feed,t,splitid,args.nickname,Xfull,parameters,imgnames,attentionvals,hyperparams,args.trial,t_sess_dir,hyperparams.include_center,isconst=False,mask=ok,doing=True,uuids=uuids,prime=False)
            logging.info("*************DOING P*************")
            plot_qualitative(sess,filt_p,post_p,bboxsp,bboxsp,feed,t,splitid,args.nickname,Xpfull,parameters,imgnamesp,attentionvals_p,hyperparams,args.trial,t_sess_dir,hyperparams.include_center,isconst=False,mask=ok,doing=True,uuids=uuids,prime=True)

def pascal_store( ):
    '''
    Not enough data exists to bother with the class balancing stuff, and no time for me to do detections,
    so just run on ground truth segmentations.
    '''
    pass


def store(hyperparams,nickname,seen,canmax,splitid,args:argparse.Namespace,t,modeldir,trial,savemethod="tf",timestep=None,whicht=1,sql=False,dataset='COCO',train_dataset='COCO',decode_arch='vggnet',img_s=224):
    '''
    The purpose of this function is saving representations into arch_ctx_reprs for use by affinity.py

    t - timestep from training to use.
    '''
    batchsize = int(args.batchsize)
    # depends on what is trained on.
    nclasses = len(readsql(f"SELECT * FROM splitcats WHERE splitid = {splitid} AND seen = 1 AND dataset = '{train_dataset}'",hyperparams))
    create_tables()
    imgsizes = readsql("SELECT * FROM imgsize",hyperparams)
    if dataset == train_dataset:
        catdf = np.squeeze(readsql(f"SELECT category FROM splitcats WHERE splitid = {splitid} AND dataset = '{dataset}' AND seen = {seen}",hyperparams).values)
    else: #keeping all categories if transfering.
        catdf = np.squeeze(readsql(f"SELECT category FROM splitcats WHERE dataset = '{dataset}'",hyperparams).values)
    catlist = ','.join(["'{}'".format(category) for category in catdf])
    needs_centering = True #making things simpler for now, do this work whether or not needed.
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if hyperparams.ctxop == "patches":
            nrows = 3
            attentionvals = None
        elif hyperparams.ctxop == "above_below":
            nrows = 2
            attentionvals = None
        else: nrows = 1
        X_placeholder = tf.placeholder(dtype=PRECISION, shape=[nrows * batchsize, img_s,img_s,3],name="X")
        y = tf.placeholder(dtype=PRECISION, shape=[batchsize],name="eq")
        Xfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[nrows * batchsize, img_s,img_s,3],name="Xf")
        Xcfull_placeholder =  tf.placeholder(dtype=PRECISION, shape=[nrows * batchsize, img_s,img_s,3],name="Xc")
        bbox_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize,4],name="bbox")
        dropout = tf.placeholder(PRECISION)
        if modeldir == '/data/aseewald/COCO/models/rawvgg':
            weights,biases = initialize(hyperparams.M,"vggnet",only_fromnpy=True)
        else:
            # these will get over-ridden anyway.
            weights,biases = initialize(hyperparams.M,args,hyperparams,nclasses,initialization="vggnet")
        assert(savemethod in ["tf","hdf"])
        if savemethod == "tf":
            if timestep == None:
                ckpt = tf.train.get_checkpoint_state(modeldir)
                try:
                    ck = ckpt.all_model_checkpoint_paths[-1 * (whicht)]
                    t = int(ck.split("-")[-1])
                    tf.train.Saver().restore(sess, ck)
                except:
                    logging.warn(f"Failed to restore for whicht={whicht}")
                    return False
        elif savemethod == "hdf":
            logging.info(f"DURING STORE LOADING AT WITH t={t}")
            weights,biases = dd.io.load(f'{modeldir}/{str(t)}.hdf')
            weights = {k : tf.Variable(v,dtype=PRECISION) for k,v in weights.items()}
            biases = {k : tf.Variable(v,dtype=PRECISION) for k,v in biases.items()}
            sess.run(tf.initialize_all_variables())
        allcans = readsql("SELECT * FROM candidate_bbox NATURAL JOIN ground_truth WHERE classname IN ({catlist}) AND dataset = '{dataset}'",hyperparams)
        if train_dataset != dataset:
            allcans = transfer_exclude(allcans,train_dataset,splitid,dataset)
        allcans.reindex(np.random.permutation(allcans.index)) #shuffling allcans
        canmax = min(canmax,len(allcans))
        detected_naturalfreq_candidates = allcans.sample(canmax)
        perfect_naturalfreq_candidates = readsql("SELECT * FROM perfect_bbox WHERE category IN ({catlist}) AND dataset = '{dataset}'",hyperparams)
        if len(perfect_naturalfreq_candidates) == 0:
            logging.info("Not doing perfect, probably because using pascal")
        else:
            canmax = min(canmax,len(perfect_naturalfreq_candidates))
            logging.info(f"After looking up candidates for discovery, canmax={canmax}")
            perfect_naturalfreq_candidates = perfect_naturalfreq_candidates.sample(canmax)
        # these are allowed to have fewer than canmax.
        detected_even_candidates, perfect_even_candidates = even_cans(canmax,dataset,train_dataset,splitid,hyperparams,catlist=catlist)
        maybe_mkdir(t_sess_dir(t))
        tsv = open(os.path.join(CAN_REPR_DIR,'{t_sess_dir(t)}/seen-{seen}.tsv'),'w')
        # Making debug_items
        # accidentally called miny min in this table, so this is not a typo.
        if hyperparams.isvanilla:
            network = vanilla(X_placeholder,weights,biases,dropout)
        elif hyperparams.ctxop in ['block_blur','block_intensity']:
            _,network,_,_,_ = arch_block(X_placeholder,Xfull_placeholder,(weights,biases),dropout,batchsize,hyperparams.ctxop,hyperparams.numfilts,hyperparams.M)
        elif hyperparams.ctxop in ['above_below','patches']:
            network,_,_,_,_ = arch_patch(X_placeholder,Xfull_placeholder,Xcfull_placeholder,(weights,biases),dropout,batchsize,hyperparams.loss_t,hyperparams.ctxop)
        else:
            arch_draw = draw_switch(hyperparams)
            network,_,_,_,_,_ = arch_draw(X_placeholder,Xfull_placeholder,Xcfull_placeholder,bbox_placeholder,(weights,biases),dropout,batchsize,hyperparams.relative_arch,args.distinct_decode,hyperparams.loss_t,hyperparams.baseline_t,hyperparams.numfilts,hyperparams.include_center,decode_arch,hyperparams.stop_grad,stop_pos=hyperparams.decode_stop_pos,keep_resolution=hyperparams.keep_resolution)
        # NOT EVEN NOT PERFECT
        count = 0
        for candidates in list(chunks(detected_naturalfreq_candidates,batchsize))[:-1]:
            count += 1
            logging.info(f"detected,naturalfreq: {batchsize * count / len(detected_naturalfreq_candidates)}")
            try:
                if dataset == 'COCO':
                    Xs = [imread_wrap(hyperparams.root("val_candidateimgs/") + candidate['imgname'] + "_0_objectness_" + str(candidate['canid']) + ".jpg",hyperparams) for _,candidate in candidates.iterrows()]
                elif dataset == 'pascal':
                    Xs = [imread_wrap(hyperparams.root("val_candidateimgs/") + candidate['imgname'] + "_objectness_" + str(candidate['canid']) + ".jpg",hyperparams) for _,candidate in candidates.iterrows()]
                N = len(Xs)
                feed = {X_placeholder : np.array(Xs).reshape((N,img_s,img_s,3)), dropout : 1.0}
                # need to adjust these to [0,img_s] range rather than raw image coords.
                shapes = []
                for _,candidate in candidates.iterrows():
                    shapes.append((candidate,imgsizes[imgsizes['imgname'] == candidate['imgname']].iloc[0]))
                # accidentally called miny min in this table, so this is not a typo.
                bboxs = [(img_s*candidate['min']/shape['height'],img_s*candidate['maxy']/shape['height'],img_s*candidate['minx']/shape['width'],img_s*candidate['maxx']/shape['width']) for candidate,shape in shapes]
                Xfull,j,Xcfull = np.zeros((batchsize,img_s,img_s,3)),0,np.zeros((batchsize,img_s,img_s,3))
                for _,candidate in candidates.iterrows():
                    Xfull[j] = imread_wrap(hyperparams.root("val_images/") + candidate['imgname'] + ".jpg",hyperparams)
                    if needs_centering:
                        Xcfull[j] = center_transform(Xfull[j],bboxs=bboxs[j])[0]
                feed[Xfull_placeholder],feed[Xcfull_placeholder] = Xfull,Xcfull
                if hyperparams.ctxop in ['above_below','patches']: feed = refine_single(hyperparams,feed,Xfull_placeholder,Xfull,X_placeholder,np.array(Xs),Xcfull_placeholder,Xcfull,bboxs)
                feed[bbox_placeholder] = bboxs
                output = np.squeeze(sess.run(network,feed_dict = feed))
                output = output.tolist()['similarity']
                if hyperparams.ctxop != 'DRAW': output = np.transpose(output.reshape((hyperparams.numfilts,batchsize,output.size / (batchsize * hyperparams.numfilts))),axes=[1,0,2])
                for i,out in enumerate(output):
                    if sql:
                        dosql(xplatformtime(lambda x:"INSERT INTO arch_ctx_reprs VALUES ({t},{x},'{candidates[i]}',{1},{0},{canids[i]},'{out}','{splitid}','{categories[i]}','{nickname}')",hyperparams,only_log=True))
                    else:
                        tup = (t,'now()',candidates.iloc[i]['imgname'],0,0,candidates.iloc[0]['canid'],floatserial(out,12),splitid,candidates.iloc[i]['classname'],nickname,seen,trial,dataset)
                        tsv.write("\t".join(list(map(str,tup))) + "\n")
            except:
                logging.warn(f"Failed on batch {count} of not even and detected")
                continue
        # NOT EVEN AND PERFECT
        count = 0
        for candidates in list(chunks(perfect_naturalfreq_candidates,batchsize))[:-1]:
            count += 1
            logging.info(f"perfect,naturalfreq: {batchsize * count / len(perfect_naturalfreq_candidates)}")
            try:
                Xs = [imread_wrap(candidate['patchname'],hyperparams) for _,candidate in candidates.iterrows()]
                N = len(Xs)
                feed = {X_placeholder : np.array(Xs).reshape((N,img_s,img_s,3)), dropout : 1.0}
                # need to adjust these to [0,img_s] range rather than raw image coords.
                shapes = []
                for _,candidate in candidates.iterrows():
                    shapes.append((candidate,imgsizes[imgsizes['imgname'] == os.path.splitext(os.path.split(candidate['imgname'])[1])[0]].iloc[0] ))
                # accidentally called miny min in this table, so this is not a typo.
                bboxs = [(img_s*candidate['miny']/shape['height'],img_s*candidate['maxy']/shape['height'],img_s*candidate['minx']/shape['width'],img_s*candidate['maxx']/shape['width']) for candidate,shape in shapes]
                Xfull,j,Xcfull = np.zeros((batchsize,img_s,img_s,3)),0,np.zeros((batchsize,img_s,img_s,3))
                for _,candidate in candidates.iterrows():
                    Xfull[j] = imread_wrap(candidate['imgname'],hyperparams)
                    if needs_centering: Xcfull[j] = center_transform(Xfull[j],bboxs=bboxs[j])[0] 
                feed[Xfull_placeholder],feed[Xcfull_placeholder] = Xfull,Xcfull
                if hyperparams.ctxop in ['above_below','patches']: feed = refine_single(hyperparams,feed,Xfull_placeholder,Xfull,X_placeholder,np.array(Xs),Xcfull_placeholder,Xcfull,bboxs)
                feed[bbox_placeholder] = bboxs
                output = np.squeeze(sess.run(network,feed_dict = feed))
                output = output.tolist()['similarity']
                if hyperparams.ctxop != 'DRAW': output = np.transpose(output.reshape((hyperparams.numfilts,batchsize,output.size / (batchsize * hyperparams.numfilts))),axes=[1,0,2])
                for i,out in enumerate(output):
                    if sql:
                        dosql(xplatformtime(lambda x:f"INSERT INTO arch_ctx_reprs VALUES ({t},{x},'{candidates[i]}',{1},{0},{canids[i]},'{out}','{splitid}','{categories[i]}','{nickname}')",hyperparams,only_log=True))
                    else:
                        tup = (t,'now()',candidates.iloc[i]['imgname'],1,0,candidates.iloc[0]['patchname'],floatserial(out,12),splitid,candidates.iloc[i]['category'],nickname,seen,trial,dataset)
                        tsv.write("\t".join(list(map(str,tup))) + "\n")
            except:
                logging.warn(f"Failed on batch {count} of not even and perfect")
                continue
        # EVEN AND NOT PERFECT
        count = 0
        for candidates in list(chunks(detected_even_candidates,batchsize))[:-1]:
            count += 1
            logging.info(f"detected,even: {batchsize * count / len(detected_even_candidates)}")
            try:
                if dataset == 'COCO':
                    Xs = [imread_wrap(hyperparams.root("val_candidateimgs/") + candidate['imgname'] + "_0_objectness_" + str(candidate['canid']) + ".jpg",hyperparams) for _,candidate in candidates.iterrows()]
                elif dataset == 'pascal':
                    Xs = [imread_wrap(hyperparams.root("val_candidateimgs/") + candidate['imgname'] + "_objectness_" + str(candidate['canid']) + ".jpg",hyperparams) for _,candidate in candidates.iterrows()]
                N = len(Xs)
                feed = {X_placeholder : np.array(Xs).reshape((N,img_s,img_s,3)), dropout : 1.0}
                # need to adjust these to [0,img_s] range rather than raw image coords.
                shapes = []
                for _,candidate in candidates.iterrows():
                    shapes.append((candidate,imgsizes[imgsizes['imgname'] == candidate['imgname']].iloc[0]))
                # accidentally called miny min in this table, so this is not a typo.
                bboxs = [(img_s*candidate['min']/shape['height'],img_s*candidate['maxy']/shape['height'],img_s*candidate['minx']/shape['width'],img_s*candidate['maxx']/shape['width']) for candidate,shape in shapes]
                Xfull,j,Xcfull = np.zeros((batchsize,img_s,img_s,3)),0,np.zeros((batchsize,img_s,img_s,3))
                for _,candidate in candidates.iterrows():
                    Xfull[j] = imread_wrap(hyperparams.root("val_images/") + candidate['imgname'] + ".jpg",hyperparams)
                    if needs_centering: Xcfull[j] = center_transform(Xfull[j],bboxs=bboxs[j])[0]
                feed[Xfull_placeholder],feed[Xcfull_placeholder] = Xfull,Xcfull
                if hyperparams.ctxop in ['above_below','patches']: feed = refine_single(hyperparams,feed,Xfull_placeholder,Xfull,X_placeholder,np.array(Xs),Xcfull_placeholder,Xcfull,bboxs)
                feed[bbox_placeholder] = bboxs
                output = np.squeeze(sess.run(network,feed_dict = feed))
                output = output.tolist()['similarity']
                if hyperparams.ctxop != 'DRAW': output = np.transpose(output.reshape((hyperparams.numfilts,batchsize,output.size / (batchsize * hyperparams.numfilts))),axes=[1,0,2])
                for i,out in enumerate(output):
                    if sql:
                        dosql(xplatformtime(lambda x:f"INSERT INTO arch_ctx_reprs VALUES ({t},{x},'{candidates[i]}',{1},{0},{canids[i]},'{out}','{splitid}','{categories[i]}','{nickname}')",hyperparams,only_log=True))
                    else:
                        tup = (t,'now()',candidates.iloc[i]['imgname'],0,1,candidates.iloc[0]['canid'],floatserial(out,12),splitid,candidates.iloc[i]['classname'],nickname,seen,trial,dataset)
                        tsv.write("\t".join(list(map(str,tup))) + "\n")
            except:
                logging.warn(f"Failed on batch {count} of detected and even")
                continue
        # EVEN AND PERFECT
        count = 0
        for candidates in list(chunks(perfect_even_candidates,batchsize))[:-1]:
            count += 1
            logging.info(f"perfect,even: {batchsize * count / len(perfect_naturalfreq_candidates)}")
            try:
                Xs = [imread_wrap(candidate['patchname'],hyperparams) for _,candidate in candidates.iterrows()]
                N = len(Xs)
                feed = {X_placeholder : np.array(Xs).reshape((N,img_s,img_s,3)), dropout : 1.0}
                Xfull = [imread_wrap(candidate['imgname'],hyperparams) for _,candidate in candidates.iterrows()]
                # need to adjust these to [0,img_s] range rather than raw image coords.
                shapes = []
                for _,candidate in candidates.iterrows():
                    shapes.append((candidate,imgsizes[imgsizes['imgname'] == os.path.splitext(os.path.split(candidate['imgname'])[1])[0]].iloc[0] ))
                # accidentally called miny min in this table, so this is not a typo.
                bboxs = [(img_s*candidate['miny']/shape['height'],img_s*candidate['maxy']/shape['height'],img_s*candidate['minx']/shape['width'],img_s*candidate['maxx']/shape['width']) for candidate,shape in shapes]
                Xfull,j,Xcfull = np.zeros((batchsize,img_s,img_s,3)),0,np.zeros((batchsize,img_s,img_s,3))
                for _,candidate in candidates.iterrows():
                    Xfull[j] = imread_wrap(candidate['imgname'],hyperparams)
                    if needs_centering: Xcfull[j] = center_transform(Xfull[j],bboxs=bboxs[j])[0]
                feed[Xfull_placeholder],feed[Xcfull_placeholder] = Xfull,Xcfull
                if hyperparams.ctxop in ['above_below','patches']: feed = refine_single(hyperparams,feed,Xfull_placeholder,Xfull,X_placeholder,np.array(Xs),Xcfull_placeholder,Xcfull,bboxs)
                feed[bbox_placeholder] = bboxs
                output = np.squeeze(sess.run(network,feed_dict = feed))
                output = output.tolist()['similarity']
                if hyperparams.ctxop != 'DRAW': output = np.transpose(output.reshape((hyperparams.numfilts,batchsize,output.size / (batchsize * hyperparams.numfilts))),axes=[1,0,2])
                for i,out in enumerate(output):
                    if sql:
                        dosql(xplatformtime(lambda x:f"INSERT INTO arch_ctx_reprs VALUES ({t},{x},'{candidates[i]}',{1},{0},{canids[i]},'{out}','{splitid}','{categories[i]}','{nickname}')",hyperparams,only_log=True))
                    else:
                        tup = (t,'now()',candidates.iloc[i]['imgname'],1,1,candidates.iloc[0]['patchname'],floatserial(out,12),splitid,candidates.iloc[i]['category'],nickname,seen,trial,dataset)
                        tsv.write("\t".join(list(map(str,tup))) + "\n")
            except:
                logging.warn(f"Failed on batch {count} of perfect and even")
                continue

def loss_plots( ):
    X = readsql("SELECT * FROM loss WHERE nickname = 'DRAW' OR nickname = 'DRAW_bias' OR nickname = 'DRAW_fixed'")
    P = readsql("SELECT * FROM pairloss")
    Pplus = readsql("SELECT * FROM pairloss WHERE cat_a = cat_b")
    Pneg =  readsql("SELECT * FROM pairloss WHERE cat_a <> cat_b")
    sns.pointplot(x='timestep',y='loss',hue='nickname',data=X)
    plt.scatter( )
    pass

def attention_plots(nickname):
    # Plot attention parameters over time.
    logging.info("Plotting attention parameter data")
    X = readsql("SELECT * FROM attentionvals")
    X.columns = [ ] # rename for graph display purposes.
    g_raw = sns.FacetGrid(data=X[X['nickname'] == nickname],col="filtid",row="imgname")
    g_raw.map(plt.scatter,"timestep","raw")
    plt.show( ) # call show rather than savefig because I want to manually resize and click the matplotlib save button.
    g_rawrel = sns.FacetGrid(data=X[X['nickname'] == nickname],col="filtid",row="imgname")
    g_raw.map(plt.scatter,"timestep","rawrel")
    plt.show()
    g_read = sns.FacetGrid(data=X[X['nickname'] == nickname],col="filtid",row="imgname")
    g_raw.map(plt.scatter,"timestep","readable")
    plt.show()
    g_readrel = sns.FacetGrid(data=X[X['nickname'] == nickname],col="filtid",row="imgname")
    g_raw.map(plt.scatter,"timestep","readablerel")
    plt.show()

    logging.info("Making animations")
    subprocess.call(["bash","animate.sh"])

splitid = args.splitid
if args.split_type == "random":
    split = hyperparams.possible_splits[splitid]
elif args.split_type == "illustrative":
    split = hyperparams.illustrative_splits[splitid]
elif args.split_type == "clusterbased":
    split = coco_clust[splitid]
else:
    logging.warn("Unknown split_type, see command line help.")
    sys.exit(1)
nclasses = len(split['known'])

def run_baselines( ):
    count = 0
    #for net_t in ['vggnet','random','embed']:
    for seen in [0,1]:
        for net_t in ['random','vggnet','embed']:
            for can_t in ['detected_natfreq','perfect_natfreq','detected_even','perfect_even']:
                if args.transfer_dataset == 'pascal' and "perfect" in can_t:
                    continue
                baselines(int(args.canmax),net_t,can_t,args,seen)
                count += 1
                tf.reset_default_graph()

def pr(df:pd.DataFrame) -> Tuple[List[float],List[float]]:
    precision,recall = [],[]
    for thresh in df['distance'].unique():
        positives = df[df['distance'] <= thresh]
        negatives = df[df['distance'] > thresh]
        true_positives = positives[positives['equal'] == 1]
        false_positives = positives[positives['equal'] == 0]
        false_negatives = negatives[negatives['equal'] == 1]
        true_negatives = negatives[negatives['equal'] == 0]
        if (len(true_positives) + len(false_positives)) > 0 and (len(true_positives) + len(false_negatives)) > 0:
            precision.append(len(true_positives)/(len(true_positives) + len(false_positives)))
            recall.append(len(true_positives)/(len(true_positives) + len(false_negatives)))
    return precision,recall

def pr_all(constraints):
    d = hyperparams.root('cache/prsamples/')
    samps = os.listdir(d)
    dat = []
    for samp in samps:
        try:
            dataset,nickname,splitid,trial,tstep,class_partition,min_numobj,max_numobj = samp.split('_')
        except:
            dataset,nickname,splitid,trial,tstep,class_partition = samp.split('_')
            min_numobj = 0
            max_numobj = sys.maxsize
        splitid,trial,tstep = int(splitid),int(trial),int(tstep)
        min_numobj,max_numobj = int(min_numobj),int(max_numobj) 
        if 'nocenter' in nickname: continue
        if 'dataset' in constraints.keys():
            if constraints['dataset'] != dataset or constraints['splitid'] != splitid or (tstep not in constraints['timestep']) or nickname not in constraints['nicknames']:
                continue
        logging.info("Including",samp)
        precision,recall = pr(pickle.load(open(os.path.join(d,samp),'rb')))
        for i,prec in enumerate(precision):
            dat.append({'nickname' : nickname, 'splitid' : splitid, 'trial' : trial, 'tstep' : tstep, 'class_partition' : class_partition,'minimum objects' : min_numobj, 'maximum objects' : max_numobj, 'precision' : prec, 'recall' : recall[i], 'auc' : auc(precision,recall)})
    X = pd.DataFrame(dat)
    for splitid,dfp in X.groupby('splitid'):
        for maxobj,dfpp in dfp.groupby('maximum objects'):
            plt.gcf().set_size_inches(30,30)
            nicks = dfpp['nickname'].unique()
            for nick in nicks:
                assert(nick in constants.hue_order), f"nick={nick} not in pallete"
            pal = [constants.palette[constants.hue_order.index(nick)] for nick in nicks]
            g = sns.FacetGrid(dfpp,row="tstep",col="class_partition",hue="nickname",aspect=1,hue_order=nicks,palette=pal)
            g.map(plt.plot,"recall","precision")
    #                plt.legend(loc='lower right')
            plt.tight_layout()
            plt.show()
            plt.close("all")
            fig,ax = plt.subplots()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            col = ['class_partition','auc']
            pd.set_option('precision', 4)
            for nickname in dfpp['nickname'].unique():
                dfpp['nickname'].replace(nickname,constants.papernames[constants.hue_order.index(nickname)],inplace=True)
            auc_table = dfpp[col + ['nickname']].drop_duplicates()
            auc_table = auc_table.sort(['class_partition','auc']) #hierarchical sorting
            rowcolors,rowlabels = [],[]
            for i in range(len(auc_table)):
                nick = auc_table.iloc[i]['nickname']
                rowlabels.append(nick) 
                rowcolors.append(constants.palette[constants.papernames.index(nick)])
            auc_table['auc'] = auc_table['auc'].apply(lambda x: '{:.3f}'.format(x))
            unseen_table = auc_table[auc_table['class_partition'] == 'unseen']
            seen_table = auc_table[auc_table['class_partition'] == 'seen']
            all_table = auc_table[auc_table['class_partition'] == 'all']
            border = len(rowcolors)//2
            ax.table(cellText=unseen_table[col].values,colLabels=col,rowColours=rowcolors[0:border],rowLabels=rowlabels[0:border],colWidths=[0.3,0.3],loc='center',rowLoc='right',alpha=0.5)
            plt.show()
            fig,ax = plt.subplots()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.table(cellText=seen_table[col].values,colLabels=col,rowColours=rowcolors[border:],rowLabels=rowlabels[border:],colWidths=[0.3,0.3],loc='center',rowLoc='right',alpha=0.5)
            plt.show()

def pr_run(num_samples,batchsize,splitid,tstep,baselines=False,restore_method="assign",visprob=0.0,quantifyprob=1.0,class_partition="all",track=False,dataset='COCO',train_dataset='COCO',decode_arch='vggnet',img_s=224,min_numobj=0,max_numobj=sys.maxsize):
    '''
    Store the features of 
    Is there a difference between running arch_common.initialize and then making assignments vs declaring variables?

    The raw saliency and raw closest files have row position encoding time meaning.
    May run multiple times on same timestep, so I'm working with the processed version here.
    '''
    trial = int(args.trial)
    distance_data = []
    pname = f'{t_sess_dir(tstep)}/pr_{class_partition}_{min_numobj}_{max_numobj}.hdf'
    nclasses = len(readsql(f"SELECT * FROM splitcats WHERE splitid = {splitid} AND seen = 1 AND dataset = '{train_dataset}'",hyperparams))
    if os.path.exists(pname):
        existing_samples = pickle.load(open(pname,'rb'))
        num_batches = (num_samples - len(existing_samples)) // batchsize
        if num_batches <= 0:
            logging.info("done with this nickname")
            return
        else:
            logging.info(f"{len(existing_samples)} samples so far, time to do {num_batches} batches")
    else:
        num_batches = num_samples // batchsize
    equal = np.zeros(num_batches * batchsize)
    assert(class_partition in ['seen','unseen','all'])
    if class_partition == 'seen':
        variety = "testperfect_seen"
    elif class_partition == 'unseen':
        variety = 'testperfect_unseen'
    elif class_partition == 'all':
        variety = 'testperfect_all'
    sample_fn = sample_img(batchsize,splitid,hyperparams,variety=variety,full_img=(hyperparams.ctxop in fullimg_group),dataset=dataset,min_numobj=min_numobj,max_numobj=max_numobj)
    # these two should change.
    sal_name = trackdir + '/saliency-quant' + sess_id
    close_name = trackdir + '/closest-quant' + sess_id
    if track:
        if not os.path.exists(sal_name):
            qualitative.saliency_visualize(args.nickname,splitid,trial) #this will adjust formatting of dataframe and make it the ultimate version.
        sal_data = pickle.load(open(sal_name,'rb'))
        if not os.path.exists(close_name):
            qualitative.closest_visualize(args.nickname,splitid,trial) #this will adjust formatting of dataframe and make it the ultimate version.
        close_data = pickle.load(open(close_name,'rb'))
    if hyperparams.ctxop == "patches": nrows = 3
    elif hyperparams.ctxop == "above_below": nrows = 2
    else: nrows = 1
    with tf.Session() as sess:
        X_placeholder = tf.placeholder(dtype=PRECISION, shape=[nrows * batchsize, img_s,img_s,3],name="X")
        y = tf.placeholder(dtype=PRECISION, shape=[batchsize],name="eq")
        Xfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[nrows * batchsize, img_s,img_s,3],name="Xf")
        Xcfull_placeholder = tf.placeholder(dtype=PRECISION, shape=[nrows * batchsize, img_s,img_s,3],name="Xf")
        bbox_placeholder = tf.placeholder(dtype=PRECISION, shape=[batchsize,4],name="bbox")
        dropout = tf.placeholder(PRECISION)
        parameters = initialize(hyperparams.M,args,hyperparams,nclasses,initialization=hyperparams.initialization)
        sess.run(tf.initialize_all_variables())
        if hyperparams.ctxop is not None: #for the vanilla kinds, we don't want extra loading.
            logging.info(f"DURING ROC LOADING AT WITH t={tstep}")
            npyweights,npybiases = dd.io.load(modeldir + "/" + str(tstep) + ".hdf")
            for k in npyweights.keys():
                sess.run(tf.assign(parameters[0][k],npyweights[k]))
            for k in npybiases.keys():
                sess.run(tf.assign(parameters[1][k],npybiases[k]))
        if hyperparams.ctxop == "DRAW":
            arch_draw = draw_switch(hyperparams)
            network,filt,post,attentionvals,pre_full,attention_boxes = arch_draw(X_placeholder,Xfull_placeholder,Xcfull_placeholder,bbox_placeholder,parameters,dropout,batchsize,True,False,hyperparams.loss_t,hyperparams.baseline_t,hyperparams.numfilts,hyperparams.include_center,decode_arch,hyperparams.stop_grad,stop_pos=hyperparams.decode_stop_pos,keep_resolution=hyperparams.keep_resolution)
            pre = None
        elif hyperparams.ctxop == "above_below":
            network,_,_,_,_ = arch_patch(X_placeholder,Xfull_placeholder,Xcfull_placeholder,parameters,dropout,batchsize,hyperparams.loss_t,hyperparams.ctxop)
        elif hyperparams.ctxop == None:
            network = vanilla(X_placeholder,parameters[0],parameters[1],dropout)
        elif hyperparams.ctxop in ['block_blur','block_intensity']:
            _,network,_,_,_ = arch_block(X_placeholder,Xcfull_placeholder,parameters,dropout,batchsize,hyperparams.ctxop,hyperparams.numfilts,hyperparams.M)
        else:
            logging.critical("Need to add feedforward definition for this context operation, not yet done")
            sys.exit(1)
        for i in range(num_batches):
            if track: logging.info(f"{i/num_batches} len(quant)={len(sal_data)}")
            else: logging.info(f"{i/num_batches} {args.nickname}")
            try:
                if (hyperparams.ctxop in fullimg_group):
                    X,Xfull,bboxs,y,imgnames = sample_fn()
                    Xp,Xpfull,bboxsp,yp,imgnamesp = sample_fn(tomatch=y)
                    Xcfull,alt_bboxs = center_transform(Xfull,imgnames,bboxs)
                    Xcpfull,alt_bboxsp = center_transform(Xpfull,imgnames,bboxsp)
                    feed = {X_placeholder : X,Xfull_placeholder : Xfull, bbox_placeholder: bboxs,dropout : 1.0, Xcfull_placeholder : Xcfull}
                    feedp = {X_placeholder : Xp,Xfull_placeholder : Xpfull, bbox_placeholder: bboxsp,dropout : 1.0, Xcfull_placeholder : Xcpfull}
                else:
                    X,y,imgnames = sample_fn()
                    Xp,yp,imgnamesp = sample_fn(tomatch=y)
                    feed = {X_placeholder : X,dropout : 1.0}
                    feedp = {X_placeholder : Xp, dropout : 1.0}
            except:
                continue
            equal = (y == yp).astype(np.float32)
            if (random.random() < visprob) and hyperparams.ctxop == 'DRAW':
                try:
                    plot_qualitative(sess,filt,post,bboxs,bboxs,feed,tstep,splitid,args.nickname,Xfull,parameters,imgnames,attentionvals,hyperparams,args.trial,t_sess_dir,hyperparams.include_center,class_partition=class_partition)
                except:
                    logging.warn("Failed to run plot_qualitative")
            if track and random.random() < quantifyprob and hyperparams.ctxop == 'DRAW':
                qout = quantify_patches(X,Xfull,imgnames,bboxs,post,args.nickname,splitid)
                for x in range(len(qout['saliency'])):
                    qout['saliency'][x]['tstep'] = tstep
                for x in range(len(qout['closest'])):
                    qout['closest'][x]['tstep'] = tstep
                sal_data = sal_data.append(qout['saliency'])
                close_data = close_data.append(qout['closest'])
            if hyperparams.ctxop in ['above_below','patches']:
                feed = refine_single(hyperparams,feed,Xfull_placeholder,Xfull,X_placeholder,np.array(X),Xcfull_placeholder,Xcfull,bboxs)
                feedp = refine_single(hyperparams,feedp,Xfull_placeholder,Xpfull,X_placeholder,np.array(Xp),Xcfull_placeholder,Xcpfull,bboxs)
            try:
                out1 = sess.run(network,feed)['similarity']
                out2 = sess.run(network,feedp)['similarity']
                for j in range(batchsize):
                    distance_data.append([equal[j],euclidean(out1[j].flatten(),out2[j].flatten())])
            except:
                logging.warn("Failed on batch",i)
    # Ordinary Precision/Recall
    df = pd.DataFrame(distance_data,columns=["equal","distance"])
    if os.path.exists(pname):
        df = df.append(existing_samples)
    df = df.sort(['distance'],ascending=True)
    df.to_hdf(pname,'root')
    # now, save the additional quanitication data I gathered.
    if track:
        sal_data.to_hdf(sal_name,'root')
        close_data.to_hdf(close_name,'root')
    precision,recall = pr(df)
    fig,axes=plt.subplots(ncols=3)
    axes[0].plot(precision)
    axes[1].plot(recall)
    axes[2].plot(recall,precision)
    plt.savefig(f'{sess_dir}/pr_{time.time()}.png')
   
def run_store(spacing,randomized=True):
    t0 = time.time()
    savesteps = [(x,int(x.split(".")[0])) for x in os.listdir(modeldir) if x]
    if randomized:
        random.shuffle(savesteps)
    else:
        savesteps.sort(key=lambda x:x[1])
    for i,savestep in enumerate(savesteps):
        store(args.nickname,0,args.canmax,args.splitid,args,savestep[1],modeldir,batchsize=7,savemethod='hdf')

def run_test( ):
    nclasses = len(split['known'])
    t0 = time.time()
    savesteps = [(x,int(x.split(".")[0])) for x in os.listdir(modeldir) if x]
    savesteps.sort(key=lambda x:x[1])
    random.shuffle(savesteps)
    for savestep in reversed(savesteps):
        test(splitid,args.nickname,args,savestep,nclasses,split,args.num_test)

def run_train( ):
    if hyperparams.task == "discovery": train(args,hyperparams,savemethod="hdf")
    elif hyperparams.task == "classify": train_classify(args,hyperparams,savemethod="hdf")
     
if args.action == "baselines": run_baselines()
elif args.action == "store": run_store(args.spacing)
elif args.action == "store_at":
    if args.tstep == "max":
        models = [int(x.split('.')[0]) for x in os.listdir(modeldir) if x.split('.')[1] == "hdf"]
        savestep = max(models)
    else:
        savestep = int(args.tstep)
    store(hyperparams,args.nickname,0,args.canmax,args.splitid,args,savestep,modeldir,args.trial,savemethod="hdf",dataset=args.transfer_dataset,train_dataset=args.train_dataset)
    store(hyperparams,args.nickname,1,args.canmax,args.splitid,args,savestep,modeldir,args.trial,savemethod="hdf",dataset=args.transfer_dataset,train_dataset=args.train_dataset)
elif args.action == "test":
    run_test()
elif args.action == "train":
    run_train()
elif args.action == "pr":
    #for class_partition in ["seen","unseen","all"]:
    pr_run(args.nsamples,args.batchsize,args.splitid,args.timestep,class_partition=args.class_partition,dataset=args.transfer_dataset,train_dataset=args.train_dataset,min_numobj=args.min_numobj,max_numobj=args.max_numobj)
elif args.action == 'prplot':
    # args.timestep should be a string that evals to a list of possible timesteps to include.
    #constraints = {'dataset' : args.transfer_dataset,'timestep' : eval(args.timestep),'nicknames' : args.nicknames.split(','),'splitid' : args.splitid}
    timesteps = eval(args.timestep)
    if type(timesteps) == int:
        timesteps = [timesteps]
    constraints = {'dataset' : args.transfer_dataset,'timestep' : timesteps,'nicknames' : ['DRAW4-dual-shared','vanilla-vgg'],'splitid' : args.splitid}
    #constraints = {'dataset' : 'COCO','timestep' : [29999],'nicknames' : ['DRAW4-noctx','DRAW5-dual-shared','DRAW5-shared-nocenter','DRAW4-contrastive-shared','DRAW4-dual-shared-keep-stopgrad','DRAW3-dual-shared','conv-block-blur','above-below','DRAW4-dual-shared-early','DRAW4-shared-attentiononly','DRAW4-dual-shared-stopgrad','vanilla-vgg','DRAW4-dual-shared','DRAW4-shared-fixedbiasonly'],'splitid' : 7}
    pr_all(constraints)
elif args.action == "context":
    context_similarity(args,hyperparams,restart_num=args.timestep)
elif args.action == "complete_vis":
    complete_visualizations(sess_dir)
elif args.action == "go": #actually the process is interactive and slow enough I don't tend to do this.
    run_train()
    run_test()
    run_baselines()
    run_store()
