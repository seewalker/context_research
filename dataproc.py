'''
Alexander Seewald 2016
aseewald@indiana.edu

Functions here produce the SQL tables used in learning.py, along with pickled collections of information about candidates
and (depending on type of experiment) superpixels. 
'''
import random
import copy
import argparse
import numpy as np
import psycopg2
import pickle
import sqlite3
import time
import gzip
import sys
import os
import tempfile
import math
import subprocess
import multiprocessing
from collections import Counter
from skimage.io import imsave, imread
from skimage import img_as_float
from scipy.misc import imresize
from scipy.stats import entropy
from skimage.color import rgb2gray, rgb2lab
from skimage.feature import ORB
# Ignore these warnings because sklearn is highly annoying when running 'process' function even though it works fine.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.cluster import KMeans
import scipy.misc
import scipy.ndimage.interpolation
import pandas as pd
import scipy.io
import deepdish as dd
import xml.etree.ElementTree as ET
from collections import OrderedDict
import hyperparams as hp
import constants
from utils import *
if "sqlite" in constants.dbtypes:
    import sqlite3
if "postgres" in constants.dbtypes:
    import psycopg2
import pixelwise

__author__ = "Alex Seewald"

def context_awareness(superpixels):
    return np.mean([np.max(sp.object_distribution) for sp in superpixels])
    
def easiness(candidate,superpixels) -> float:
    return candidate.score + context_awareness(superpixels)

def expand_candidate(bbox,imgshape,decr=0.01):
    '''
    Returns a new bbox with extra surrounding image data to incorperate context information.
    If this box would leave bounds of image, simply use the original bounding box.
    '''
    miny,maxy,minx,maxx = bbox
    dy,dx = maxy-miny,maxx-minx
    expand_y = min(frame_expand_max * imgshape[0],relative_expand_max * dy)
    expand_x = min(frame_expand_max * imgshape[1],relative_expand_max * dx)
    within_frame = False
    while not within_frame:
        minyp,maxyp = int(min(bbox[0],miny - expand_y)), int(max(bbox[1],maxy + expand_y))
        minxp,maxxp = int(min(minx - expand_x,bbox[2])), int(min(maxx + expand_x,bbox[3]))
        within_frame = (minyp >= 0) and (maxyp <= imgshape[0] - 1) and (minxp >= 0) and (maxxp <= imgshape[1] - 1)
        if within_frame:
            expanded = [minyp,maxyp,minxp,maxxp] 
        else:
            expand_y /= (1 + decr)
            expand_x /= (1 + decr)
    else:
        return [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])], (expand_y,expand_x)

def imgsize(hyperparams):
    dosql("CREATE TABLE IF NOT EXISTS imgsize (imgname TEXT, height INT, width INT, PRIMARY KEY(imgname))",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS avgcolor(imgname TEXT, channel INT, meanval FLOAT)",hyperparams)
    for dirname in [hyperparams.root("train_images"),hyperparams.root("val_images")]:
        fs = os.listdir(dirname)
        for f in reversed(fs):
            im = imread(os.path.join(dirname,f))
            trimname = os.path.splitext(f)[0]
            h,w = im.shape[0],im.shape[1]
            try:
                dosql(f"INSERT INTO imgsize VALUES ('{trimname}',{h},{w})",hyperparams)
            except:
                continue
            if len(im.shape) != 3:
                continue
            amt = np.sum(im,axis=(0,1)) / (h * w)
            for chan in range(im.shape[2]): 
                dosql(f"INSERT INTO avgcolor VALUES ('{trimname}',{chan},{amt[chan]})",hyperparams)

def addclass(candidates,splitid,imgNames,classifier_t):
    '''
    Most efficient if using many candidates
    '''
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i, candidate in enumerate(candidates):
            out = imresize(candidate.img_patch, (constants.cnn_h[classifier_t], constants.cnn_w[classifier_t]))
            imsave(tmpdirname + "/" + str(i) + ".jpg",out)
        try:
            subprocess.check_call([hyperparams.anaconda2,"hacks/add_classes.py",tmpdirname,classifier_t,str(len(candidates)),str(splitid)])
            print("successfully added classes for " + str(imgNames))
        except:
            sys.exit(1)
        distributions = np.load(tmpdirname + "/distr.npy")
        if not np.array_equal(distributions,np.nan_to_num(distributions)):
            print(f"warning, got nan answers at imgname={imgName}, splitid={splitid}. continuing treating it as a zero vector")
            distributions = np.nan_to_num(distributions)
    return distributions

def addsplit(hyperparams):
    '''
    Adds a split after the fact.
    Take 10 images -> Extract all candidates -> Run CNN -> Save in 
    '''
    insert_ifnotexists("SELECT canid FROM entropy WHERE imgName = '{}' AND splitid = {} AND canid = {} AND type = '{}' AND classifiertype = '{}'".format(batch_imgnames[j],splitid,k,hyperparams.candidate_method,classifier_t),"INSERT INTO entropy VALUES('{}',{},{},'{}','{}',{})".format(batch_imgnames[j],splitid,k,hyperparams.candidate_method,classifier_t,entropy(candidate.object_distribution[classifier_t][splitid])))
    insert_ifnotexists("SELECT canid FROM candidate_object_distribution WHERE imgName = '{}' AND splitid = {} AND canid = {} AND type = '{}'".format(batch_imgnames[j],splitid,k,hyperparams.candidate_method),"INSERT INTO candidate_object_distribution VALUES('{}',{},{},'{}','{}','{}')".format(batch_imgnames[j],splitid,k,hyperparams.candidate_method,classifier_t,str(candidate.object_distribution[classifier_t][splitid].tolist())))

def blockit(hyperparams,forwards,nickname,tstep,batchsize,takenum,adding_sps=False):
    if forwards == True:
        allnames = list(hyperparams.val_names())
        allblocks = chunks(allnames,batchsize)
    elif forwards == False:
        allnames = list(reversed(list(hyperparams.val_names())))
        allblocks = chunks(allnames,batchsize)
    elif forwards == "shuffle":
        allnames = list(hyperparams.val_names())
        allblocks = list(chunks(allnames,batchsize))
        random.shuffle(allblocks)
    blocks = []
    canout = []
    # previously used this to determine on the first split.
    for block in allblocks:
        spnames = ["{}/{}_{}_{}.pkl.gz".format(hyperparams.root("val_superpixels"),imgName,nickname,tstep) for imgName in block]
        canmarkers = ["{}/{}_{}_{}".format(hyperparams.root("val_canmarkers"),imgName,nickname,tstep) for imgName in block]
        if not adding_sps:
            # I used to have this line, what was it doing??
            #if all([os.path.exists(spname) for spname in spnames]): #adding candidates, don't bother doing it when not all sps exist.
            if not all([os.path.exists(canname) for canname in canmarkers]):
                blocks.append(block)
                canout.append(canmarkers)
        else: #adding superpixels, so do it when not all of them exist.
            if not all([os.path.exists(spname) for spname in spnames]): #looking at the blocks which are todo.
                blocks.append(block)
        if len(blocks) == takenum:
            break
    return blocks,canout

def hdf_restore(weights,biases,modeldir,t,sess):
    npy_w,npy_b = dd.io.load(modeldir + "/" + str(t) + ".hdf")
    sess.run(tf.initialize_all_variables())
    for k in weights.keys():
        sess.run(weights[k].assign(npy_w[k]) )
    for k in biases.keys():
        sess.run(biases[k].assign(npy_b[k]))

def add_can_pred(hyperparams,nickname,splitid,modeldir,tstep,hack_diversity_scale=1,numfuse=2,savemethod="hdf",dbinsert=False):
    '''
    Note; the same batchsize used in mksuperpixels must be used here.

    This is used for object graph.
    '''
    # now, use 'canid' for either int-like in 
    dosql("CREATE TABLE IF NOT EXISTS perfect_candistrs(nickname TEXT, vec TEXT, splitid INT, imgname TEXT, patchname TEXT)",hyperparams)
    dosql("CREATE TABLE IF NOT EXISTS candistrs(nickname TEXT, vec TEXT, splitid INT, imgname TEXT, canid INT)",hyperparams)
    perfect_bboxs = readsql("SELECT * FROM perfect_bbox WHERE isxl = 0 AND isexpanded = 0 AND patchname NOT LIKE '%ctxpatches%'",hyperparams)
    perfect_bboxs['imgname'] = list(map(lambda x: os.path.splitext(os.path.split(x)[1])[0],perfect_bboxs['imgname']))
    can_bboxs = readsql("SELECT * From candidate_bbox",hyperparams)
    conn = psycopg2.connect(**hyperparams.pg)
    cursor = conn.cursor()
    # should be +1 but it seems I messed up and have an additional unused class (due to the +1 naming for None inconsistency).
    num_classes = len(readsql("SELECT * FROM splitcats WHERE splitid = {} AND dataset = 'COCO' AND seen = 1".format(splitid),hyperparams))
    print("About to use nickname={},iter={},splitid={},num_classes={} to make candiates".format(nickname,tstep,splitid,num_classes))
    markerdir = hyperparams.root("val_canmarkers")
    if not os.path.exists(markerdir):
        subprocess.call(["mkdir",markerdir])
    with tf.Session() as sess:
        weights,biases = pixelwise.initialize(hyperparams.pixhp,num_classes,numfuse,hyperparams.pixhp.opt_t)
        batchsize = 12
        X_placeholder = tf.placeholder(tf.float32,shape=[batchsize,224,224,3])
        dropout = tf.placeholder(tf.float32)
        bgscale = tf.placeholder(tf.float32)
        ptsv = open('/fast-data/aseewald/pcan_{}_{}'.format(time.time(),nickname),'w')
        ctsv = open('/fast-data/aseewald/ccan_{}_{}'.format(time.time(),nickname),'w')
        aname = os.path.join(modeldir,"alphas-{}".format(tstep))
        if not os.path.exists(aname) and numfuse == 2:
            alphas = {'upsample5' : 0.333, 'upsample4' : 0.333}
        elif not os.path.exists(aname) and numfuse == 1:
            alphas = {'upsample5' : 0.5}
        elif os.path.path.exists(aname):
            alphas = pickle.load(open(aname,'rb'))
        else:
            assert(False)
        arch = pixelwise.archof('vgg',hyperparams.pixhp.opt_t)
        net = arch(X_placeholder,weights,biases,dropout,num_classes,batchsize,numfuse,alphas=alphas)
        if savemethod == "hdf":
            hdf_restore(weights,biases,modeldir,tstep,sess)
        elif savemethod == "tf":
            saver = tf.train.Saver( )
            saver.restore(sess,modeldir + '/' + 'model-' + str(tstep))
        times = []
        while True:
            blocks,canmarkers = blockit(hyperparams,"shuffle",nickname,tstep,batchsize,5,adding_sps=False)
            if len(blocks) == 0:
                print("All done")
                break
            imgNames,canmarkers = blocks[0],canmarkers[0]
            if (len(times) > 0):
                print("{} blocks left, estimated minutes left {}".format(len(blocks),len(blocks) * np.mean(times) / 60))
            t0 = time.time()
            raw_imgs = [img_as_float(imread(os.path.join(hyperparams.root('val_images'),img + ".jpg"))) for img in imgNames]
            imgs = [imread_wrap(os.path.join(hyperparams.root('val_images'),img + ".jpg"),hyperparams) for img in imgNames]
            superpix_segs = []
            for imgName in imgNames:
                try:
                    superpix_segs.append(np.genfromtxt("{}/{}.jpg_{}.csv.gz".format(hyperparams.root("val_superpixelation"), imgName, str(constants.superpixel_amount)), delimiter=","))
                except:
                    superpix_segs.append(None) 
            feed = {X_placeholder : imgs,dropout : 1.0}
            pix_res = sess.run(net,feed_dict=feed)
            pix_imgs = hack_diversity_scale * pix_res['net'].reshape((batchsize,224,224,num_classes + 1))
            pix_imgs = normalize_unscaled_logits(pix_imgs)
            pinserts,cinserts = [], []
            imgshapes = [img.shape for img in raw_imgs]
            for i,imgName in enumerate(imgNames):
                shape = imgshapes[i]
                rel_y = shape[0] / 224
                rel_x = shape[1] / 224
                pboxs = perfect_bboxs[perfect_bboxs['imgname'] == imgName]
                cboxs = can_bboxs[can_bboxs['imgname'] == imgName]
                for _,pbox in pboxs.iterrows():
                    miny,maxy = int(pbox['miny'] / rel_y),int(pbox['maxy'] / rel_y)
                    minx,maxx = int(pbox['minx'] / rel_x),int(pbox['maxx'] / rel_x)
                    if maxy == miny or maxx == minx:
                        print("Warning, boundaries are same somehow with perfect,imgname={}pname={}".format(imgName,pbox['patchname']))
                        continue
                    try:
                        pixs = np.nan_to_num(pix_imgs[i][miny:maxy,minx:maxx])
                        pix = np.mean(pixs.reshape((pixs.size // (num_classes+1),num_classes+1)),axis=0)
                        if np.any(np.isnan(pix)):
                            assert False
                        pinserts.append((imgName,pbox['patchname'],splitid,nickname,floatserial(pix,10),tstep))
                    except:
                        print("failed on perfect")
                        continue
                for _,cbox in cboxs.iterrows():
                    miny,maxy = int(cbox['min'] / rel_y),int(cbox['maxy'] / rel_y)
                    minx,maxx = int(cbox['minx'] / rel_x),int(cbox['maxx'] / rel_x)
                    if maxy == miny or maxx == minx:
                        print("Warning, boundaries are same somehow with candidates,imgname={}cid={}".format(imgName,cbox['cid']))
                        continue
                    try:
                        pixs = np.nan_to_num(pix_imgs[i][miny:maxy,minx:maxx])
                        pix = np.mean(pixs.reshape((pixs.size // (num_classes+1),num_classes+1)),axis=0)
                        if np.any(np.isnan(pix)):
                            assert False
                        cinserts.append((imgName,floatserial(pix,12),splitid,nickname,cbox['canid'],tstep))
                    except:
                        print("failed on imperfect")
            if dbinsert:
                cursor.executemany("INSERT INTO perfect_candistrs VALUES (%s,%s,%s,%s,%s,%s)",pinserts)
                cursor.executemany("INSERT INTO candistrs VALUES (%s,%s,%s,%s,%s,%s)",cinserts)
                conn.commit()
            else:
                [ptsv.write('\t'.join(map(str,pinsert)) + '\n') for pinsert in pinserts]
                [ctsv.write('\t'.join(map(str,cinsert)) + '\n') for cinsert in cinserts]
            pinserts,cinserts = [], []
            times.append(time.time() - t0)
            for canmarker in canmarkers:
                subprocess.call(["touch",canmarker])

def mksuperpixels(hyperparams,dataset,splitid,numfuse,trial,nickname,savemethod="hdf",forwards="shuffle",noredo=False,pickleit=False,newSps=False,tstep=None,careful=False):
    '''
    savemethod is either "hdf" or "tf" depending on whether fully convolutional network saved its parameters as hdf5 files or as tensorflow checkpoints.
    Note: I had trouble getting checkpoint-style parameters loaded correctly, so its easiest to stick with hdf5 and so that is the default.

    Most of the parameters here are to locate the modeldir.
    '''
    if not os.path.exists('sps'):
        subprocess.call(["mkdir","sps"])
    if os.environ['CUDA_VISIBLE_DEVICES'] == '0':
        tsv = open('sps/{}_{}.tsv'.format(time.time(),tstep),'a')
    elif os.environ['CUDA_VISIBLE_DEVICES'] == '1':
        tsv = open('/fast-data/aseewald/sps/{}_{}.tsv'.format(time.time(),tstep),'a')
    elif os.environ['CUDA_VISIBLE_DEVICES'] == '2':
        tsv = open('/ssd/aseewald/sps/{}_{}.tsv'.format(time.time(),tstep),'a')
    elif os.environ['CUDA_VISIBLE_DEVICES'] == '3':
        tsv = open('/ssd/aseewald/sps/{}_{}.tsv'.format(time.time(),tstep),'a')
    create_tables()
    assert(savemethod in ["hdf","tf"])
    if not os.path.exists(hyperparams.root("val_superpixels")):
        subprocess.call(["mkdir",hyperparams.root("val_superpixels")])
    split = readsql("SELECT * FROM splitcats WHERE dataset = '{}' AND splitid = {}".format(dataset,splitid),hyperparams)
    num_classes = len(split[split['seen'] == 1])
    print("Working with dataset={},splitid={}, which has {} classes".format(dataset,splitid,num_classes))
    imgcount = 0
    modeldir = hyperparams.root('cnn/fullyconv_{}_{}_{}_{}_{}'.format(nickname,trial,splitid,numfuse,dataset))
    if savemethod == "tf":
        if not os.path.exists(modeldir):
            print(modeldir," not yet trained")
            return
    else:
        if tstep is not None:
            subprocess.call(["cp",os.path.join(modeldir,str(tstep) + ".hdf"),"best.hdf"])
        if not os.path.exists(modeldir):
            print(modeldir," not yet trained")
            return False
        if not os.path.exists(modeldir + "/" + "best.hdf"):
            print("Save the model you want as best.hdf")
            return False
    print("Loading network for splitid={}".format(splitid))
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1/nsess)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session() as sess:
        weights,biases = pixelwise.initialize(num_classes,numfuse)
        batchsize = 24
        X_placeholder = tf.placeholder(tf.float32,shape=[batchsize,224,224,3])
        dropout = tf.placeholder(tf.float32)
        bgscale = tf.placeholder(tf.float32)
        aname = os.path.join(modeldir,"alphas-{}".format(tstep))
        if not os.path.exists(aname) and numfuse == 2:
            alphas = {'upsample5' : 0.333, 'upsample4' : 0.333}
        elif not os.path.exists(aname) and numfuse == 1:
            alphas = {'upsample5' : 0.5}
        elif os.path.path.exists(aname):
            alphas = pickle.load(open(aname,'rb'))
        else:
            assert(False)
        net = pixelwise.arch_vgg(X_placeholder, weights,biases,dropout,num_classes,batchsize,numfuse,alphas=alphas)
        if savemethod == "tf":
            saver = tf.train.Saver( )
            ckpt = tf.train.get_checkpoint_state(modeldir)
            if tstep is None:
                tstep = int(ckpt.model_checkpoint_path.split("-")[-1])
                saver.restore(sess,ckpt.model_checkpoint_path)
            else:
                saver.restore(sess,os.path.join(modeldir,'model-'+str(tstep)))
            if ckpt == None:
                sess.close()
                return
            print("Making superpixels from timestep={}".format(tstep))
        elif savemethod == "hdf":
            npy_w,npy_b = dd.io.load(modeldir + "/" + "best.hdf")
            assert(npy_w.keys() == weights.keys() and npy_b.keys() == biases.keys())
            sess.run(tf.initialize_all_variables())
            for k in weights.keys():
                sess.run(weights[k].assign(npy_w[k]) )
            for k in biases.keys():
                sess.run(biases[k].assign(npy_b[k]))
        while True:
            blocks,_ = blockit(forwards,nickname,tstep,batchsize,float("inf"),adding_sps=True)
            print("{} blocks left".format(len(blocks)))
            imgNames = random.choice(blocks)
            if len(imgNames) < batchsize:
                print("Not properly handled non mutliple yet. continuing...")
                continue
            raw_imgs = [imread(os.path.join(hyperparams.root('val_images'),img + ".jpg")) for img in imgNames]
            imgs = [imread_wrap(os.path.join(hyperparams.root('val_images'),img + ".jpg")) for img in imgNames]
            superpix_segs = []
            for imgName in imgNames:
                try:
                    superpix_segs.append(np.genfromtxt("{}/{}.jpg_{}.csv.gz".format(hyperparams.root("val_superpixelation"), imgName, str(constants.superpixel_amount)), delimiter=","))
                except:
                    superpix_segs.append(None) 
            sanity_number = 150 #must be a bit above the desired number of superpixels.
            feed = {X_placeholder : imgs,dropout : 1.0}
            # why is this returning tensors???
            pix_res = sess.run(net,feed_dict=feed)
            # fuse the outputs here because that's not done in the arch function.
            pix_imgs = pix_res['net'].reshape((batchsize,224,224,num_classes + 1))
            pix_imgs = normalize_unscaled_logits(pix_imgs)
            spnames = ["{}/{}_{}_{}.pkl.gz".format(hyperparams.root("val_superpixels"),imgName,nickname,tstep) for imgName in imgNames]
            t0 = time.time()
            if all([os.path.exists(spname) for spname in spnames]) and pickleit:
                batch_superpixels = [pickle.loads(gzip.open(spname,"rb").read()) for spname in spnames]
                print("Unpickling took {} seconds".format(time.time() - t0))
            else:
                batch_superpixels = []
                for k in range(batchsize):
                    superpix_seg,imgName,img = superpix_segs[k],imgNames[k],raw_imgs[k]
                    superpixels = []
                    for i, spid in enumerate(np.unique(superpix_seg)[1:]):
                        superpixel = Superpixel(imgName,img,superpix_seg,spid)
                        superpixels.append(superpixel)
                        if newSps:
                            a,b,c,d = superpixel.bbox
                            insert_ifnotexists("SELECT * FROM sp_bbox WHERE imgName = '{}' AND spid = {}".format(imgName,i),"INSERT INTO sp_bbox VALUES('{}',{},{},{},{},{},'{}')".format(imgName,i,a,b,c,d,dataset),whichdb="postgres")
                            insert_ifnotexists("SELECT * FROM sp_centroid WHERE imgName = '{}' AND spid = {}".format(imgName,i),"INSERT INTO sp_centroid VALUES('{}',{},{},{},'{}')".format(imgName,i,superpixel.centroid[0],superpixel.centroid[1],dataset),whichdb="postgres")
                    if not os.path.exists(spnames[k]) and pickleit:
                        tsup = gzip.open(spnames[k],"wb")
                        tsup.write(pickle.dumps(superpixels))
                        tsup.close()
                    else:
                        subprocess.call(["touch",spnames[k]])
                    batch_superpixels.append(superpixels)
                print("Making and saving superpixels took {} seconds".format(time.time() - t0))
            tb = time.time()
            for k in range(batchsize):
                superpixels = batch_superpixels[k]
                pix_img,img,imgName = pix_imgs[k],raw_imgs[k],imgNames[k]
                pix_reshaped = np.zeros((img.shape[0],img.shape[1],pix_img.shape[2]))
                for chan in range(pix_img.shape[2]):
                    pix_reshaped[:,:,chan] = scipy.ndimage.interpolation.zoom(pix_img[:,:,chan],(img.shape[0]/pix_img.shape[0],img.shape[1]/pix_img.shape[1]))
                for spid,superpixel in enumerate(superpixels):
                    # looks weird, but this is the correct numpy syntax for getting the right pixels from the {(y1,x1) ... } set.
                    pixmat = np.array(list(superpixel.chosen_pixels)).T
                    pixs = pix_reshaped[[pixmat[0],pixmat[1]]]
                    object_distribution = np.max(pixs,axis=0)
                    if careful:
                        insert_ifnotexists("SELECT * FROM sp_object_distribution WHERE imgName = '{}' AND splitid = {} AND spid = {} AND method = '{}' AND dataset = '{}' AND nickname = '{}'".format(imgName,splitid,spid,hyperparams.candidate_method,dataset,nickname),"INSERT INTO sp_object_distribution VALUES('{}',{},{},'{}','{}','{}','{}')".format(imgName,splitid,spid,hyperparams.candidate_method,str(object_distribution.tolist()),dataset,nickname))
                    else:
                        tsv.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(imgName,splitid,spid,hyperparams.candidate_method,str(object_distribution.tolist()),dataset,nickname,tstep))
                imgcount += 1
            print("Db insert took {} seconds".format(time.time() - tb))

def mkcandidates(hyperparams,available,expand_t="none",skipclass=True):
    '''
    Information is maintained in sqlite database for quick access and in pickled objects which joins everything together and is comprehensive.
    '''
    create_tables()
    codebook = pickle.load(open("{}/codebook".format(hyperparams.root()),"rb"))
    runObjectness(Nper=14)
    if hyperparams.candidate_method == "objectness":
        cans = pickle.load(open(hyperparams.root("objectness/objness_chosen.pkl"),'rb'))
    verified_completed = 35000
    imgcount = verified_completed
    # Because we start at 'verified_completed', imgcount has to have relative not absolute meaning.
    subprocess.call(["mkdir",hyperparams.root("val_candidates")])
    subprocess.call(["mkdir",hyperparams.root("val_candidateimgs")])
    distr_batch_candidates,batch_imgmarkers,batch_imgnames,batchsize = [],[],[],30
    (todo,dataset) = (hyperparams.val_names_gt(),"pascal") if hyperparams.experimentName == "VOC2008" else (hyperparams.val_names(),"COCO")
    for imgName in list(todo)[verified_completed:]:
        print("working on " + imgName, imgcount/len(todo))
        if hyperparams.objectness_method == "whatisanobject?" and not os.path.exists('/data/aseewald/COCO/objectness/' + imgName + ".jpg.mat"):
            print("data not ready for {}".format(imgName))
            continue
        img = imread("{}/{}.jpg".format(hyperparams.root("val_images"),imgName))
        extractor = ORB()
        extractor.detect_and_extract(rgb2gray(img))
        if type(hyperparams) == hp.GreedyParams:
            try:
                superpix_seg = np.genfromtxt("{}/{}.jpg_{}.csv.gz".format(hyperparams.root("val_superpixelation"), imgName, str(constants.superpixel_amount)), delimiter=",")
            except:
                print("superpixelation for {} not found. continuing")
                continue
        sanity_number = 150 #must be a bit above the desired number of superpixels.
        key = "_".join([imgName,hyperparams.candidate_method])
        old_key = "_".join([imgName,str(0),hyperparams.candidate_method])
        old_exists,new_exists = os.path.exists("{}/{}.pkl.gz".format(hyperparams.root("val_candidates"),old_key)), os.path.exists("{}/{}.pkl.gz".format(hyperparams.root("val_candidates"),key))
        #dbExists = len(readsql("SELECT * FROM candidate_object_distribution WHERE imgname = '{}' AND splitid = {} AND canid = {} AND type = '{}'".format(imgName,splitid,0,hyperparams.candidate_method))) > 0
        if old_exists and not (new_exists):
            print("should eventually add further classes here")
        if old_exists or new_exists:
            # do updates if applicable.
            print(key + "exists. continuing")
            imgcount += 1
            continue # for now, don't need to worry about rest of block.
            candidates = pickle.loads(gzip.open("{}/{}.pkl.gz".format(hyperparams.root("val_candidates"),key),"rb").read())
            if (splitid == 0) and not all([os.path.exists(hyperparams.root("val_candidateimgs/{}_{}.jpg".format(key,i))) for i in range(len(candidates))]):
                for i,candidate in enumerate(candidates):
                    imsave(hyperparams.root("val_candidateimgs/{}_{}.jpg".format(key,i)),candidate.img_patch)
            if all([hasattr(candidate,'sal_img') for candidate in candidates]):
                continue
            for candidate in candidates:
                try:
                    candidate.sal_img = imread(hyperparams.root("val_saliency/{}.jpg".format(imgName)))[candidate.bbox[0]:candidate.bbox[1],candidate.bbox[2]:candidate.bbox[3]]
                except:
                    continue
            tcan = gzip.open("{}/{}.pkl.gz".format(hyperparams.root("val_candidates"),key),"wb")
            tcan.write(pickle.dumps(candidates))
            tcan.close()
            continue
        seg_counter = 0
        insert_ifnotexists("SELECT * FROM imgsize WHERE imgname = '{}'".format(imgName),"INSERT INTO imgsize VALUES('{}',{},{})".format(imgName,img.shape[0],img.shape[1]))
        if hyperparams.candidate_method == "objectness":
            canid = 0
            imgcans = cans[cans['imgname'] == imgName]
            candidates = []
            for rowid,row in imgcans.iterrows():
                bbox_raw = [row['miny'],row['maxy'],row['minx'],row['maxx']]
                # these conditions aren't exactly right, but I've moved past using expansions.
                if expand_t == "regular":
                    (miny,maxy,minx,maxx),(expand_y,expand_x) = expand_candidate(bbox_raw,img.shape)
                elif expand_t == "xl":
                    (miny,maxy,minx,maxx),(expand_y,expand_x) = expand_candidate(bbox_raw,img.shape)
                else:
                    miny,maxy,minx,maxx = bbox_raw
                bbox,score = [miny,maxy,minx,maxx],row['objness']
                candidate = BING_candidate(imgName,img,bbox,bbox_raw,score,extractor.keypoints,extractor.descriptors,codebook,canid)
                candidates.append(candidate)
                saveout = hyperparams.root("val_candidateimgs/{}_{}.jpg".format(key,canid))
                if not os.path.exists(saveout):
                    imsave(saveout,candidate.img_patch)
                insert_ifnotexists("SELECT * FROM candidate_bbox WHERE imgName = '{}' AND canid = {} AND type = 'objectness' AND dataset = '{}'".format(imgName,canid,dataset),"INSERT INTO candidate_bbox VALUES('{}',{},'objectness',{},{},{},{},'{}')".format(imgName,canid,miny,maxy,minx,maxx,dataset))
                insert_ifnotexists("SELECT * FROM candidate_centroid WHERE imgName = '{}' AND canid = {} AND type = 'objectness' AND dataset = '{}'".format(imgName,canid,dataset),"INSERT INTO candidate_centroid VALUES('{}',{},'objectness',{},{},'{}')".format(imgName,canid,int((maxy-miny)/2),int((maxx-minx)/2),dataset))
                print("Maybe inserted new")
                canid += 1
            distr_batch_candidates = distr_batch_candidates + candidates
            batch_imgnames.append(imgName)
            if len(batch_imgmarkers) == 0:
                batch_imgmarkers.append(canid)
            else:
                assert(batch_imgmarkers[-1] == max(batch_imgmarkers))
                batch_imgmarkers.append(batch_imgmarkers[-1] + canid) #at this point canid has been incremented as many times as candidates are added.
                assert(len(distr_batch_candidates) == batch_imgmarkers[-1])
        elif hyperparams.candidate_method == "segment":
            for num_segs in constants.segment_amounts:
                seg = np.genfromtxt("{}/{}_{}.csv.gz".format(hyperparams.root("val_segmentation"), imgName, num_segs), delimiter=",")
                if not (np.unique(seg).size <= (num_segs + 1)):
                    above_mask = seg > num_segs
                    seg[above_mask] = 0
                    print("warning: more segs than are specified, there are {}".format(np.count_nonzero(above_mask)))
                for i, segid in enumerate(np.unique(seg)[1:]):
                    if not pickledExists:
                        candidate = Segment_candidate(imgName,split,img,seg,segid,extractor.keypoints,extractor.descriptors,codebook)
                        candidates.append(candidate)
                    else:
                        candidate = candidates[seg_counter]
                    a,b,c,d = candidate.bbox
                    imsave(hyperparams.root("val_candidateimgs/{}_{}.jpg".format(key,seg_counter)),candidate.raw_img)
                    insert_ifnotexists("SELECT * FROM candidate_bbox WHERE imgName = '{}' AND canid = {} AND type = 'segment'".format(imgName,seg_counter),"INSERT INTO candidate_bbox VALUES('{}',{},'segment',{},{},{},{})".format(imgName,seg_counter,a,b,c,d))
                    insert_ifnotexists("SELECT * FROM candidate_centroid WHERE imgName = '{}' AND canid = {} AND type = 'segment'".format(imgName,seg_counter),"INSERT INTO candidate_centroid VALUES('{}',{},'segment',{},{})".format(imgName,seg_counter,candidate.centroid[0],candidate.centroid[1]))
                    seg_counter += 1
            distr_batch_candidates = distr_batch_candidates + candidates
            batch_imgnames.append(imgName)
        # Run CNNs on batch of candidates.
        if imgcount % batchsize == (batchsize - 1) and not skipclass:
            for splitid in range(len(hyperparams.possible_splits)):
                k=0
                if splitid not in available:
                    continue
                distributions = addclass(distr_batch_candidates,splitid,imgName,classifier_t)
                for i, candidate in enumerate(distr_batch_candidates):
                    candidate.object_distribution[splitid] = distributions[i]
                    # Using canid rather than '*' because sqlite and postgres disagree about * for equality testing i do.
                    insert_ifnotexists("SELECT canid FROM entropy WHERE imgName = '{}' AND splitid = {} AND canid = {} AND type = '{}'".format(batch_imgnames[j],splitid,k,hyperparams.candidate_method),"INSERT INTO entropy VALUES('{}',{},{},'{}','{}',{})".format(batch_imgnames[j],splitid,k,hyperparams.candidate_method,entropy(candidate.object_distribution[splitid])))
                    insert_ifnotexists("SELECT canid FROM candidate_object_distribution WHERE imgName = '{}' AND splitid = {} AND canid = {} AND type = '{}' AND dataset = '{}'".format(batch_imgnames[j],splitid,k,hyperparams.candidate_method,dataset),"INSERT INTO candidate_object_distribution VALUES('{}',{},{},'{}','{}','{}')".format(batch_imgnames[j],splitid,k,hyperparams.candidate_method,str(candidate.object_distribution[classifier_t][splitid].tolist()),datset))
                    k += 1
            batch_imgnames,batch_imgmarkers,distr_batch_candidates = [],[],[]
        if not os.path.exists(hyperparams.root("val_candidates")):
            subprocess.call(["mkdir",hyperparams.root("val_candidates")])
        tcan = gzip.open("{}/{}.pkl.gz".format(hyperparams.root("val_candidates"),key),"wb")
        tcan.write(pickle.dumps(candidates))
        tcan.close()
        imgcount += 1
    if skipClass:
        subprocess.check_call([hyperparams.anaconda2,"hacks/add_gt.py"])
        # Add these once candidate easiness is known.
        for splitid, split in enumerate(hyperparams.possible_splits[0:4]):
            for imgcount,imgName in enumerate(list(hyperparams.val_names())[verified_completed:]):
                for j,candidate in enumerate(candidates):
                    candidate.easiness = easiness(candidate,superpixels=superpixels)
                    insert_ifnotexists("SELECT * FROM easiness WHERE imgname = '{}' AND splitid = {} AND canid = {} AND type = '{}' AND classifier_t = '{}'".format(imgName,splitid,j,hyperparams.candidate_method,),"INSERT INTO easiness VALUES('{}',{},{},'{}',{})".format(imgName,splitid,j,hyperparams.candidate_method,candidate.easiness))

def mksplit(hyperparams,orig_name):
    '''
    This is used for datasets which do not already split up their training and validation sets.
    for pascal2008, this will be called on SegmentationClass.
    '''
    train, val = open(hyperparams.root("train.txt"),"w"), open(hyperparams.root("val.txt"),"w")
    traindir, valdir = hyperparams.root("train_images"), hyperparams.root("val_images")
    subprocess.call(["mkdir", traindir])
    subprocess.call(["mkdir", valdir])
    for imgfile in os.listdir(hyperparams.root(orig_name)):
        out = os.path.basename(imgfile).split(".")[0]
        if random.random() < constants.prop_val:
            val.write(out + "\n")
            subprocess.call(["cp", os.path.join(hyperparams.root(orig_name), imgfile), os.path.join(valdir,imgfile)])
        else:
            train.write(out + "\n")
            subprocess.call(["cp", os.path.join(hyperparams.root(orig_name), imgfile), os.path.join(traindir,imgfile)])
    train_gt, val_gt = open(hyperparams.root("train.txt"),"w"), open(hyperparams.root("val.txt"),"w")
    traindir_gt, valdir_gt = hyperparams.root("train_images"), hyperparams.root("val_images")
    subprocess.call(["mkdir", traindir_gt])
    subprocess.call(["mkdir", valdir_gt])
    train.close()
    val.close()

def ytbb_tables(csv_name:str,):
    '''
    Establi
    '''
    gt = pd.read_csv( )
    
    pass


def add_gt_pascal(hyperparams):
    bboxs = readsql("SELECT * FROM candidate_bbox WHERE dataset = 'pascal'",hyperparams)
    d = '/data_b/aseewald/data/VOC2008/ground_truth'
    ds = os.listdir(d)
    conn = psycopg2.connect(**hyperparams.pg)
    cursor = conn.cursor()
    int2cat = {}
    int2cat[-1] = 'None'
    for k,v in constants.voc2008_labels.items():
        int2cat[v[1]] = k
    for xi,x in enumerate(ds):
        print("working on {},{},{}".format(x,xi,xi/len(ds)))
        imgbboxs = bboxs[bboxs['imgname'] == x.replace('.png','')]
        X = pascal_to_labelspace(imread(os.path.join(d,x)))
        for idx,series in imgbboxs.iterrows():
            cats = []
            # accidentally called 'miny' 'min'
            for i in range(series['min'],series['maxy']-1):
                for j in range(series['minx'],series['maxx']-1):
                    try: #boundary conditions and off-by-1 counter stuff. too lazy to figure it out right now
                        cats.append(int2cat[X[i,j]])
                    except:
                        continue
            count = list(filter(lambda x:x[0] != 'None',Counter(cats).most_common()))
            if len(count) == 0:
                maxcat,catprop = 'None',1
            else:
                maxcat,catprop = count[0]
                catprop = catprop / len(cats)
            print("imgname={},canid={},maxcat={},catprop={}".format(series['imgname'],series['canid'],maxcat,catprop))
            cursor.execute("INSERT INTO ground_truth VALUES ('{}',{},'{}','{}',{})".format(series['imgname'],series['canid'],maxcat,'pascal',catprop))
        conn.commit()

def pascal_dense(hyperparams):
    d = '/data_b/aseewald/data/VOC2008/ground_truth'
    ds = os.listdir(d)
    conn = psycopg2.connect(**hyperparams.pg)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS pascal_pixgt(imgname TEXT,y INT,x INT,category TEXT)")
    int2cat = {}
    int2cat[-1] = 'None'
    done = np.squeeze(pd.read_csv('/data/aseewald/donef.sql').values)
    for k,v in constants.voc2008_labels.items():
        int2cat[v[1]] = k
    for xi,x in enumerate(ds):
        if x in done:
            print("already did",x)
            continue
        print("working on {},{},{}".format(x,xi,xi/len(ds)))
        X = pascal_to_labelspace(imread(os.path.join(d,x)))
        batch = []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                batch.append([x,i,j,int2cat[X[i,j]],-1])
        cursor.executemany("INSERT INTO pascal_pixgt VALUES(%s,%s,%s,%s,%s)",batch)
        conn.commit()
    conn.close()

def pascal_to_labelspace(img):
    '''
    Takes a PASCAL 2008 ground truth image. In these, color indicates object class of a region.
    Produces a 2D array of same dimensions where object classes are associated with integer values.
    '''
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb
    out_img = constants.voc_neutral * np.ones((img.shape[0],img.shape[1]), dtype=np.int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if tuple(img[i][j]) not in voc_neutral:
                hexlabel = rgb_to_hex(tuple(img[i][j]))
                for (hexval, intval) in constants.voc2008_labels.values():
                    if hexval == hexlabel:
                        out_img[i][j] = intval
                        break
    return out_img

def unround(hyperparams):
    '''
    For speed's sake, the segmentation code I wrote downsamples the image, performs normalized cuts segmentation on the smaller version, and then upsamples the segment matrix to the original size of the image. The interpolation involved in resizing leads to non-integer segment labels at the boundaries. This scans through the segment matricies and .
    
    '''
    #for segmentdir in [hyperparams.root("val_superpixelation"),hyperparams.root("val_segmentation")]:
    for segmentdir in [hyperparams.root("val_superpixelation")]:
        for imgname in os.listdir(segmentdir):
            if imgname[-2:] != "gz":
                continue
            imgpath = segmentdir + "/" + imgname
            try:
                img = np.genfromtxt(imgpath, delimiter=",")
            except ValueError as e:
                print("error on " + imgname)
                continue
            # just extracting the number of segments within from the file name.
            num_segments = int(imgname.split("_")[3].split(".")[0])
            sanity_check = 100
            if np.unique(img).size < sanity_check:
                # for now, because previous things might have been messed up.
                if np.count_nonzero(img) == img.size:
                    print("removing " + imgname + " of {} unique".format(np.unique(img).size))
                    subprocess.call(['rm',imgpath])
                else:
                    print("continuing " + imgname + " of {} unique".format(np.unique(img).size))
                continue
            print("doing " + imgname + " of {} unique".format(np.unique(img).size))
            imgout = np.zeros_like(img, dtype="int")
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    # do this in the case of segments.
                    if (not float(img[i][j]).is_integer()):
                        guess = 0
                    else:
                        guess = img[i][j]
                    imgout[i][j] = guess
            print("at end: {} unique".format(np.unique(imgout).size))
            np.savetxt(imgpath, imgout, delimiter=",")

voc_neutral = {(0,0,0), (224, 224, 192)}

def pascalExtractPatches(hyperparams):
    '''
    '''
    all_imgnames = {imgName[:-1] for imgName in open(hyperparams.root("all.txt")).readlines()}
    val_imgnames = hyperparams.val_names()
    subprocess.call(["mkdir",hyperparams.root("val_patches")])
    subprocess.call(["mkdir",hyperparams.root("train_patches")])
    for imgName in all_imgnames:
        doc = ET.parse("{}/{}.xml".format(hyperparams.root("Annotations"),imgName)).getroot()
        img = imread("{}/{}.jpg".format(hyperparams.root("JPEGImages"),imgName))
        for objid, obj in enumerate(doc.findall('object')):
           bbox = obj.find('bndbox')
           objname = obj.find('name').text
           xmin = math.floor(float((bbox.find('xmin').text)))
           xmax = math.floor(float((bbox.find('xmax').text)))
           ymin = math.floor(float((bbox.find('ymin').text)))
           ymax = math.floor(float((bbox.find('ymax').text)))
           # these next two lines do not look correct, but they are correct. that is how pascal voc labels are named.
           raw_patch = img[ymin:ymax,xmin:xmax]
           if imgName in val_imgnames:
               subprocess.call(["mkdir",hyperparams.root("val_patches")])
               patchname = "{}/{}_{}_{}.png".format(hyperparams.root("val_patches"),imgName,objname,objid)
               for i,split in enumerate(hyperparams.possible_splits):
                   if objname in split:
                       # the reason for 'all_to_split' is different splits have different identifiers
                       # for the classes.
                       labelfile = open("{}/cnn/labels_val_".format(hyperparams.root()) + str(i),"a")
                       labelfile.write( )
               if objname in constants.voc2008_set6['known']:
                   lf6.write("{} {}\n".format(patchname, constants.all_to_split(constants.voc2008_set6,constants.voc2008_labels[objname][1])))
                   lf6.close()
           else:
               subprocess.call(["mkdir",hyperparams.root("train_patches")])
               patchname = "{}/{}_{}_{}.png".format(hyperparams.root("train_patches"),imgName,objname,objid)
               #imsave(patchname,raw_patch)
               #labelfile = open("{}/cnn/labels_train".format(hyperparams.root()), "a")
               #labelfile.write("{} {}\n".format(patchname, hyperparams.gt_labels[objname][1])) # the id of the object name.
               #labelfile.close()
               lf1, lf2, lf3, lf4, lf5, lf6 = open("{}/cnn/labels_train_1".format(hyperparams.root()), "a"), open("{}/cnn/labels_train_2".format(hyperparams.root()), "a"),open("{}/cnn/labels_train_3".format(hyperparams.root()), "a"),open("{}/cnn/labels_train_4".format(hyperparams.root()), "a"),open("{}/cnn/labels_train_5".format(hyperparams.root()), "a"),open("{}/cnn/labels_train_6".format(hyperparams.root()), "a"),
               if objname in constants.voc2008_set6['known']:
                   lf6.write("{} {}\n".format(patchname, constants.all_to_split(constants.voc2008_set6,constants.voc2008_labels[objname][1])))
                   lf6.close()

def climatlab(argstr):
    "Wrapper function for invoking matlab to do what we want."
    return subprocess.call(["matlab","-nojvm","-nodisplay","-nosplash","-r",argstr + ";quit"])


# FOR OBJECT CANDIDATES
def runSp(hyperparams,N_sp):
    try:
        subprocess.call(["mkdir",hyperparams.root("train_superpixelation")])
        subprocess.call(["mkdir",hyperparams.root("val_superpixelation")])
        climatlab("datadir='{}';destdir='{}';N_sp={};extract_superpixels".format(hyperparams.root(),N_sp, hyperparams.root("train_superpixelation")))
        climatlab("datadir='{}';destdir='{}';N_sp={};extract_superpixels".format(hyperparams.root(), hyperparams.root("val_superpixelation"), N_sp))
    except subprocess.CalledProcessError as e: 
        print("Couldn't run matlab")
        return False

def runSeg(N_seg):
    print("datadir='{}';destdir='{}';num_segs={};extract_segments".format(hyperparams.root(),hyperparams.root("train_segmentation"),N_seg))
    try:
        subprocess.call(["mkdir",hyperparams.root("train_segmentation")])
        subprocess.call(["mkdir",hyperparams.root("val_segmentation")])
        climatlab("datadir='{}';destdir='{}';num_segs={};extract_segments".format(hyperparams.root(),hyperparams.root("train_segmentation"),N_seg))
        climatlab("datadir='{}';destdir='{}';num_segs={};extract_segments".format(hyperparams.root(),hyperparams.root("val_segmentation"),N_seg))
    except subprocess.CalledProcessError as e: 
        print("Couldn't run matlab")
        return False

class ObjectCandidate:
    '''
    I have got to make Segment and RCNN inherit from this.
    '''
    def __init__(self,imgName,raw_img,chosen_pixels,keypoints,descriptors,codebook,hyperparams,score=None,bbox=None,validation=True):
        self.imgName = imgName
        self.centroid = round(np.mean(chosen_pixels[0])), round(np.mean(chosen_pixels[1]))
        self.object_distribution = {classifier : {splitid : np.zeros(len(hyperparams.possible_splits[splitid]['known'])) for splitid in range(len(hyperparams.possible_splits))} for classifier in constants.classifiers}
        if not bbox: #if there is not a bbox, there is a set of pixels (dense and wasteful).
            #tmpmin_y, tmpmax_y = int(math.floor(np.min(chosen_pixels[0]))), int(math.floor(np.max(chosen_pixels[0])))
            #tmpmin_x, tmpmax_x = int(math.floor(np.min(chosen_pixels[1]))), int(math.floor(np.max(chosen_pixels[1])))
            self.chosen_pixels = set(zip(chosen_pixels[0], chosen_pixels[1]))
            self.bbox_raw = [min_y, max_y, min_x, max_x]
            self.bbox,(expand_y,expand_x) = expand_candidate(self.bbox_raw,img.shape)
            min_y,max_y,min_x,max_x = self.bbox
        else:
            # If not working with segments, expand_candidate has already run.
            self.bbox = bbox
            min_y,max_y,min_x,max_x = bbox
        self.imgShape = raw_img.shape
        # I need to write a script to go through adding in the saliency image.
        self.sal_img = imread(hyperparams.root("val_saliency/{}.jpg".format(imgName)))[min_y:max_y,min_x:max_x]
        self.saliency_count = np.where(self.sal_img > 0)[0].size
        self.img_patch = raw_img[min_y:(max_y+1),min_x:(max_x+1)]
        if len(self.img_patch.shape) == 3:
            lab_patch = rgb2lab(self.img_patch)
        elif len(self.img_patch.shape) in [2,4]:
            lab_patch = np.zeros((self.img_patch.shape[0],self.img_patch.shape[1],3))
        else:
            sys.exit(1)
        self.color_histogram = np.concatenate( (np.histogram(lab_patch[:,:,0], bins=constants.color_bins)[0], \
                                                np.histogram(lab_patch[:,:,1], bins=constants.color_bins)[0], \
                                                np.histogram(lab_patch[:,:,2], bins=constants.color_bins)[0]))
        try:
            texton_patch = scipy.io.loadmat(hyperparams.root("val_textons/{}.jpg.mat".format(imgName)))['rotInvarFeats_textonMap'][min_y:max_y,min_x:max_x]
            self.texton_histogram = np.histogram(texton_patch, bins=constants.num_texton_clusters)[0]
        except:
            self.texton_histogram = np.zeros(constants.num_texton_clusters)
        if validation and hyperparams.maskBased: #e.g. VOC2008.
            corresponding_gt = imread(hyperparams.root("ground_truth") + "/{}.png".format(imgName))
            pixlabels = pascal_to_labelspace(corresponding_gt[min_x:max_x,min_y:max_y]).flatten()
            meaningful_pixlabels = np.delete(pixlabels,np.where(pixlabels == constants.voc_neutral))
            if meaningful_pixlabels.size == 0:
                self.label = None
            else:
                self.label = np.argmax(np.bincount(meaningful_pixlabels))
        elif validation: #e.g. COCO
            self.label = None #add it later.
        if hyperparams.objectness_method == "whatisanobject?":
            objectness = scipy.io.loadmat(hyperparams.root("objectness/{}.jpg.mat".format(imgName)))['results']
            intersections, scores = [], []
            for row in objectness: #assuming they come in sorted, nonincreasing order of objectness
                intersection = area({'ymin' : row[1], 'ymax' : row[3], 'xmin' : row[0], 'xmax' : row[2]},
                                    {'ymin' : min_y, 'ymax' : max_y, 'xmin' : min_x, 'xmax' : max_x})
                if intersection == 0:
                    continue
                segarea, objectness_area = (max_y - min_y) * (max_x - min_x), (row[2] - row[0]) * (row[3] -row[1])
                intersections.append(intersection / segarea)
                scores.append(row[4])
                if intersection == segarea:
                    break
            merged = np.array(intersections) * np.array(scores)
            if merged.size == 0:
                self.objectness = 0
            else:
                self.objectness = np.max(scores[np.argmax(merged)])
        elif hyperparams.objectness_method == "BING":
            self.objectness = score


class Segment_candidate(ObjectCandidate):
    '''
    '''
    def __init__(self,imgName,split,raw_img,segmented_img,segid,keypoints,descriptors,codebook):
        self.seg_amount = np.unique(segmented_img).size - 1
        chosen_pixels = np.where(segmented_img == segid)
        self.chosen_pixels = chosen_pixels
        assert(hyperparams.objectness_method == "whatisanobject?")
        ObjectCandidate.__init__(self,imgName,split,raw_img,chosen_pixels,keypoints,descriptors,codebook)
        self.edge_pixels = []
        for pix in self.chosen_pixels:
             if (pix[0] - 1, pix[1]) not in self.chosen_pixels or (pix[0] + 1, pix[1]) not in self.chosen_pixels or (pix[0], pix[1] - 1) not in self.chosen_pixels or (pix[0], pix[1] + 1) not in self.chosen_pixels:
                 self.edge_pixels.append(pix)
        self.coded = (codebook.n_clusters + 1) * np.ones_like(self.img_patch) #this avoids overlap with codebook predictions. to make for neutral space.
        relevant_descriptors = []
        for i, keypoint in enumerate(keypoints):
            rel_kpy, rel_kpx = math.floor(keypoint[0] - min_y), math.floor(keypoint[1] - min_x)# because keypoints are absolute positions to be adjusted.
            if segmented_img[tuple(keypoint)] == segid:
                relevant_descriptors.append(descriptors[i])
                self.coded[(rel_kpy, rel_kpx)] = codebook.predict(descriptors[i])
        codes = [codebook.predict(descr)[0] for descr in relevant_descriptors]
        self.code_histogram = np.histogram(codes,bins=constants.num_pyramid_clusters)[0]
    def count_overlap(self,pixels):
        return len(pixels.intersection(self.chosen_pixels))

class RCNN_candidate(ObjectCandidate):
    '''
    Score is probability of being an object, as determined by RCNN. This is a binary problem.
    Preliminary tests were not promising, so this has not been developed further.
    '''
    def __init__(self,imgName,split,raw_img,bbox,score,keypoints,descriptors,codebook):
        self.chosen_pixels,chosen_pixels = bbox,bbox
        ObjectCandidate.__init__(self,imgName,split,raw_img,chosen_pixels,keypoints,descriptors,codebook)
        
        pass

class BING_candidate(ObjectCandidate):
    '''
    Score is probability of being an object, as determined by opencv bing.
    '''
    def __init__(self,imgName,raw_img,bbox,bbox_raw,score,keypoints,descriptors,codebook,canid):
        self.chosen_pixels,chosen_pixels = bbox,bbox
        self.score = score
        ObjectCandidate.__init__(self,imgName,raw_img,bbox,keypoints,descriptors,codebook,bbox=bbox,score=score)
        self.coded = (codebook.n_clusters + 1) * np.ones_like(self.img_patch) #this avoids overlap with codebook predictions. to make for neutral space.
        relevant_descriptors = []
        min_y,max_y,min_x,max_x = bbox
        for i, keypoint in enumerate(keypoints):
            rel_kpy, rel_kpx = math.floor(keypoint[0] - min_y), math.floor(keypoint[1] - min_x)# because keypoints are absolute positions to be adjusted.
            if (bbox[0] <= keypoint[0] <= bbox[1]) and (bbox[2] <= keypoint[1] <= bbox[3]):
                relevant_descriptors.append(descriptors[i])
                self.coded[(rel_kpy, rel_kpx)] = codebook.predict(descriptors[i])
        codes = [codebook.predict(descr)[0] for descr in relevant_descriptors]
        self.code_histogram = np.histogram(codes,bins=constants.num_pyramid_clusters)[0]
        self.canid = canid
    def count_overlap(self,pixels):
        return len(list(filter(lambda pix: (self.bbox[0] < pix[0] < self.bbox[1]) and (self.bbox[2] < pix[1] < self.bbox[3]),pixels)))

class Superpixel:
    '''
    These are the fine-grained units of image data whose object probabilities are pooled according to
    the receptive field learning described in (paper) and for our implementation of Object Graph Descriptors.
    This data is not relevant for the receptive field learning process in (paper).
    '''
    def __init__(self,imgName,raw_img,superpixeled_img,spid,segments=None):
        self.imgName = imgName
        self.object_distribution = {}
        chosen_pixels = np.where(superpixeled_img == spid)
        self.centroid = round(np.mean(chosen_pixels[0])), round(np.mean(chosen_pixels[1]))
        min_y, max_y = math.floor(np.min(chosen_pixels[0])), math.floor(np.max(chosen_pixels[0]))
        min_x, max_x = math.floor(np.min(chosen_pixels[1])), math.floor(np.max(chosen_pixels[1]))
        self.bbox = [min_y, max_y, min_x, max_x]
        self.chosen_pixels = set(zip(chosen_pixels[0], chosen_pixels[1]))
        sal_img = imread(hyperparams.root("val_saliency/{}.jpg".format(imgName)))[min_x:max_x,min_y:max_y]
        self.saliency_count = np.where(sal_img > 0)[0].size
        self.edge_pixels = []
        for pix in self.chosen_pixels:
             if (pix[0] - 1, pix[1]) not in self.chosen_pixels or (pix[0] + 1, pix[1]) not in self.chosen_pixels or (pix[0], pix[1] - 1) not in self.chosen_pixels or (pix[0], pix[1] + 1) not in self.chosen_pixels:
                 self.edge_pixels.append(pix)
        self.img_patch = raw_img[min_y:max_y,min_x:max_x]
        lab_patch = rgb2lab(self.img_patch)
        self.color_histogram = np.concatenate( (np.histogram(lab_patch[:,:,0], bins=constants.color_bins)[0], \
                                                np.histogram(lab_patch[:,:,1], bins=constants.color_bins)[0], \
                                                np.histogram(lab_patch[:,:,2], bins=constants.color_bins)[0]))
        try:
            texton_patch = scipy.io.loadmat(hyperparams.root("val_textons/{}.jpg.mat".format(imgName)))['rotInvarFeats_textonMap'][min_x:max_x,min_y:max_y]
            self.texton_histogram = np.histogram(texton_patch, bins=constants.num_texton_clusters)[0]
        except:
            self.texton_histogram = None
        if segments != None:
            objectnesses = []
            for segment in segments:
                intersection = segment.count_overlap(self.chosen_pixels)
                objectnesses.append((intersection / len(self.chosen_pixels)) * segment.objectness)
            self.objectness = max(objectnesses)
        else:
            self.objectness = None
    def __lt__(self,other):
        "Not meaningful, but required for technical reasons."
        return True

def runObjectness(hyperparams,Nper):
    if hyperparams.objectness_method == "whatisanobject?":
        try:
            climatlab("indir='{}';outdir='{}';num_boxes={};extract_objectness".format(hyperparams.root('val_images'),hyperparams.root("objectness"),constants.num_objectness_boxes))
        except subprocess.CalledProcessError as e: 
            print("Couldn't run matlab")
            return False
    fin,fout = hyperparams.root("objectness/objness.csv"),hyperparams.root("objectness/objness_chosen.hdf")
    if (hyperparams.objectness_method == "BING") and (not os.path.exists(fout)):
        try:
            # This produces enough object proposals that they don't all fit into memory.
            if not os.path.exists(fin):
                subprocess.call(["./saliency"])
            N = len(hyperparams.val_names())
            data = pd.read_csv(hyperparams.root(fin),names=["imgname","minx","miny","maxx","maxy","objness"])
            data = data.sort('objness',ascending=True)
            chosen = data.head(N * Nper)
            chosen['imgname'] = chosen['imgname'].map(os.path.basename)
            chosen['imgname'] = chosen['imgname'].map(lambda x:x.split('.')[0])
            chosen.to_hdf(fout,'root')
        except:
            print("Couldn't run saliency_bing. Is it compiled? If not, see Makefile")
        
        

def create_tables():
    '''
    Here, canid and spid refers to the placement in the pickled candidate and superpixel lists.
    For candidate-related tables, 'type' means segment or objectness or RCNN.
    '''
    stmts = [
    '''CREATE TABLE IF NOT EXISTS entropy
        (imgname VARCHAR, splitid INT, canid INT, type TEXT, classifiertype TEXT, entropy FLOAT,
         PRIMARY KEY(imgname,splitid,canid,type,classifiertype));
    ''',
    '''CREATE TABLE IF NOT EXISTS easiness
        (imgname VARCHAR, splitid INT, canid INT, type TEXT, classifiertype TEXT, easiness FLOAT,
          PRIMARY KEY(imgname,splitid,canid,type,classifiertype));
    ''',
    '''CREATE TABLE IF NOT EXISTS candidate_object_distribution
        (imgname VARCHAR, splitid INT, canid INT, type TEXT, distr TEXT, dataset TEXT,
          PRIMARY KEY(imgname,splitid,canid,type,dataset));
    ''',
    '''CREATE TABLE IF NOT EXISTS candidate_centroid
        (imgname VARCHAR, canid INT, type TEXT, y INT, x INT, dataset TEXT,
          PRIMARY KEY(imgname,canid,type,dataset));
    ''',
    '''CREATE TABLE IF NOT EXISTS imgsize
        (imgname VARCHAR, height INT, width INT,
          PRIMARY KEY(imgname));
    ''',
    '''CREATE TABLE IF NOT EXISTS ground_truth
        (imgname VARCHAR, canid INT, classname TEXT);
    ''',
    '''CREATE TABLE IF NOT EXISTS candidate_bbox
        (imgname VARCHAR, canid INT, type TEXT, min INT, maxy INT, minx INT, maxx INT, dataset TEXT,
          PRIMARY KEY(imgname,canid,type,dataset));
    ''',
    # superpixels don't have 'types'
    '''CREATE TABLE IF NOT EXISTS sp_centroid
        (imgname VARCHAR, spid INT, y INT, x INT,
          PRIMARY KEY(imgname,spid));
    ''',
    '''CREATE TABLE IF NOT EXISTS sp_object_distribution
        (imgname VARCHAR, splitid TEXT, spid INT, classifiertype TEXT,distr TEXT,
          PRIMARY KEY(imgname,splitid,spid,classifiertype));
    ''',
    '''CREATE TABLE IF NOT EXISTS sp_bbox
        (imgname VARCHAR, spid INT, miny INT, maxy INT, minx INT, maxx INT,
          PRIMARY KEY(imgname,spid));
    '''
    ]
    for stmt in stmts:
        dosql(stmt,hyperparams,whichdb="postgres")

def area(a, b):
    dx = min(a['xmax'], b['xmax']) - max(a['xmin'], b['xmin'])
    dy = min(a['ymax'], b['ymax']) - max(a['ymin'], b['ymin'])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

# FOR APPEARANCE BASED FEATURES (baseline)
def runTexton(hyperparams):
    '''
    Textons are one of many possible appearance features to use for object class discovery. This is not essential to the main idea of this work.
    Note, this stores intermediate files of large size, so be sure to have a few hundred gigabytes to spare.
    '''
    argstr_train = "texton('{}','{}','{}','jpg',{},{})".format(hyperparams.root("train_images/"), hyperparams.root("train_textons/"),
                                                             hyperparams.root("train_texton_aux/"), constants.num_texton_responses,
                                                             constants.num_texton_clusters)
    argstr_val = "texton('{}','{}','{}','jpg',{},{})".format(hyperparams.root("val_images/"), hyperparams.root("val_textons/"),
                                                             hyperparams.root("val_texton_aux/"), constants.num_texton_responses,
                                                             constants.num_texton_clusters)
    try:
        climatlab(argstr_train)
        climatlab(argstr_val)
    except subprocess.CalledProcessError as e:
        print("Couldn't run matlab")
        return False

def form_codebook(hyperparams,descriptors):
    "The codebook is built based on the "
    codebook = KMeans(n_clusters=constants.num_pyramid_clusters,verbose=1)
    print("forming the codebook...")
    codebook.fit(descriptors)
    pickle.dump(codebook, open(hyperparams.root("codebook"),"wb"))
    return codebook

def mk_OG_vectors():
    '''
    For the old approach
    train_ogvectors/above_below
    train_og_left_right
    train_og_random
    train_og_gradient
    '''	
    # first, the hardcoded ideas, which all share
    for imgName in train_imgnames:
        for (pool_fn, result_dir) in [(objectGraph.above_below, "{}/train_ogvectors/above_below"), (objectGraph.left_right, "{}/train_ogvectors/left_right"),
                                      (objectGraph.randomly, "{}/train_ogvectors/randomly"),
                                      (objectGraph.split_at_gradient, "{}/train_ogvectors/split_at_gradient"),
                                      (objectGraph.ogvector, " ")]:
            og = objectGraph.object_graph(segment,superpixels,pool_fn)
            pickle.dump(result_dir + "/" + imgName, og)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action')
    candistr_parser = subparsers.add_parser('candistr')
    candistr_parser.add_argument('nickname')
    candistr_parser.add_argument('trial',type=int)
    parser.add_argument('tstep',type=int)
    args = parser.parse_args()
    if args.action == 'candistr':
        hyperparams = hp.greedy_hp[args.nickname]
        modeldir,_ = pixelwise_modeldir(hyperparams,hyperparams.pixnickname,args.trial,hyperparams.splitid,hyperparams.pixhp.numfuse,hyperparams.experimentName)
        add_can_pred(hyperparams,args.nickname,hyperparams.splitid,modeldir,args.tstep,numfuse=hyperparams.pixhp.numfuse) 
