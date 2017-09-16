'''
Alex Seewald 2016
aseewald@indiana.edu
'''
import pickle
import os
import random
import gzip
import pandas as pd
import getpass
import math
import socket
import sqlite3
import numpy as np
from typing import Optional
from scipy.stats import entropy
from enum import Enum
import constants
import datasets
from mytypes import *

def chisquared(x,y):
    if type(x) == list or type(y) == list:
        x, y = np.array(x), np.array(y)
    return 0.5 * np.sum(np.nan_to_num(np.divide(np.square(x - y),(x+y))))

def strs_of_vars(variables,local_env):
    '''
    Takes list of variables and produces string representations of them.
    This requires searching through the environment because python does not have an easy inverse to "eval".
    Pass in locals() as local_env.
    '''
    strs = []
    for var in variables:
        for k, v in list(local_env.items()):
            if v is var:
                strs.append(k)
    return strs

class HyperParams:
    '''
    Iterations of this project share these in common.
    If there are multiple candidate methods attempted with a shared experimentName, they end up in the same database.
    The 'type' attribute of the tables distinguishes them.
    '''
    def __init__(self,datadir,experimentName,candidate_method,objectness_method):
        self.datadir = datadir
        self.experimentName = experimentName
        if experimentName == "VOC2008":
            self.maskBased = True
            self.possible_splits = datasets.voc2008
        elif experimentName in ["COCO"]:
            self.maskBased = False
            self.possible_splits = datasets.coco
        elif experimentName == 'ytbb_classify':
            self.possible_splits = datasets.ytbb
        elif experimentName == 'ytbb_detect':
            self.possible_splits = datasets.ytbb
        self.candidate_method = candidate_method
        self.objectness_method = objectness_method
        self.anaconda2 = "/usr/local/anaconda2/bin/python"
        if "sqlite" in constants.dbtypes:
            self.read_db = '/data/aseewald/ctx_archu.db'
            if socket.gethostname() == 'madthunder': #the 'main' host. Others can mount related directories with sshfs.
                self.db = '/data/aseewald/ctx_archu.db'
                self.islocal = True
            else:
                self.db = self.root("ctx_arch_{}.db".format(socket.gethostname()))
                self.islocal = False
        if "postgres" in constants.dbtypes:
            # This dict will be expanded to keyword arguments in python functions making connections.
            # The settings that I use, but these can be modified without breaking these experiments.
            self.pg = {'dbname' : 'ctx2', 'user' : getpass.getuser(), 'host' : 'localhost', 'password' : 'boblawlaw'}
    def encode(self):
        '''
        This standardizes how result files will have their names indicate what experimental parameters they used.
        '''
        pass
    def root(self,arg=None):
        if arg:
            return(os.path.join(self.datadir, self.experimentName, arg))
        else:
            return(os.path.join(self.datadir, self.experimentName))
    # Some datasets like VOC2008 have ground truth available for only some of the data, so these functions return names
    # of data with and without it.
    def train_names(self):
        "Precondition: mksplit ran."
        if self.experimentName == "VOC2008":
            return [imgName[:-1] for imgName in open(self.root("train.txt")).readlines()]
        else:
            return [imgName.split(".")[0] for imgName in os.listdir(self.root("train_images"))]
    def val_names(self):
        "Precondition: mksplit ran."
        if self.experimentName == "VOC2008":
            return [imgName[:-1] for imgName in open(self.root("val.txt")).readlines()]
        else:
            return [imgName.split(".")[0] for imgName in os.listdir(self.root("val_images"))]
    def train_names_gt(self):
        "Precondition: mksplit ran."
        if self.experimentName == "VOC2008":
            return [imgName[:-1] for imgName in open(self.root("train_gt.txt")).readlines()]
        else:
            return [imgName.split(".")[0] for imgName in os.listdir(self.root("train_images"))]
    def val_names_gt(self,splitid=1,thresh=False):
        '''
        Precondition: mksplit ran.
        if 'tresh', another precondition is that the segments exist.
        '''
        if self.experimentName == "VOC2008":
            return [imgName[:-1] for imgName in open(self.root("val_gt.txt")).readlines()]
        else:
            return [imgName.split(".")[0] for imgName in os.listdir(self.root("val_images"))]

        

class PixelwiseHyperParams(HyperParams):
    '''
    
    '''
    def __init__(self,lr:float,opt_t:str,numfuse:int,stopgrad:bool,budget:float,resolution_t:str,pool_prop:Optional[str],pool_iter_prop:Optional[str],sparsity_model:Optional[str],schedule_t:Optional[str],active_dim:Optional[List[int]],active_network:Optional[str],ensemble_num=Optional[int],datadir='/data/aseewald',experimentName='COCO',candidate_method='objectness',objectness_method='BING'):
        self.uncertainty_types = ['uncertainty']
        if opt_t != 'classify':
            assert(active_dim is not None)
            # alternating N means do N iterations of traing main network followed by N of active network. convergence N means N iterations of main network followed by N iterations of active network repeatedly until (it is better than before OR it has stopped improving.)
            assert(schedule_t[0] in ['alternating','convergence'])
            # emoc is expected model output change. layered and regressor are type of the uncertainty prediction.
            assert(active_network in self.uncertainty_types + ['emoc' + 'reward-based'])
            assert(sparsity_model[0] in ['always','afterwards','never'])
            assert(0 <= pool_prop <= 1.0 and pool_prop is not None)
            assert(0 <= pool_iter_prop <= 1.0 and pool_iter_prop is not None)
        else:
            assert(active_dim is None)
            assert(active_network is None)
            assert(schedule_t is None)
            assert(pool_prop is None and pool_iter_prop is None)
        assert (resolution_t in ['pixel','detection'])
        self.lr,self.opt_t,self.stopgrad = lr,opt_t,stopgrad
        self.active_dim,self.active_network = active_dim,active_network 
        self.ensemble_num,self.numfuse = ensemble_num,numfuse
        self.resolution_t,self.schedule_t = resolution_t,schedule_t
        self.sparsity_model,self.budget = sparsity_model,budget
        self.pool_prop,self.pool_iter_prop = pool_prop,pool_iter_prop
        self.ACTIVE_TIMEOUT = 100
        HyperParams.__init__(self,datadir,experimentName,candidate_method,objectness_method)

pixelwise_mappings = {
    'long3' : 'pix',
    'long7' : 'pix'
}
pixelwise_hp = {
    'pix' : PixelwiseHyperParams(5e-5,'classify',2,False,1.0,'pixel',None,None,None,None,None,None,None),
    'uncert-joint-upsample' : PixelwiseHyperParams(5e-5,'joint',2,False,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',None),
    'uncert-joint-upsample-stopgrad' : PixelwiseHyperParams(5e-5,'joint',2,True,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',None),
    'highsparse' : PixelwiseHyperParams(5e-5,'joint',2,True,0.08,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',None),
    'lowsparse' : PixelwiseHyperParams(5e-5,'joint',2,True,0.333,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',None),
    'uncert-joint-upsample-stopgrad-pow' : PixelwiseHyperParams(5e-5,'joint',2,True,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',10,100),[40],'uncertainty',None),
    'uncert-uncertainty-upsample' : PixelwiseHyperParams(5e-5,'uncertainty',2,False,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',None),
    'uncert-joint-ensemble' : PixelwiseHyperParams(5e-5,'joint-ensemble',2,False,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',4),
    'uncert-uncertainty-ensemble': PixelwiseHyperParams(5e-5,'uncertainty-ensemble',2,False,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',4),
    'uncert-joint-upsample' : PixelwiseHyperParams(5e-5,'joint',2,False,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40,20],'uncertainty',None),
    'uncert-joint-upsample-convergence' : PixelwiseHyperParams(5e-5,'joint',2,False,0.1,'pixel',0.5,0.5,['afterwards'],('convergence',3,1),[40,20],'uncertainty',None),
    'uncert-uncertainty-upsample' : PixelwiseHyperParams(5e-5,'uncertainty',2,False,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40,20],'uncertainty',None),
    'uncert-joint-ensemble' : PixelwiseHyperParams(5e-5,'joint-ensemble',2,False,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40,20],'uncertainty',4),
    'uncert-uncertainty-ensemble': PixelwiseHyperParams(5e-5,'uncertainty-ensemble',2,False,0.1,'pixel',0.5,0.5,['afterwards'],('alternating',1,1),[40,20],'uncertainty',4),
    'uncert-joint-upsample-detection' : PixelwiseHyperParams(5e-5,'joint',2,False,0.1,'detection',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',None),
    'uncert-joint-upsample-stopgrad-detection' : PixelwiseHyperParams(5e-5,'joint',2,True,0.1,'detection',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',None),
    'uncert-uncertainty-upsample-detection' : PixelwiseHyperParams(5e-5,'uncertainty',2,False,0.1,'detection',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',None),
    'uncert-joint-ensemble-detection' : PixelwiseHyperParams(5e-5,'joint-ensemble',2,False,0.1,'detection',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',4),
    'uncert-uncertainty-ensemble-detection': PixelwiseHyperParams(5e-5,'uncertainty-ensemble',2,False,0.1,'detection',0.5,0.5,['afterwards'],('alternating',1,1),[40],'uncertainty',4),
    'uncert-joint-upsample-detection' : PixelwiseHyperParams(5e-5,'joint',2,False,0.1,'detection',0.5,0.5,['afterwards'],('alternating',1,1),[40,20],'uncertainty',None),
    'uncert-uncertainty-upsample-detection' : PixelwiseHyperParams(5e-5,'uncertainty',2,False,0.1,'detection',0.5,0.5,['afterwards'],('alternating',1,1),[40,20],'uncertainty',None),
    'uncert-joint-ensemble-detection' : PixelwiseHyperParams(5e-5,'joint-ensemble',2,False,0.1,'detection',0.5,0.5,['afterwards'],('alternating',1,1),[40,20],'uncertainty',4),
    'uncert-uncertainty-ensemble-detection': PixelwiseHyperParams(5e-5,'uncertainty-ensemble',2,False,0.1,'detection',0.5,0.5,['afterwards'],('alternating',1,1),[40,20],'uncertainty',4)
}

class GreedyHyperParams(HyperParams):
    '''
    og_num_scales specifies multi-scale behavoir of an older idea not considered in current implementations.
    og_k
    objgraph_distancefn is the default distance function used to define similarity of object graphs.

    nicknames are uniquely associated with a nickname of a pixelwise classifier, pixnickname.
    '''
    def __init__(self,pixnickname:str,splitid:int,reg_lambda:float,include_entropy:bool,entropy_const:float,field_t:str,prop_grid:Optional[float],dataset:str,datasrc_nickname:str,datadir="/data/aseewald",experimentName="COCO",splits=datasets.coco,candidate_method="objectness",objectness_method="BING", hs=2, og_k=4, og_num_scales=3, objgraph_distancefn=chisquared):
        assert(field_t in ["split","grid","mix"])
        if field_t == "mix":
            assert(prop_grid is not None)
        else:
            assert(prop_grid is None)
        HyperParams.__init__(self,datadir,experimentName,candidate_method,objectness_method)
        self.db,self.dataset,self.datasrc_nickname = '/fast-data/aseewald/ctx_arch.db',dataset,datasrc_nickname
        self.hs, self.og_k  = hs, og_k
        # explicitly saying int because somehow it gets to be floating point otherwise...
        self.pixnickname,self.splitid,self.prop_grid = pixnickname,int(splitid),prop_grid
        self.field_t,self.reg_lambda,self.include_entropy,self.entropy_const = field_t,reg_lambda,include_entropy,entropy_const
        self.pixhp = pixelwise_hp[pixelwise_mappings[pixnickname]] # to get numfuse and such.
     
class ArchHyperParams(HyperParams):
    '''
    See the command line argument help in arch.py for explanations of these variables.
    '''
    def __init__(self,loss_t:str,task:str,ctxop:Optional[str], M:Optional[int],include_center:Optional[bool],compute_t:Optional[str],distance_t:Optional[str],keep_resolution:Optional[bool],numfilts:Optional[int],conv_w:Optional[int],baseline_t:Optional[str],stop_grad:Optional[bool],decode_stop_pos:int,decoder_arch='vggnet',datadir="/data/aseewald",experimentName="COCO",candidate_method="objectness",objectness_method="BING",initialization="vggnet",isvanilla=False,negprop=0.5,lr=3e-5,img_s=224,relative_arch=True,distinct_decode=False,decode_arch='vggnet',dropout=0.65):
        #known baseline types.
        assert(ctxop in ['DRAW','DRAW3D','block_blur','block_intensity','above_below','expand','patches',None])
        assert(initialization in ['vggnet','embed','random'])
        assert(experimentName in ['COCO','VOC2008','ytbb_classify','ytbb_detect'])
        if ctxop == 'DRAW':
            assert(stop_grad is not None)
        else:
            assert(stop_grad is None)
        if isvanilla:
            assert(numfilts == None or numfilts == 1) #so shapes line up.
        if ctxop == "DRAW":
            basetypes = ["full","biasonly","fixed_pos","patches","above-below","fixed-biasonly","attentiononly"]
            assert(M is None)
            assert(baseline_t in basetypes)
            assert(compute_t in ['full','shared'])
            assert(decoder_arch in ['vggnet'])
        elif ctxop == "DRAW3D":
            assert(decoder_arch in ['c3d'])
        elif ctxop in ["block_blur",'block_intensity']:
            assert(M is not None)
            assert(baseline_t is None)
            assert(compute_t is None)
        elif ctxop in ['above_below','patches']:
            assert(M is None)
            assert(compute_t is None)
        elif ctxop is None:
            assert(isvanilla)
        assert(loss_t in ['dual','contrastive'])
        self.dataset_t = 'image' if experimentName in ['VOC2008','COCO'] else 'video'
        self.M,self.numfilts,self.stop_grad = M,numfilts,stop_grad
        self.conv_w,self.initialization,self.isvanilla = conv_w,initialization,isvanilla
        self.negprop,self.lr,self.loss_t,self.include_center = negprop,lr,loss_t,include_center
        self.task,self.ctxop,self.baseline_t  = task,ctxop,baseline_t
        self.compute_t,self.distance_t = compute_t,distance_t
        self.keep_resolution,self.img_s = keep_resolution,img_s
        self.relative_arch,self.distinct_decode,self.decode_arch = relative_arch,distinct_decode,decode_arch
        self.dropout,self.decode_stop_pos = dropout,decode_stop_pos
        HyperParams.__init__(self,datadir,experimentName,candidate_method,objectness_method)

usual_lr = 4e-5
arch_hp = {
    'DRAW4-contrastive' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='full',distance_t='euclidean',stop_grad=False),
    'DRAW4-dual' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='full',distance_t='euclidean',stop_grad=False),
    'DRAW4-dual-shared-metric' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='metric',stop_grad=False),
    'DRAW4-dual-shared-keep' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=True,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='metric',stop_grad=False),
    'DRAW3-dual-shared-keep' : ArchHyperParams(M=None,initialization="vggnet",numfilts=3,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=True,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='metric',stop_grad=False),
    'DRAW4-dual-shared-keep-stopgrad' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=True,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='metric',stop_grad=True),
    'DRAW3-dual-shared-keep-stopgrad' : ArchHyperParams(M=None,initialization="vggnet",numfilts=3,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=True,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='metric',stop_grad=True),
    'DRAW4-noctx' : ArchHyperParams(M=None,initialization="vggnet",numfilts=1,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=False),
    'DRAW4-dual-shared' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=False),
    'DRAW4-dual-shared-early' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=1,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=False),
    'DRAW4-dual-shared-stopgrad' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=True),
    'DRAW5-dual-shared-stopgrad' : ArchHyperParams(M=None,initialization="vggnet",numfilts=5,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=True),
    'DRAW3D-4-dual-shared' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',experimentName='ytbb_detect',stop_grad=False),
    'DRAW4-contrastive-shared' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=False),
    'DRAW5-dual-shared' : ArchHyperParams(M=None,initialization="vggnet",numfilts=5,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=False),
    'DRAW4-dual-pascal' : ArchHyperParams(experimentName="VOC2008",
                                    M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',stop_grad=False,\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='full',distance_t='euclidean'),
    'DRAW3-dual' : ArchHyperParams(M=None,initialization="vggnet",numfilts=3,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='full',distance_t='euclidean',stop_grad=False),
    'DRAW3-dual-shared' : ArchHyperParams(M=None,initialization="vggnet",numfilts=3,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=False),
    'DRAW4-biasonly' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='biasonly',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='full',distance_t='euclidean',stop_grad=False),
    'DRAW4-dual-shared-biasonly' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='biasonly',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=False),
    'DRAW3-biasonly' : ArchHyperParams(M=None,initialization="vggnet",numfilts=3,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='biasonly',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='full',distance_t='euclidean',stop_grad=False),
    'DRAW4-nocenter' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=False,compute_t='full',distance_t='euclidean',stop_grad=False),
    'DRAW5-shared-nocenter' : ArchHyperParams(M=None,initialization="vggnet",numfilts=5,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=False,compute_t='shared',distance_t='euclidean',stop_grad=False),
    'DRAW4-fixed' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='fixed_pos',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='full',distance_t='euclidean',stop_grad=False),
    'DRAW4-fixed-shared' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='fixed_pos',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=False),
    # this is the correct "fixed" idea because constant dynamic (and non-sensicle non-trained,distance_t='euclidean') things are not happening.
    'DRAW4-shared-fixedbiasonly' : ArchHyperParams(M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='fixed-biasonly',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean',stop_grad=False),
    'DRAW3-fixedbias' : ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",\
                                    M=None,initialization="vggnet",numfilts=3,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='fixed-biasonly',stop_grad=False,\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='full',distance_t='euclidean'),
    'DRAW4-attentiononly' : ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",\
                                    M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='attentiononly',stop_grad=False,\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='full',distance_t='euclidean'),
    'DRAW4-shared-attentiononly' : ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",\
                                    M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="dual",baseline_t='attentiononly',stop_grad=False,\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='DRAW',include_center=True,compute_t='shared',distance_t='euclidean'),
    'conv-block-intensity' : ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",\
                                    M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t=None,\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='block_intensity',include_center=False,compute_t=None,distance_t='euclidean',stop_grad=None),
    'conv-block-blur' : ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",stop_grad=None,\
                                    M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t=None,keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='block_blur',include_center=False,compute_t=None,distance_t='euclidean'),
    'expanded' : ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",\
                                    M=None,initialization="vggnet",numfilts=2,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='above-below',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='expand',include_center=True,compute_t='full',distance_t='euclidean',stop_grad=None),
    'above-below' : ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",\
                                    M=None,initialization="vggnet",numfilts=2,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='above-below',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='above_below',include_center=True,compute_t=None,distance_t='euclidean',stop_grad=None),
    'rpatches4' : ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",\
                                    M=None,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='patches',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop='patches',include_center=False,compute_t=None,distance_t='euclidean',stop_grad=None),
    # vanilla architectures without context representation
    'vanilla-embed' : ArchHyperParams(datadir="/data/aseewald/",experimentName="VOC2008",candidate_method= "objectness",objectness_method= "BING",\
                                    M=None,initialization="embed",numfilts=1,conv_w=5,isvanilla=True,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop=None,include_center=False,compute_t=None,distance_t='euclidean',stop_grad=None),
    'vanilla-vgg' : ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",\
                                    M=None,initialization="vggnet",numfilts=1,conv_w=5,isvanilla=True,negprop=0.5,lr=1e-6,loss_t="contrastive",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop=None,include_center=False,compute_t=None,distance_t='euclidean',stop_grad=None),
    'vanilla-rand' : ArchHyperParams(datadir="/data/aseewald/",experimentName="VOC2008",candidate_method= "objectness",objectness_method= "BING",\
                                    M=None,initialization="random",numfilts=1,conv_w=5,isvanilla=True,negprop=0.5,lr=usual_lr,loss_t="contrastive",baseline_t='full',\
                                    keep_resolution=False,decode_stop_pos=3,task='discovery',ctxop=None,include_center=False,compute_t=None,distance_t='euclidean',stop_grad=None)
}


greedy_hp = {
    'feb9small' : GreedyHyperParams('long3',3,0.001,False,None,'split',None,'COCO','feb9small'),
    'feb9' : GreedyHyperParams('long7',7,0.001,False,None,'split',None,'COCO','feb9'),
    'feb9small-entropy' : GreedyHyperParams('long3',3,0.001,True,0.1,'split',None,'COCO','feb9small'),
    'feb9-entropy' : GreedyHyperParams('long7',7,0.001,True,0.1,'split',None,'COCO','feb9'),
    'feb9-bigentropy' : GreedyHyperParams('long7',7,0.001,True,0.25,'split',None,'COCO','feb9'),
    'feb9small-grid-entropy' : GreedyHyperParams('long3',3,0.001,True,0.1,'grid',None,'COCO','feb9small'),
    'feb9-grid-entropy' : GreedyHyperParams('long7',7,0.001,True,0.1,'grid',None,'COCO','feb9'),
    'feb9small-mix' : GreedyHyperParams('long3',3,0.001,False,None,'mix',0.5,'COCO','feb9small'),
    'feb9-mix' : GreedyHyperParams('long7',7,0.001,False,None,'mix',0.5,'COCO','feb9')
}

# examples to get a hyperparams object when the other attributes don't really matter.
minimal_dset = {
    'COCO' : arch_hp['DRAW4-dual-shared'],
    'VOC2008' : arch_hp['DRAW4-dual-pascal'],
    'ytbb' : arch_hp['DRAW3D-4-dual-shared']
}

#hyperparams = GreedyHyperParams("/data/aseewald/","VOC2008", datasets.coco,"objectness", "BING",hs=2, og_k=4)
#hyperparams = ArchHyperParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="embed",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='full', task='discovery',ctxop='DRAW')

mainhost = "madthunder"
