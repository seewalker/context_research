import copy
import argparse
import tensorflow as tf
import numpy as np
import scipy.stats
import sys
import cv2
from typing import Dict,Tuple,List
import arch_draw
import arch_block
import arch_patch
import hyperparams as hp
from utils import *
from arch_visualize import draw_rangealt
from mytypes import *

MARGIN_VAL = 1.0 # this has been confirmed to work well.
#HACK_CONTRASTIVE = 1.4 # minimum value found to work so far.
HACK_CONTRASTIVE = 1.0 # minimum value found to work so far.
sigmanet_layers = 'conv'

def arch_args() -> Tuple[argparse.Namespace,hp.ArchHyperParams]:
    parser = argparse.ArgumentParser()
    parser.add_argument('nickname',help="key for naming the output produced, both in databases and in files.",type=str)
    parser.add_argument('splitid',type=int)
    # arguments related to distributed tensorflow.
    parser.add_argument('--rtdir',default="../data/runtime_stats")
    parser.add_argument('--cachedir',default="../data/cache")
    parser.add_argument('--model_root',default="../data/modeldir")
    parser.add_argument('--log_root',default="../data/logdir")
    parser.add_argument('--max_numobj',default=sys.maxsize)
    parser.add_argument('--min_numobj',default=0)
    parser.add_argument('--task_index',type=int)
    parser.add_argument('--job_name',type=str)
    parser.add_argument('--root_port',default=2800,type=int)
    # 
    parser.add_argument('--gpus',help="comma-separated list of GPUs",type=str)
    parser.add_argument('--split_type',default="random",help="split_type = (random | illustration | clusterbased)",type=str)
    parser.add_argument('--batchsize',default=32,type=int)
    parser.add_argument('--from_scratch',action='store_true',default=False)
    parser.add_argument('--cpu_only',action='store_true',default=False)
    parser.add_argument('--trial',default=0,type=int)
    parser.add_argument('--train_dataset',default="COCO",help="Dataset model was trained on. For now, always coco.",type=str)
    parser.add_argument('--transfer_dataset',default="COCO",help="Dataset model is evaluated with.",type=str)
    parser.add_argument('--decode_dim',default=32,help="Dimensionality of decoder vector",type=int)
    parser.add_argument('--simdim',default=128,help="Dimensionality of similarity feature vector",type=int)
    parser.add_argument('--distinct_decode',default=False,help="use distinct variables for the decoder module, or share weights with main CNN?",type=bool)
    # make subparsers based on action
    subparsers = parser.add_subparsers(dest='action',help=" ")
    train_parser = subparsers.add_parser('train',help=" ")
    test_parser = subparsers.add_parser('test',help=" ")
    store_parser = subparsers.add_parser('store',help="Store representations we've trained at a variety of timesteps")
    storeat_parser = subparsers.add_parser('store_at',help="Store representations we've trained at a specified timestep")
    baseline_parser = subparsers.add_parser('baselines',help="Store representations of object candidates due to out-of-the-box CNN features.")
    go_parser = subparsers.add_parser('go',help="Meta-subcommand for running everything. This will take awhile.")
    ctx_parser = subparsers.add_parser('context')
    pr_parser = subparsers.add_parser('pr')
    prplot_parser = subparsers.add_parser('prplot')
    complete_parser = subparsers.add_parser('complete_vis')
    # train subparser.
    train_parser.add_argument('--sigmanet_parallelism',default=1,help="number of distinct sigmanets producing distinct MxM parameters, whose results are used in parallel across depth dimension.",type=int)
    train_parser.add_argument('--summary',default=True,help="run tensorflow summary ops?")
    train_parser.add_argument('--num_candidates',default=1000,help="Used when action=store",type=int)
    train_parser.add_argument('--num_epochs',default=5,type=int)
    train_parser.add_argument('--zoom_t',default="out",help="in | out. out means artificially center by padding with zeros, capturing the full image.",type=str)
    # store parsers.
    baseline_parser.add_argument('--num_candidates',default=3000,help="Used when action=store",type=int)
    baseline_parser.add_argument('--canmax',default=3000,type=int)
    store_parser.add_argument('--zoom_t',default="out",help="in | out. out means artificially center by padding with zeros, capturing the full image.",type=str)
    store_parser.add_argument('--spacing',default=3,type=int)
    store_parser.add_argument('--canmax',default=3000,type=int)
    storeat_parser.add_argument('--spacing',default=3,type=int)
    storeat_parser.add_argument('--canmax',default=3000,type=int)
    storeat_parser.add_argument('--tstep',default="max")
    # evaluation parsers
    prplot_parser.add_argument('--nicknames',type=str)
    ctx_parser.add_argument('--ctx_k',default=2,type=int)
    pr_parser.add_argument('--timestep',type=int)
    pr_parser.add_argument('--nsamples',type=int,default=5000)
    pr_parser.add_argument('--class_partition',type=str,default="all")
    ctx_parser.add_argument('--timestep',type=int)
    ctx_parser.add_argument('--niter',default=100,type=int)
    prplot_parser.add_argument('--timestep',type=str)
    test_parser.add_argument('--zoom_t',default="out",help="in | out. out means artificially center by padding with zeros, capturing the full image.",type=str)
    test_parser.add_argument('--num_test',default=400,help=" ")
    args = parser.parse_args()
    hyperparams = hp.arch_hp[args.nickname]
    return args,hyperparams

def signature_saliency(img:np.ndarray) -> np.ndarray:
    """
    This is borrowed from salienpy commit 6658f060536addd641fe69e6a4530bcb4746b17a.

    Signature Saliency.
    X. Hou, J. Harel, and C. Koch, "Image Signature: Highlighting Sparse Salient
    Regions." IEEE Trans. Pattern Anal. Mach. Intell. 34(1): 194-201 (2012)
    """
    def img_padded_for_dct(img):
        h = img.shape[0]
        w = img.shape[1]
        if (h%2 == 1):
            h=h+1
        if (w%2 == 1):
            w=w+1
        return cv2.copyMakeBorder(img, top=0,  bottom=(h-img.shape[0]),
                                       left=0, right=(w-img.shape[1]),
                                       borderType=cv2.BORDER_REPLICATE)
    old_shape = (img.shape[0],img.shape[1])
    img = img_padded_for_dct(img)
    img = img/255.0
    sal = []
    for c in range(img.shape[2]):
        channel = img[:,:,c].astype(np.dtype('float32'))
        channel_dct = np.sign(cv2.dct(channel))
        s = cv2.idct(channel_dct)
        s = np.multiply(s,s)
        sal.append(s)
    sal = sum(sal)/3.0
    sal = cv2.GaussianBlur(sal, (11,11), 0)
    sal = sal[:old_shape[0], :old_shape[1]]
    return sal

def vanilla(X:tf.Tensor,weights:weight_t,biases:weight_t,dropout:tf.Tensor, \
            stop_fc7=False,output_keys=['similarity'],leaky=False) -> tf.Tensor:
    return(arch_tail(weights,biases,arch_head(weights,biases,X,dropout),dropout,stop_fc7=stop_fc7,output_keys=output_keys,leaky=leaky))
 
def arch_head(weights:weight_t,biases:weight_t,img:tf.Tensor,dropout:tf.Tensor,\
             keys=None,apply_pos='conv0',keep_resolution=False) -> tf.Tensor:
    '''
    This is the portion of the network before the context representation layer.
    There is programmed to be some amount of variation of where the context representation layer is,
    so this has conditionals at the end.
    It is modeled after VGGNET.

    img - (?,img_s,img_s,3)
    dropout - a placeholder.
    keys - If provided, operations in arch_head get the provided prefix prepended to their name.

    The interruption can happen anywhere in the conv layers.
    '''
    assert(apply_pos in ['conv0','conv1','conv2','conv3','conv4','conv5'])
    if apply_pos == 'conv0':
        return img
    headnames = ['conv1_1','conv1_2','conv2_1','conv2_2']
    if not keys:
        keys = {k : k for k in headnames}
    conv1_1 = conv2d(keys['conv1_1'], img, weights[keys['conv1_1']], biases[keys['conv1_1']])
    conv1_2 = conv2d(keys['conv1_2'], conv1_1, weights[keys['conv1_2']], biases[keys['conv1_2']])
    if not keep_resolution:
        pool1 = max_pool('pool1', conv1_2, k=2)
    else:
        pool1 = conv1_2
    norm1 = lrn('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, dropout)
    if apply_pos == 'conv1':
        return norm1
    conv2_1 = conv2d(keys['conv2_1'], norm1, weights[keys['conv2_1']], biases[keys['conv2_1']])
    conv2_2 = conv2d(keys['conv2_2'], conv2_1, weights[keys['conv2_2']], biases[keys['conv2_2']])
    if not keep_resolution:
        pool2 = max_pool('pool2', conv2_2, k=2)
    else:
        pool2 = conv2_2
    norm2 = lrn('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, dropout)
    if apply_pos == 'conv2':
        return norm2
    conv3_1 = conv2d('conv3_1', headout, weights['conv3_1'], biases['conv3_1'])
    conv3_2 = conv2d('conv3_2', conv3_1, weights['conv3_2'], biases['conv3_2'])
    conv3_3 = conv2d('conv3_3', conv3_2, weights['conv3_3'], biases['conv3_3'])
    if not keep_resolution:
        pool3 = max_pool('pool3', conv3_3, k=2)
    else:
        pool3 = conv3_3
    norm3 = lrn('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, dropout)
    if apply_pos == 'conv3':
        return norm3
    conv4_1 = conv2d('conv4_1', norm3, weights['conv4_1'], biases['conv4_1'])
    conv4_2 = conv2d('conv4_2', conv4_1, weights['conv4_2'], biases['conv4_2'])
    conv4_3 = conv2d('conv4_3', conv4_2, weights['conv4_3'], biases['conv4_3'])
    if not keep_resolution:
        pool4 = max_pool('pool4', conv4_3, k=2)
    else:
        pool4 = conv4_3
    norm4 = lrn('norm4', pool4, lsize=4)
    norm4 = tf.nn.dropout(norm4, dropout)
    if apply_pos == 'conv4':
        return norm4
    conv5_1 = conv2d('conv5_1', norm4, weights['conv5_1'], biases['conv5_1'])
    conv5_2 = conv2d('conv5_2', conv5_1, weights['conv5_2'], biases['conv5_2'])
    conv5_3 = conv2d('conv5_3', conv5_2, weights['conv5_3'], biases['conv5_3'])
    if not keep_resolution:
        pool5 = max_pool('pool5', conv5_3, k=2)
    else:
        pool5 = conv5_3
    norm5 = lrn('norm5', pool5, lsize=5)
    norm5 = tf.nn.dropout(norm5, dropout)
    if apply_pos == 'conv5':
        return norm5

def arch_tail(weights:weight_t,biases:weight_t,headout:tf.Tensor,dropout:tf.Tensor,\
              stop_fc7=False,output_keys=['similarity'],leaky=False,apply_pos='conv0',keep_resolution=False) \
              -> tf.Tensor:
    '''
    If using ctx_draw, we have to deal with the patch already being downsampled, so conditionally avoid max pooling.

    Note, the conv2d function has RELU and biases included in it, so I'm not missing anything.

    This anticipates a set of possible starting points and it also anticipates 
    '''
    assert(apply_pos in ['conv0','conv1','conv2','conv3','conv4','conv5'])
    def cnn_k(start,expected_resolution=7):
        outputs = {}
        outputs['conv5'] = start
        # actually, should determine flatnum from weights shape.
        flatnum = weights['fc6'].get_shape()[0].value
        if leaky:
            fc6 = leaky_relu('fc6-leak',tf.matmul(tf.reshape(start,[-1,flatnum],name="fc_reshape"), weights['fc6'])+biases['fc6'])
            fc7 = leaky_relu('fc7-leak',tf.matmul(fc6,weights['fc7']) + biases['fc7'])
        else:
            fc6 = tf.nn.relu(tf.matmul(tf.reshape(start,[-1,flatnum],name="fc_reshape"), weights['fc6'])+biases['fc6'])
            fc7 = tf.nn.relu(tf.matmul(fc6,weights['fc7']) + biases['fc7'])
        if stop_fc7:
            return fc7
        for output_key in output_keys: #for separate outs at the end.
            if leaky:
                outputs[output_key] = leaky_relu(output_key + '-leak',tf.matmul(fc7,weights[output_key]) + biases[output_key])
            else:
                outputs[output_key] = tf.nn.relu(tf.matmul(fc7,weights[output_key]) + biases[output_key],name=output_key + '-leak')
        return outputs
    def from4(start,expected_resolution=7):
        b,h,w,c = [x.value for x in start.get_shape()]
        downsample_todo = h//expected_resolution
        # can't handle more downsampling above this much.
        assert(downsample_todo in [2,4,8,16])
        stride5_1 = 2 if downsample_todo == 16 else 1
        stride5_2 = 2 if downsample_todo >= 8 else 1
        stride5_3 = 2 if downsample_todo >= 4 else 1
        conv5_1 = conv2d('conv5_1', start, weights['conv5_1'], biases['conv5_1'],strides=[1,stride5_1,stride5_1,1])
        conv5_2 = conv2d('conv5_2', conv5_1, weights['conv5_2'], biases['conv5_2'],strides=[1,stride5_2,stride5_2,1])
        conv5_3 = conv2d('conv5_3', conv5_2, weights['conv5_3'], biases['conv5_3'],strides=[1,stride5_3,stride5_3,1])
        # actually, I should have strides in conv5 according to how much I need to reduce size.
        pool5 = max_pool('pool5', conv5_3, k=2)
        norm5 = lrn('norm5', pool5, lsize=5)
        norm5 = tf.nn.dropout(norm5, dropout)
        return cnn_k(norm5)
    def from3(start,expected_resolution=7):
        conv4_1 = conv2d('conv4_1', start, weights['conv4_1'], biases['conv4_1'])
        conv4_2 = conv2d('conv4_2', conv4_1, weights['conv4_2'], biases['conv4_2'])
        conv4_3 = conv2d('conv4_3', conv4_2, weights['conv4_3'], biases['conv4_3'])
        s4 = [x.value for x in conv4_3.get_shape()]
        if (s4[1] > 14) and (s4[2] > 14):
            pool4 = max_pool('pool4', conv4_3, k=2)
        else:
            pool4 = tf.identity(conv4_3)
        norm4 = lrn('norm4', pool4, lsize=4)
        norm4 = tf.nn.dropout(norm4, dropout)
        return from4(norm4)
    def from2(start,expected_resolution=7):
        conv3_1 = conv2d('conv3_1', start, weights['conv3_1'], biases['conv3_1'])
        conv3_2 = conv2d('conv3_2', conv3_1, weights['conv3_2'], biases['conv3_2'])
        conv3_3 = conv2d('conv3_3', conv3_2, weights['conv3_3'], biases['conv3_3'])
        s3 = [x.value for x in conv3_3.get_shape()]
        if (s3[1] > 14) and (s3[2] > 14):
            pool3 = max_pool('pool3', conv3_3, k=2)
        else:
            pool3 = tf.identity(conv3_3)
        norm3 = lrn('norm3', pool3, lsize=4)
        norm3 = tf.nn.dropout(norm3, dropout)
        return from3(norm3)
    def from1(start,expected_resolution=7):
        conv2_1 = conv2d('conv2_1', start, weights['conv2_1'], biases['conv2_1'])
        conv2_2 = conv2d('conv2_2', conv2_1, weights['conv2_2'], biases['conv2_2'])
        pool2 = max_pool('pool2', conv2_2, k=2)
        norm2 = lrn('norm2', pool2, lsize=4)
        norm2 = tf.nn.dropout(norm2, dropout)
        return from2(norm2)
    def from0(start,expected_resolution=7):
        conv1_1 = conv2d('conv1_1', start, weights['conv1_1'], biases['conv1_1'])
        conv1_2 = conv2d('conv1_2', conv1_1, weights['conv1_2'], biases['conv1_2'])
        pool1 = max_pool('pool1', conv1_2, k=2)
        norm1 = lrn('norm1', pool1, lsize=4)
        norm1 = tf.nn.dropout(norm1, dropout)
        return from1(norm1)
    if apply_pos == 'conv0':
        return from0(headout)
    elif apply_pos == 'conv1':
        return from1(headout)
    elif apply_pos == 'conv2':
        return from2(headout)
    elif apply_pos == 'conv3':
        return from3(headout)
    elif apply_pos == 'conv4':
        return from4(headout)
    else:
        return cnn_k(headout)

def contrastive_loss(norm:tf.Tensor,y:tf.Tensor,debug_items=None) -> tf.Tensor:
    '''
    high norm (distance) and y=1 (meaning labeled similar) => high error from first term, low error from second term.
    high norm (distance) and y=0 (meaning labeled dissimilar) => low error on both terms.
    low norm (distance) and y=1 (meaning labeled similar) => low error on both terms.
    low norm (distance) and y=0 (meaning labeled dissimilar) => hihg error from second term, low error on first term.
    '''
    if tf.__version__[0] == '0':
        unpack,sub,mul,batch_matmul = tf.unpack,tf.sub,tf.mul,tf.batch_matmul
    else:
        unpack,sub,mul,batch_matmul = tf.unstack,tf.subtract,tf.multiply,tf.matmul
    margin = tf.constant(MARGIN_VAL,dtype=tf.float32)
    distances = mul(y,norm) + HACK_CONTRASTIVE * mul(1 - y,tf.maximum(margin - norm,0))
    return tf.reduce_mean(distances)
        
def lossfn(loss_t:str,norm:tf.Tensor,y:tf.Tensor,y_a:tf.Tensor,y_b:tf.Tensor,
           wholeout_a:tf.Tensor,wholeout_b:tf.Tensor,batchsize:int,class_scale=0.4,debug_items=None) -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor]:
    '''
    norm - distance in embedding space.
    y - siamese ground truth. 1 for equal, 0 for not equal.
    y_{a,b} - ground truth of individual class labels.
    wholeout_{a,b} - something produced by the network related to individual class labels by example. This is not used when using just contrastive loss.

    y_a and y_b are stacked such that "batchsize" is the outer-most dimension.

    
    '''
    if tf.__version__[0] == '0':
        pack,unpack,sub,mul,batch_matmul = tf.pack,tf.unpack,tf.sub,tf.mul,tf.batch_matmul
    else:
        pack,unpack,sub,mul,batch_matmul = tf.stack,tf.unstack,tf.subtract,tf.multiply,tf.matmul
    if loss_t != 'contrastive':
        fstdim, wdim = norm.get_shape()[0].value, wholeout_a.get_shape()[0].value
        assert(fstdim % batchsize == 0 == wdim % batchsize)
        filts_used = wdim // batchsize
    else:
        fstdim = norm.get_shape()[0].value
        assert(fstdim % batchsize == 0)
        filts_used = fstdim // batchsize
    num_classes = y_a.get_shape()[1].value
    if filts_used > 1: #duplicate ground truth when using classification with multiple patches.
        y_a = tf.reshape(tf.transpose(pack(filts_used * [y_a]),perm=[1,0,2]),[batchsize * filts_used,-1])
        y_b = tf.reshape(tf.transpose(pack(filts_used * [y_b]),perm=[1,0,2]),[batchsize * filts_used,-1])
    # 
    if loss_t == "euclidean":
        y = 1 - y # Need to reverse this for euclidean to make sense.
        distances = tf.square(sub(tf.sqrt(norm), y))
        loss = tf.sqrt(tf.reduce_sum(distances))
        classify_term = similarity_term = None
    elif loss_t == "contrastive":
        loss = contrastive_loss(norm,y,debug_items=debug_items)
        classify_term = None
        similarity_term = loss
    elif loss_t == "dual":
        # This assumes y_a is output of whole object candidate so not really suitable for DRAW.
        assert(y_a != None and y_b != None and wholeout_a != None and wholeout_b != None), "Using loss_t=dual requires individual class labels"
        similarity_term = contrastive_loss(norm,y)
        classify_term = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=wholeout_a,labels=y_a)) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=wholeout_b,labels=y_b))
        loss = similarity_term + class_scale * classify_term
    return loss,similarity_term,classify_term

def pydraw(imgs,vs):
    '''
    Used to test the idea of DRAW. Keeping it around...
    '''
    batchsize = len(imgs)
    N,A = 56,224
    identical_cols = np.tile(np.arange(A),N).reshape((N,A)).astype(np.float32)
    range_const = np.arange(N) - N/2 - 0.5
    for i,img in enumerate(imgs):
        g_X,g_Y,sigmasq,stride,intensity = draw_rangealt(vs[i])
        print(f"g_X={g_X},g_Y={g_Y},sigmasq={sigmasq},stride={stride},intensity={intensity}")
        mu_X = g_X + range_const * stride
        mu_Y = g_Y + range_const * stride
        expscale = (-1/2) * 1/sigmasq
        F_X = np.exp(expscale * np.square(identical_cols - np.expand_dims(mu_X,1)))
        F_X = F_X / np.expand_dims(np.sum(F_X,axis=1),1)
        F_Y = np.exp(expscale * np.square(identical_cols - np.expand_dims(mu_Y,1)))
        F_Y = F_Y / np.expand_dims(np.sum(F_Y,axis=1),1)
        imgout = np.zeros((N,N,3))
        for chan in range(3):
            imgout[:,:,chan] = np.dot(np.dot(F_X,img[:,:,chan]),F_Y.T)
        plt.imshow(imgout)
        plt.show()
    
def pyloss(o1,o2,y1,y2,y1_class,y2_class,loss_t,class_scale=0.25):
    loss,losstups = 0,{}
    if loss_t in ["contrastive","dual"]:
        losstups['contrastive'] = []
        if len(o1['similarity'].shape) == 3:
            o1['similarity'] = np.transpose(o1['similarity'],axes=[1,2,0])
            o2['similarity'] = np.transpose(o2['similarity'],axes=[1,2,0])
            o1['similarity'] = o1['similarity'].reshape((o1['similarity'].shape[0] * o1['similarity'].shape[1],o1['similarity'].shape[2]))
            o2['similarity'] = o2['similarity'].reshape((o2['similarity'].shape[0] * o2['similarity'].shape[1],o2['similarity'].shape[2]))
        d = np.apply_along_axis(np.linalg.norm,0,o1['similarity']-o2['similarity'])
        lossvec = (y1 == y2).astype(np.int) * d + (y1 != y2).astype(np.int) * np.maximum(MARGIN_VAL - d,np.zeros_like(d))
        loss += np.mean(lossvec)
        for i,lossval in enumerate(lossvec):
            losstups['contrastive'].append((y1[i],y2[i],lossval))
    elif loss_t in ["softmax","dual"]:
        losstups['classify'] = []
        numberators1 = np.exp(o1['classify'][y1_class.astype(np.bool)])
        numberators2 = np.exp(o2['classify'][y2_class.astype(np.bool)])
        denom1 = np.sum(np.exp(o1['classify']),axis=1)
        denom2 = np.sum(np.exp(o2['classify']),axis=1)
        lossvec1 = -1 * np.log(numberators1 / denom1)
        lossvec2 = -1 * np.log(numberators2 / denom2)
        loss += class_scale * (np.mean(lossvec1) + np.mean(lossvec2))
        for i,lossval in enumerate(lossvec1):
            losstups['classify'].append((y1[i],o1['classify']))
            losstups['classify'].append((y2[i],o2['classify']))
    return loss,losstups

# todo - get rid of "y_alt" which is confusing.
def netops(parameters:parameters_t,X:tf.Tensor,Xp:tf.Tensor,Xfull:tf.Tensor,Xpfull:tf.Tensor,Xcfull:tf.Tensor,Xcpfull:tf.Tensor,\
           bbox_loc:tf.Tensor,bboxp_loc:tf.Tensor,y:tf.Tensor,ya:tf.Tensor,yb:tf.Tensor,dropout:tf.Tensor,batchsize:int,\
           hyperparams,args,keep_p=False,devices=['gpu:0'],debug_items=None) \
            -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor]:
    '''
    Makes an optimizer on the siamese architecture.
    y : whether they are equal.
    '''
    try:
        if len(devices) == 1 or args.distributed:
            left_devs = right_devs = devices
        else: #if not distributed, use different gpus for different parts of the architecture.
            left_devs,right_devs = devices[0:len(devices)//2],devices[len(devices)//2:]
    except EnvironmentError:
        sys.exit(1)
    if tf.__version__[0] == '0':
        unpack,sub,mul,batch_matmul = tf.unpack,tf.sub,tf.mul,tf.batch_matmul
    else:
        unpack,sub,mul,batch_matmul = tf.unstack,tf.subtract,tf.multiply,tf.matmul
    if hyperparams.ctxop == 'DRAW3D':
        out1,filts,post,attentionvals,headout_full,attention_boxs = arch_draw_video(X,Xfull,Xcfull,bbox_loc,parameters,dropout,batchsize,hyperparams.loss_t,hyperparams.baseline_t,hyperparams.numfilts,hyperparams.include_center,hyperparams.stop_grad)
        out2,filtsp,postp,attentionvalsp,headout_fullp,attention_boxsp = arch_draw_video(Xp,Xpfull,Xcpfull,bboxp_loc,parameters,dropout,batchsize,hyperparams.loss_t,hyperparams.baseline_t,hyperparams.numfilts,hyperparams.include_center,hyperparams.stop_grad)
        attention = (attentionvals,attention_boxes)
        attentionp = (attentionvalsp,attention_boxesp)
        diff = sub(out1['similarity'],out2['similarity'])
        if hyperparams.distance_t == 'euclidean':
            norm = tf.reduce_sum(tf.square(diff),[1,2])
        elif hyperparams.distance_t == 'metric':
            diffp = tf.reshape(diff,[batchsize,-1])
            # have to express this (x-mu) M (x-mu) weirdly because of lack of nice dot semantics.
            norm = tf.reduce_sum(diffp * tf.matmul(diffp,parameters[0]['metric']),1)
    elif hyperparams.ctxop == "DRAW":
        arch_drawfn = arch_draw.draw_switch(hyperparams)
        out1,filts,post,attentionvals,headout_full,attention_boxes = arch_drawfn(X,Xfull,Xcfull,bbox_loc,parameters,dropout,batchsize,hyperparams.relative_arch,hyperparams.distinct_decode,hyperparams.loss_t,hyperparams.baseline_t,hyperparams.numfilts,hyperparams.include_center,hyperparams.decode_arch,hyperparams.stop_grad,debug_items=debug_items,devices=left_devs,keep_resolution=hyperparams.keep_resolution,stop_pos=hyperparams.decode_stop_pos)
        out2,filtsp,postp,attentionvalsp,headout_fullp,attention_boxesp = arch_drawfn(Xp,Xpfull,Xcpfull,bboxp_loc,parameters,dropout,batchsize,hyperparams.relative_arch,hyperparams.distinct_decode,hyperparams.loss_t,hyperparams.baseline_t,hyperparams.numfilts,hyperparams.include_center,hyperparams.decode_arch,hyperparams.stop_grad,debug_items=debug_items,devices=right_devs,keep_resolution=hyperparams.keep_resolution,stop_pos=hyperparams.decode_stop_pos)
        attention = (attentionvals,attention_boxes)
        attentionp = (attentionvalsp,attention_boxesp)
        diff = sub(out1['similarity'],out2['similarity'])
        if hyperparams.distance_t == 'euclidean':
            norm = tf.reduce_sum(tf.square(diff),[1,2])
        elif hyperparams.distance_t == 'metric':
            diffp = tf.reshape(diff,[batchsize,-1])
            # have to express this (x-mu) M (x-mu) weirdly because of lack of nice dot semantics.
            norm = tf.reduce_sum(diffp * tf.matmul(diffp,parameters[0]['metric']),1)
        pre,prep = None,None #for now, no easy way to get this.
    elif hyperparams.ctxop in ["patches","above_below","expand"]:
        out1,filts,post,attention,headout_full = arch_patch.arch_patch(X,Xfull,Xcfull,parameters,dropout,batchsize,hyperparams.loss_t,hyperparams.ctxop,debug_items=debug_items,devices=left_devs)
        out2,filtsp,postp,attentionp,headout_fullp = arch_patch.arch_patch(Xp,Xpfull,Xcpfull,parameters,dropout,batchsize,hyperparams.loss_t,hyperparams.ctxop,devices=right_devs)
        sh = len(out1['similarity'].get_shape())
        if sh == 3:
            norm = tf.reduce_sum(tf.square(sub(out1['similarity'],out2['similarity'])),[1,2])
        elif sh == 2:
            norm = tf.reduce_sum(tf.square(sub(out1['similarity'],out2['similarity'])),1)
        pre,filts = None,None
    else:
        extra = []
        headout_full = None #not needed in this case.
        attention1,out1,pre,post,filts = arch_block.arch_block(X,Xcfull,parameters,dropout,batchsize,hyperparams.ctxop,hyperparams.numfilts,hyperparams.M,debug_items=debug_items,devices=left_devs)
        attention2,out2,prep,postp,filtsp = arch_block.arch_block(Xp,Xcpfull,parameters,dropout,batchsize,hyperparams.ctxop,hyperparams.numfilts,hyperparams.M,debug_items=debug_items,devices=right_devs)
        attention = (attention1,attention2)
        wholeout1,wholeout2 = out1,out2
        # norm is squared euclidean distance.
        norm = tf.reduce_sum(tf.square(sub(out1['similarity'],out2['similarity'])),[1,2])
    if hyperparams.loss_t == 'dual':
        numfilts = hyperparams.numfilts
        # -1 means inferring the right number of columns to use.
        if hyperparams.include_center:
            oc1,oc2 = tf.reshape(out1['classify'], [batchsize * numfilts,-1]),tf.reshape(out2['classify'],[batchsize * numfilts,-1])
        else: #apply a minus one because not including the center.
            oc1,oc2 = tf.reshape(out1['classify'], [batchsize * (numfilts - 1),-1]),tf.reshape(out2['classify'],[batchsize * (numfilts - 1),-1])
    else:
        oc1,oc2 = None,None
    # y_alt is y reshaped according to number of filters.
    loss,loss_contrastive,loss_classify = lossfn(hyperparams.loss_t,norm,y,ya,yb,oc1,oc2,batchsize,debug_items=debug_items)
    optimizer = tf.train.AdamOptimizer(learning_rate=hyperparams.lr).minimize(loss)
    gvs = None
    # compare distance to 
    euclid = tf.square(sub(tf.sqrt(norm), (1 - y)))
    if keep_p:
        attention = (attention,attentionp)
        pre = (pre,prep)
        post = (post,postp)
        headout_full = (headout_full,headout_fullp)
        filts = (filts,filtsp)
    return((loss,loss_contrastive,loss_classify),euclid,optimizer,attention,pre,headout_full,post,out1,out2,filts,gvs)

def inner(M:int,key:str,weights,xavier,trainable,flatnum_ctx):
    ''' 
       Helper function for initialization with the sigmanet flavor.
    '''
    if sigmanet_layers == 1:
        if hyperparams.baseline_t == "full":
            weights[key] = [1e-5 * np.random.randn(flatnum_ctx,M * M).T, 1e-5 * np.random.randn(M * M).T]
            xavier[key] = True
            trainable[key] = True
        elif hyperparams.baseline_t == "biasonly":
            weights[key] = [np.zeros((flatnum_ctx,M * M)).T, 1e-5 * np.random.randn(M * M).T]
            xavier[key] = False
            trainable[key] = (False,True)
    if sigmanet_layers == 2:
        mnum = constants.hidden_scale * M * M
        if hyperparams.baseline_t == "full":
            weights[key] = [1e-5 * np.random.randn(flatnum_ctx,mnum).T, 1e-5 * np.random.randn(mnum).T]
            weights[f'hidden{key}'] = [1e-5 * np.random.randn(mnum,M * M).T, 1e-5 * np.random.randn(M * M).T]
            xavier[f'hidden{key}'],xavier[key] = True,True
            trainable[key],trainable[f'hidden{key}'] = True,True
        elif hyperparams.baseline_t == "biasonly":
            weights[key] = [np.zeros((flatnum_ctx,mnum)).T, 1e-5 * np.random.randn(mnum).T]
            weights[f'hidden{key}'] = [np.zeros((mnum,M * M)).T, 1e-5 * np.random.randn(M * M).T]
            xavier[f'hidden{key}'],xavier[key] = (False,True),(False,True)
            trainable[key],trainable[f'hidden{key}'] = (False,True),(False,True)
    return weights,xavier,trainable

def initialize(M:int,args,hyperparams,nclasses:int, \
               initialization=None,combine_method="subtraction",only_fromnpy=False,splitid=None,attention_debug=True,img_s=224) \
               -> parameters_t :
    '''
    This is some spaggetti code that correctly initializes the appropriate tensors for a given set of parameters from params.py.
    random normal values are provided whether or not xavier initialization is ultimately used.
    '''
    if hyperparams.isvanilla: assert(hyperparams.ctxop is None)
    extra_weights,extra_biases = OrderedDict({}),OrderedDict({})
    if initialization == None: initialization = hyperparams.initialization
    if initialization in ["random"]:
        weights = vggrand(0.01)
        xavier = {k : True for k in weights.keys()}
        trainable = {k : False for k in weights.keys()}
        weights['similarity'] = [0.0001 * np.random.randn(4096,args.simdim), 0.001 * np.random.randn(args.simdim)]
        xavier['similarity'],trainable['similarity'] = True,False
        return(totensors(weights,True,xavier=xavier))
    if initialization in ['vanilla.rand','vanilla.vgg']: #the split-based
        # if splitid is not none, use a VGGnet finetuned on the class split specified.
        if splitid == None: name = hyperparams.root(f'cnn/npymodels/{initialization}.npy')
        else: name = hyperparams.root(f'cnn/npymodels/{initialization}_{splitid}.npy')
    elif initialization in ['embed','vggnet']: name = hyperparams.root(f'cnn/npymodels/{initialization}.npy')
    weights = np.load(name,encoding='bytes').item()
    print(f"DONE LOADING {name}")
    if attention_debug: #seeing if making other things not trainable changes how DRAW gradients work.
        trainable = {k : False for k in weights.keys()}
    else:
        trainable = {k : True for k in weights.keys()}
    if initialization == "embed":
        del(weights['loc_out']); del(weights['fcreduce'])
    xavier = {k : False for k in weights.keys()}
    if initialization == "vggnet" and not only_fromnpy: #testing for only_fromnpy in case we want to just use npy info.
        del(weights['fc8'])
    if only_fromnpy:
        return(totensors(weights,True))
    # this happens after deleting fc8, so that the same set of keys is in all dicts.
    if hyperparams.distance_t == 'euclidean':
        print("Loss function involving norm with euclidean distance, no need for parameters here.")
    elif hyperparams.distance_t == 'metric':
        print("Loss function involving norm with learned metric distance, so some parameters go here.")
        featdim = hyperparams.numfilts * args.simdim
        weights['metric'] = [np.eye(featdim) + 0.01 * np.random.randn(featdim,featdim),np.zeros(featdim)]
        xavier['metric'],trainable['metric'] = False,True
    if hyperparams.ctxop == 'DRAW' and args.distinct_decode:#we need a decoder with distinct, and would like a decoder with initially these trained weights.
        for k in list(weights.keys()):
            kp = 'decode-' + k
            xavier[kp],trainable[kp] = False,True
            weights[kp] = copy.deepcopy(weights[k])
    nfmap = {1 : 64, 2 : 128, 3 : 256} #number of featuremaps after each of these layers.
    if hyperparams.ctxop == 'DRAW':
        assert(img_s % (2 **hyperparams.decode_stop_pos) == 0)
        if hyperparams.keep_resolution:
            # assuming pool all at once, but keeping a bit more resolution than usual case (but the cropping is still in the 224x224 space).
            fh=2**(hyperparams.decode_stop_pos-1)
        else:
            fh=2**(hyperparams.decode_stop_pos)
            # determining based on featuremaps that have undergone 4 pools.
        weights['decode'] = [0.01 * np.random.randn(img_s//fh * img_s//fh * nfmap[hyperparams.decode_stop_pos],args.decode_dim), np.random.randn(args.decode_dim)]
        xavier['decode'],trainable['decode'] = True,True
    if 'fc8' in xavier: del(xavier['fc8'])
    if 'fc8' in trainable: del(trainable['fc8'])
    if hyperparams.loss_t == 'dual':
        weights['classify'] = [0.0001 * np.random.randn(4096,nclasses), 0.001 * np.random.randn(nclasses)]
        xavier['classify'],trainable['classify'] = True, True
    weights['similarity'] = [0.0001 * np.random.randn(4096,args.simdim), 0.001 * np.random.randn(args.simdim)]
    xavier['similarity'],trainable['similarity'] = True,True
    if hyperparams.isvanilla or hyperparams.ctxop in ["patches","above-below"]:
        return(totensors(weights,trainable,extra_weights=extra_weights,extra_biases=extra_biases,xavier=xavier))
    if hyperparams.ctxop in ["block_intensity","block_blur"]:
        if sigmanet_layers == 'conv':
            weights['snet_0'] = [0.01 * np.random.randn(3,3,3,64), 0.01 * np.random.randn(64)]
            weights['snet_1'] = [0.01 * np.random.randn(3,3,64,64), 0.01 * np.random.randn(64) ]
            weights['snet_2'] = [0.05 * np.random.randn(3,3,64,128), 0.05 * np.random.randn(128) ]
            for i in range(hyperparams.numfilts):
                kk = f'snet_out_{i}'
                weights[kk] = [0.01 * np.random.randn(3,3,128,1),None]
                xavier[kk],trainable[kk] = True, True
            xavier['snet_0'],xavier['snet_1'],xavier['snet_2'] = True,True,True
            trainable['snet_0'],trainable['snet_1'],trainable['snet_2'] = True,True,True
        elif sigmanet_layers == 'highlevel':
            weights['snet'] = [0.001 * np.random.randn(4096,M * M), 0.01 * np.random.randn(M * M) ]
            xavier['snet'],trainable['snet'] = True,True
        elif sigmanet_layers in ['mlp_1','mlp_2']:
            weights,xavier,trainable = inner(M,'sigma',weights,xavier,trainable,np.product([img_s,img_s,3]))
        else:
            assert(False), f"unknown sigmanet_layers:{sigmanet_layers}"
    elif hyperparams.ctxop == 'DRAW':
        #ctx_locs = [-0.2,0.2,-0.45,0.45] #don't have guesses gravitate towards middle. Try to look at surrounding patches.
        ctx_locs = [0] #don't have guesses gravitate towards middle. Try to look at surrounding patches.
        for i in range(hyperparams.numfilts-1): #minus one to leave space for candidate patch itself.
            key = f'attention_{i}'
            gx_guess = scipy.stats.norm.rvs(random.choice(ctx_locs),0.3)
            gy_guess = scipy.stats.norm.rvs(random.choice(ctx_locs),0.3)
            stride_guess = scipy.stats.norm.rvs(-0.6,0.25)
            intensity_guess = scipy.stats.norm.rvs(0,0.10) # zero-mean like in draw paper.
            sigmasq_guess = scipy.stats.norm.rvs(0,0.10)   # zero-mean like in draw paper.
            alt = draw_rangealt({'gp_X' : gx_guess,'gp_Y' : gy_guess,'sigmasqp' : sigmasq_guess,'stridep' : stride_guess,'intensityp' : intensity_guess},report_relative=True)
            print("Starting with filtid={}, relative gx={},relative gy={},sigmasq={},stride={},intensity={}".format(i,*alt))
            # plus 4 so it can take into account the bbox location, if concatenating with bbox (which is unnecessary if using relative arch).
            concatfeatsize = 0 if hyperparams.relative_arch else 4
            # two times the decode_dim because it is a concatenation of decoder on full image and bbox img.
            weights[key] = [1e-4 * np.random.randn(5,2 * args.decode_dim + concatfeatsize), [gx_guess,gy_guess,sigmasq_guess,stride_guess,intensity_guess]]
            xavier[key] = False
            if hyperparams.baseline_t == 'full': trainable[key] = (True,True)
            elif hyperparams.baseline_t == 'biasonly': trainable[key] = (False,True)
            elif hyperparams.baseline_t == 'fixed_pos': trainable[key] = (False,False)
            elif hyperparams.baseline_t == 'fixed-biasonly': trainable[key] = (False,False)
        if hyperparams.baseline_t == 'attentiononly': # this is a last minute check, if not attentiononly, trainability works like above.
            for k in weights.keys():
                is_attention = ("snet" in k) or ("attention" in k)
                trainable[k] = (is_attention, is_attention)
    return(totensors(weights,trainable,extra_weights=extra_weights,extra_biases=extra_biases,xavier=xavier))

def netops_vanilla(hyperparams,parameters,X,Xp,y,ya,yb,dropout,debug=False):
    '''
    Makes an optimizer on the siamese architecture.
    '''
    weights,biases = parameters
    batchsize = X.get_shape()[0].value
    out1 = vanilla(X,weights,biases,dropout)
    out2 = vanilla(X,weights,biases,dropout)
    norm = tf.reduce_sum(tf.square(tf.sub(out1['similarity'],out2['similarity'])),1)
    if hyperparams.loss_t == 'contrastive':
        out1c,out2c = None,None
    else:
        out1c,out2c = out1['classify'],out2['classify']
    loss,junk,_ = lossfn(hyperparams.loss_t,norm,y,ya,yb,out1c,out2c,batchsize)
    partial_accuracy = tf.square(tf.sub(tf.sqrt(norm), y))
    optimizer = tf.train.AdamOptimizer(learning_rate=hyperparams.lr,beta1=0.99,beta2=0.99).minimize(loss)
    return(loss,partial_accuracy,optimizer,out1,out2)
