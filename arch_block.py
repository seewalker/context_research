import math
import numpy as np
import tensorflow as tf
import arch_common
from scipy.stats import pearsonr
import constants
from utils import *
from mytypes import *

PRECISION = tf.float32
sigmanet_layers = 'conv'
def toblocks(arr:tf.Tensor,M:int,blk_s:int) -> List[tf.Tensor]:
    '''
    arr - BHWC data
    returns: a list of equally sized blocks along the spatial dimensions.
    '''
    batchlist = tf.unpack(arr)
    batchsize = len(batchlist)
    img_blocks,blks = [],[]
    for batchpos in range(batchsize):
        img = batchlist[batchpos]
        shape = img.get_shape()
        #channels = tf.unpack(tf.reshape(img,[shape[2].value,shape[0].value,shape[1].value,1]))
        channels = tf.unpack(tf.transpose(tf.expand_dims(img,0),perm=[3,1,2,0]))
        for channel in channels:
            for blk_i in range(M):
                for blk_j in range(M):
                    block = tf.image.crop_to_bounding_box(channel,blk_i * blk_s,blk_j * blk_s,blk_s,blk_s)
                    blks.append(block)
        img_blocks.append(tf.transpose(tf.pack(blks),perm=[3,1,2,0]))
        blks = []
    return img_blocks

def fromblocks(blocks,M):
    '''
    Inverse of toblocks.
    '''
    hbands = []
    for j in range(M):
        hband = blocks[M * j:M * (j+1)]
        hbands.append(tf.concat(1,hband))
    return(tf.concat(0,hbands))

def parametric_block_intensity(arr,sigmas,M,blk_s,debug_items=None):
    '''
    Sigma here does not mean blurring, I'm just continuing to use that symbol.
    '''
    img_blocks = tf.squeeze(toblocks(arr,M,blk_s))
    scales = tf.transpose(tf.maximum(tf.constant(1,dtype=PRECISION), sigmas))
    scales_shapes = tf.expand_dims(tf.expand_dims(scales,1),1)
    scaled = scales_shapes * img_blocks
    origs = []
    for v in tf.unpack(scaled):
        origs.append(fromblocks(tf.unpack(tf.transpose(v,perm=[2,0,1])),M))
    return tf.pack(origs),scales

def parametric_block_blur(arr,sigmas,M,conv_w,batchsize,blk_s,normalize=True,debug_items=None):
    '''
    '''
    try:
        inv = tf.inv
    except:
        inv = lambda x:1/x
    img_blocks = toblocks(arr,M,blk_s)
    xs,ys = np.meshgrid(np.arange(0,conv_w),np.arange(0,conv_w))
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    unscaled_filter = tf.constant(-0.5 * np.square(np.sqrt(np.abs(xs - conv_w//2)) + np.sqrt(np.abs(ys - conv_w//2))))
    dup_filters = tf.reshape(tf.pack(batchsize * M * M * [unscaled_filter]), [batchsize,M * M,conv_w,conv_w])
    kludge_sigmasq = tf.reshape(tf.maximum(tf.square(sigmas),tf.constant(1e-15)),[batchsize,M*M,1,1])
    gauss_constants = tf.constant(1 / math.sqrt(2 * math.pi)) * inv(tf.sqrt(kludge_sigmasq))
    unnorm_filters = tf.mul(gauss_constants,tf.exp(tf.div(dup_filters, kludge_sigmasq)))
    if normalize:
        filtersums = tf.reshape(tf.reduce_sum(unnorm_filters,reduction_indices=[2,3]),[batchsize,M * M,1,1])
        filters = tf.div(unnorm_filters,filtersums)
    else:
        filters = unnorm_filters
    filts = tf.unpack(filters)
    imgs_blurred = []
    for i in range(batchsize):
        filters_depthwise = tf.expand_dims(tf.transpose(filts[i],perm=[1,2,0]),3) 
        filt_out = tf.nn.depthwise_conv2d(img_blocks[i],filters_depthwise,[1,1,1,1],'SAME')
        blurred_blocks = tf.unpack(tf.squeeze(tf.transpose(filt_out,perm=[3,1,2,0])))
        imgs_blurred.append(fromblocks(blurred_blocks,M))
    return tf.reshape(tf.pack(imgs_blurred), arr.get_shape()), filters

def sigmanet_conv(bottom,weights,biases,batchsize,dropout,numfilts):
    '''
    In this case, determine sigmas as the values at the appropriate locations after repeated downsampling.
    '''
    ks = [7,4,2]
    s = bottom.get_shape()[1].value
    # keep doing convs until at MxM resolution. Then, dot products along channels to decide on sigmasq values.
    maps1 = conv2d('snet_conv0',bottom,weights['snet_0'],biases['snet_0'],leaky=True)
    pool1 = tf.nn.max_pool(maps1,ksize=[1,ks[0],ks[0],1],strides=[1,ks[0],ks[0],1],padding='SAME')
    lrn1 = lrn('snet1',pool1,lsize=4)
    maps2 = conv2d('snet_conv1',lrn1,weights['snet_1'],biases['snet_1'],leaky=True)
    pool2 = tf.nn.max_pool(maps2,ksize=[1,ks[1],ks[1],1],strides=[1,ks[1],ks[1],1],padding='SAME')
    lrn2 = lrn('snet1',pool2,lsize=4)
    if ks[2] > 1:
        maps3 = conv2d('snet_conv2',lrn2,weights['snet_2'],biases['snet_2'],leaky=True)
        pool3 = tf.nn.max_pool(maps3,ksize=[1,ks[2],ks[2],1],strides=[1,ks[2],ks[2],1],padding='SAME')
        Ms = lrn('snet1',pool3,lsize=4)
    else:
        Ms = lrn2
    # RELU is a good idea because zero means no blurring and we don't want to have negative numbers be symmetrical to positive numbers.
    # we want a range of inputs meaning "no blurring".
    out = []
    for filtid in range(numfilts):
        out.append(tf.nn.relu(tf.nn.conv2d(Ms,weights[f'snet_out_{filtid}'],[1,1,1,1],'SAME')))
    return out

def sigmanet_highlevel(bottom,weights,biases,batchsize,dropout):
    '''
    In this case, determine sigmas by full pass through VGG-net along with output matrix snet.
    '''
    return arch_common.vanilla(bottom,weights,biases,dropout,output_keys=['snet'])['snet']
    
def sigmanet_mlp(bottom,weights,biases,batchsize,dropout,i=None):
    '''
    In this case, determine sigmas with a multilayer perceptron.
    '''
    if i != None:
        key = f'sigma_{i}'
    else:
        key = 'sigma'
    flatnum = np.product([x.value for x in bottom.get_shape()]) / batchsize
    z1 = tf.nn.relu(tf.matmul(weights[key],tf.reshape(bottom,[flatnum,batchsize]),transpose_a=False) + tf.expand_dims(biases[key],1))
    if sigmanet_layers == 'mlp_1':
        return z1
    if i != None:
        hkey = f'hiddensigma_{i}'
    else:
        hkey = 'hiddensigma'
    z2 = tf.nn.relu(tf.matmul(weights[hkey],z1) + tf.expand_dims(biases[hkey],1))
    if sigmanet_layers == 'mlp_2':
        return z2

def ctx_pixels(X:tf.Tensor,Xfull:tf.Tensor,weights:weight_t,biases:weight_t,dropout:tf.Tensor,ctxop:str,numfilts:int,M:int,debug_items=None):
    '''
    '''
    ctxshape = (224,224,3)
    batchsize = X.get_shape()[0].value
    sigma_list,posts = [], []
    img_channels = tf.unpack(tf.transpose(Xfull))
    # It actually appears these lines are messing things up.
    # Maybe will have to use transpose instead of reshape.
    img_channels[0] = tf.expand_dims(tf.transpose(img_channels[0],perm=[2,1,0]),3)
    img_channels[1] = tf.expand_dims(tf.transpose(img_channels[1],perm=[2,1,0]),3)
    img_channels[2] = tf.expand_dims(tf.transpose(img_channels[2],perm=[2,1,0]),3)
    # we can assign functions to sigmanet because they share a common interface, returning an MxM block of parameters.
    if sigmanet_layers in ['mlp_1','mlp_2']:
        sigmanet = sigmanet_mlp
    elif sigmanet_layers == 'highlevel':
        sigmanet = sigmanet_highlevel
    elif sigmanet_layers == 'conv':
        sigmanet = sigmanet_conv
    sigmas_mat = sigmanet(X,weights,biases,batchsize,dropout,numfilts)
    sigma_list.append(sigmas_mat)
    for sigmas in sigmas_mat:
        if ctxop == 'block_blur':
            Rimg_blurred,rfilt = parametric_block_blur(img_channels[0],sigmas,M,constants.block_conv_w,batchsize,224//M,debug_items=debug_items)
            Gimg_blurred,gfilt = parametric_block_blur(img_channels[1],sigmas,M,constants.block_conv_w,batchsize,224//M,debug_items=debug_items)
            Bimg_blurred,bfilt = parametric_block_blur(img_channels[2],sigmas,M,constants.block_conv_w,batchsize,224//M,debug_items=debug_items)
            posts.append(tf.expand_dims(tf.concat(3,[Rimg_blurred,Gimg_blurred,Bimg_blurred]),0))
        elif ctxop == 'block_intensity':
            Rimg_darkened,rfilt = parametric_block_intensity(img_channels[0],sigmas,M,224//M,debug_items=debug_items)
            Gimg_darkened,gfilt = parametric_block_intensity(img_channels[1],sigmas,M,224//M,debug_items=debug_items)
            Bimg_darkened,bfilt = parametric_block_intensity(img_channels[2],sigmas,M,224//M,debug_items=debug_items)
            posts.append(tf.expand_dims(tf.transpose(tf.pack([Rimg_darkened,Gimg_darkened,Bimg_darkened]),perm=[1,2,3,0]),0))
    filts = tf.pack([rfilt,gfilt,bfilt])
    filts = tf.pack(filts)
    # do some reshaping to get batchsize to be outermost dimension, so when we unstack it will be correct.
    post = tf.concat(0,posts)
    post = tf.reshape(tf.transpose(post,[1,0,2,3,4]),[batchsize * numfilts,224,224,3])
    return post,sigma_list,filts

def arch_block(X,Xfull,parameters,dropout,batchsize,ctxop,numfilts,M,with_normalization=False,debug_items=None,devices=['/gpu:0']):
    '''
    This function is called arch_block because context representation layer uses sigmas determined by fully connected op.
    '''
    # initialize.
    weights,biases = parameters
    headout = arch_common.arch_head(weights,biases,X,dropout,keys=None)
    assert(224 % M == 0)
    with tf.device(devices[0 % len(devices)]):
        ctxout,sigma_list,filts = ctx_pixels(X,Xfull,weights,biases,dropout,ctxop,numfilts,M)
    with tf.device(devices[1 % len(devices)]):
        ctx_repr = arch_common.arch_tail(weights,biases,ctxout,dropout,output_keys=['similarity'],leaky=True)
    with tf.device(devices[2 % len(devices)]):
        can_repr = arch_common.vanilla(X,weights,biases,dropout,leaky=True)
    for k in can_repr.keys():
        if k == 'conv5': continue #don't bother.
        ctx_repr[k] =  tf.reshape(tf.concat(0,[can_repr[k],ctx_repr[k]]), [batchsize,numfilts,-1]) #concatenate candiate features.
    return sigma_list,ctx_repr,headout,ctxout,filts

def block_properties(sigmasout,Xs,splitid,M):
    '''
    Measure relationship between sigmasq values in blocks and various properties.
    E.g. proportion of object pixels.
         saliency.
    relative to baselines.
    '''
    salbins = np.linspace(-5,5,32)
    sals = []
    sigvec,salvec,diffs = [],[],[]
    for X in Xs:
        sals.append(arch_common.signature_saliency(X))
    sals = np.array(sals)
    for which in range(len(sigmasout)):
        for b in range(len(Xs)):
            for i in range(M-1):
                for j in range(M-1):
                    sv = sigmasout[which,b,i,j][0]
                    sigvec.append(sv)
                    bsal = np.mean(sals[b,(i * 224/M):((i+1) * (224/M)),(j * 224/M):((j+1) * 224/M)])
                    salvec.append(bsal)
                    diffs.append(sv - bsal)
    # interested in correlation, and also distribution of similarity.
    return {'saliency_corr' : pearsonr(sigvec,salvec),
            'distr' : np.histogram(diffs,bins=salbins)[0]}
