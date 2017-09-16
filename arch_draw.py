'''

'''
import numpy as np
import tensorflow as tf
import arch_common
from scipy.misc import imresize
from scipy.spatial.distance import euclidean
from typing import Callable
from utils import *
from mytypes import *

PRECISION = tf.float32

def decoder_vggnet(X:tf.Tensor,weights:weight_t,biases:weight_t,dropout:float,distinct_decode:bool,stop_grad:bool,debug_items=None,keep_resolution=False,keep_downsamp=2,stop_pos=3):
    '''
    Place these operations on the same GPU, because they are a pipeline and take up memory distinct from non-decoder.
    X : NHWC tensor input.
    weights: 
    biases: 
    
    '''
    pool_count = 0
    def cond_pool(name,x,count):
        if keep_resolution:
            return x,count
        else:
            return max_pool(name,x,k=2),count+1
    # difference between these conditionals : different keys into the parameters.
    if distinct_decode:
        conv1_1 = conv2d('decode-conv1_1', X, weights['decode-conv1_1'], biases['decode-conv1_1'])
        conv1_2 = conv2d('decode-conv1_2', conv1_1, weights['decode-conv1_2'], biases['decode-conv1_2'])
        pool1,pool_count = cond_pool('decode-pool1', conv1_2,pool_count)
        norm1 = lrn('decode-norm1', pool1, lsize=4)
        norm1 = tf.nn.dropout(norm1, dropout)
        if stop_pos > 1:
            conv2_1 = conv2d('decode-conv2_1', norm1, weights['decode-conv2_1'], biases['decode-conv2_1'])
            conv2_2 = conv2d('decode-conv2_2', conv2_1, weights['decode-conv2_2'], biases['decode-conv2_2'])
            pool2,pool_count = cond_pool('decode-pool2', conv2_2,pool_count)
            norm2 = lrn('decode-norm2', pool2, lsize=4)
            norm2 = tf.nn.dropout(norm2, dropout)
            if stop_pos > 2:
                conv3_1 = conv2d('decode-conv3_1', norm2, weights['decode-conv3_1'], biases['decode-conv3_1'])
                conv3_2 = conv2d('decode-conv3_2', conv3_1, weights['decode-conv3_2'], biases['decode-conv3_2'])
                conv3_3 = conv2d('decode-conv3_3', conv3_2, weights['decode-conv3_3'], biases['decode-conv3_3'])
                pool3,pool_count = cond_pool('decode-pool3', conv3_3,pool_count)
                feats = norm3 = lrn('decode-norm3', pool3, lsize=4)
            else:
                feats = norm2
        else:
            feats = norm1
    else:
        conv1_1 = conv2d('conv1_1', X, weights['conv1_1'], biases['conv1_1'])
        conv1_2 = conv2d('conv1_2', conv1_1, weights['conv1_2'], biases['conv1_2'])
        pool1,pool_count = cond_pool('pool1', conv1_2,pool_count)
        norm1 = lrn('norm1', pool1, lsize=4)
        norm1 = tf.nn.dropout(norm1, dropout)
        if stop_pos > 1:
            conv2_1 = conv2d('conv2_1', norm1, weights['conv2_1'], biases['conv2_1'])
            conv2_2 = conv2d('conv2_2', conv2_1, weights['conv2_2'], biases['conv2_2'])
            pool2,pool_count = cond_pool('pool2', conv2_2,pool_count)
            norm2 = lrn('norm2', pool2, lsize=4)
            norm2 = tf.nn.dropout(norm2, dropout)
            if stop_pos > 2:
                conv3_1 = conv2d('conv3_1', norm2, weights['conv3_1'], biases['conv3_1'])
                conv3_2 = conv2d('conv3_2', conv3_1, weights['conv3_2'], biases['conv3_2'])
                conv3_3 = conv2d('conv3_3', conv3_2, weights['conv3_3'], biases['conv3_3'])
                pool3,pool_count = cond_pool('pool3', conv3_3,pool_count)
                feats = norm3 = lrn('norm3', pool3, lsize=4)
            else:
                feats = norm2 
        else:
            feats = norm1
    feats = tf.nn.dropout(feats, dropout)
    b,h,w,c = feats.get_shape()[0].value,feats.get_shape()[1].value,feats.get_shape()[2].value,feats.get_shape()[3].value
    if keep_resolution:
        down = max_pool('big_decode_downsample',feats,4)
        prod = np.product([x.value for x in down.get_shape()])
        net_out = tf.reshape(down,[b,prod//b])
    else:
        net_out = tf.reshape(feats,[b,h*w*c])
    if stop_grad:
        net_out = tf.stop_gradient(net_out)
    out = tf.matmul(net_out,weights['decode']) + biases['decode']
    return tf.nn.relu(out),feats

def decoder_video( ):
    '''
     
    '''
    conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = max_pool('pool1', conv1, k=1)

    conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    pool2 = max_pool('pool2', conv2, k=2)

    # Convolution Layer
    conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    pool3 = max_pool('pool3', conv3, k=2)

    # Convolution Layer
    conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    pool4 = max_pool('pool4', conv4, k=2)

    # Convolution Layer
    conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')
    pool5 = max_pool('pool5', conv5, k=2)

    # Fully connected layer
    pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
    dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

    dense1 = tf.nn.relu(dense1, name='fc1') # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout)

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
    dense2 = tf.nn.dropout(dense2, _dropout)

    # Output: class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']

def ctx_draw_video( ):
    '''

    '''
    A,B,numchan = headout_full.get_shape().dims[2].value,headout_full.get_shape().dims[3].value,headout_full.get_shape().dims[0].value
    filtsize = 4 #patches become 56x56m which is 1/4 of the full size.
    N = A // filtsize
    identical_cols = np.tile(np.arange(A),N * batchsize).reshape((batchsize,N,A)).astype(np.float32)
    # i could make this same range-const with numpy to remove a bit from the graph.
    range_const = tf.constant(np.tile(np.arange(N) - N/2 - 0.5,[batchsize,1]),dtype=PRECISION)
    #range_const = tf.reshape(tf.tile(tf.constant(np.arange(N) - N/2 - 0.5,dtype=tf.float32),[batchsize]), [batchsize,N])
    x_adjust,y_adjust = tf.constant((A+1)/2,dtype=PRECISION), tf.constant((B+1)/2,dtype=PRECISION)
    stride_adjust = tf.constant((max(A,B)-1)/(N-1))
    epsilon = tf.constant(1e-20)
    # multiply by 224 because these are (0,1) values.
    if tf.__version__[0] == '0':
        unpack,sub,mul,batch_matmul = tf.unpack,tf.sub,tf.mul,tf.batch_matmul
        concat = lambda x,y: tf.concat(x,y)
    else:
        unpack,sub,mul,batch_matmul = tf.unstack,tf.subtract,tf.multiply,tf.matmul
        concat = lambda x,y: tf.concat(y,x)
    min_ys,max_ys,min_xs,max_xs = unpack(tf.transpose(bbox))
    patches,filts,attention_numbers,attention_boxes = [],[],[],[]
    pixmax = tf.reduce_max(headout_full,[0,2,3],keep_dims=False)
    try:
        inv = tf.inv
    except:
        inv = lambda x:1/x
    for i in range(numfilts-1):
        # these are vectors over the batch.
        if baseline_t in ['full','fixed_pos','attentiononly']: # I accidentally used non-zero weights for 'fixed' baseline_t so stuck with that name.
            gp_X,gp_Y,sigmasqp, stridep, intensityp = unpack(tf.matmul(weights[f'attention_{i}'],tf.transpose(h_dec)) + tf.expand_dims(biases[f'attention_{i}'],1))
        elif baseline_t in ['biasonly','fixed-biasonly']: #
            # this times zero is necessary because weights must not do anything but they must seem to exist to tensorflow, because i'm doing fancy stuff with gradient
            # which doesn't handle missing parameters well.
            gp_X,gp_Y,sigmasqp, stridep, intensityp = unpack(0 * tf.matmul(weights[f'attention_{i}'],tf.transpose(h_dec)) + tf.expand_dims(biases[f'attention_{i}'],1))
        invsigmasq = inv(tf.exp(sigmasqp) + epsilon) #epsilon to avoid division by zero errors.
        if prevent_zoomout:
            stridep = tf.minimum(stridep,tf.constant(0,dtype=PRECISION))
        stride = tf.expand_dims(stride_adjust * tf.exp(stridep),1)
        # decided to not use intensity.
        #intensity = tf.exp(intensityp)
        g_X =  x_adjust * (gp_X + tf.constant(1,dtype=PRECISION)) 
        g_Y =  y_adjust * (gp_Y + tf.constant(1,dtype=PRECISION))
        # here, g_X and g_Y have draw-paper meanings. They are abolute coordinates in (0,224) x (0,224)
        if relative_arch:
            rel_X = (max_xs - min_xs)/2 + g_X-(img_s/2) # an adjusted g_X to be relative to object candidate.
            rel_Y = (max_ys - min_ys)/2 + g_Y-(img_s/2) # so g_X being 112 is like zero.
            # if too far to the left or right, place at the corresonding border.
            g_X = tf.maximum(tf.minimum(rel_X,img_s-border),border) #at least 'border' pixel away from edge.
            g_Y = tf.maximum(tf.minimum(rel_Y,img_s-border),border)
        # determines mean location at row i
        mu_X = tf.expand_dims(g_X,1) + range_const * stride
        # determines mean location at column j
        mu_Y = tf.expand_dims(g_Y,1) + range_const * stride
        # the raw versions are not normalized.
        expscale = tf.reshape(tf.constant(-1/2,dtype=PRECISION) * invsigmasq,[batchsize,1,1])
        # stride=1 means no zoom so dy would be 224/4 or generally A/filtsize.
        patch_dy = 0.5 * (A/filtsize) * tf.squeeze(stride)
        patch_dx = 0.5 * (B/filtsize) * tf.squeeze(stride)
        attention_boxes.append([g_Y - patch_dy,g_X - patch_dx,g_Y + patch_dy,g_X + patch_dx])
        attention_numbers.append({'filtid':i,'gp_X' : gp_X,'gp_Y' : gp_Y,'sigmasqp' : sigmasqp,'stridep' : stridep,'intensityp' : intensityp})
    #make batchid is now the outermost dimension
    patches = tf.transpose(patches,perm=[1,0,2,3,4])
    return patches,tf.identity(filts),attention_numbers,attention_boxes

def ctx_draw(weights:Dict,biases:Dict,headout_full:tf.Tensor,h_dec:tf.Tensor,bbox:tf.Tensor,dropout:tf.Tensor, \
             batchsize:int,relative_arch:bool,baseline_t:str,numfilts:int,debug_items=None,prevent_zoomout=True, \
             border=0,get_patches=True,img_s=224) -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor]:
    '''
    weights:
    biases:
    headout_full: NHWC input tensor representing either the data or .
    h_dec: 
    bbox: 
    dropout: scalar tensor of named signifiance.
    batchsize: this function requires this to be determined, so this can't be a placeholder.
    relative_arch
    baseline_t: controls how the attention parameters are determined. 
        if full, the usual idea.
        if fixed-biasonly, use and learn only biases.
        if fixed-pos, attention is not learned, it keeps the initialization.
        if attentiononly, 
    This operates on one channel at a time.

    Returns
    '''
    A,B,numchan = headout_full.get_shape().dims[2].value,headout_full.get_shape().dims[3].value,headout_full.get_shape().dims[0].value
    filtsize = 4 #patches become 56x56m which is 1/4 of the full size.
    N = A // filtsize
    identical_cols = np.tile(np.arange(A),N * batchsize).reshape((batchsize,N,A)).astype(np.float32)
    # i could make this same range-const with numpy to remove a bit from the graph.
    range_const = tf.constant(np.tile(np.arange(N) - N/2 - 0.5,[batchsize,1]),dtype=PRECISION)
    #range_const = tf.reshape(tf.tile(tf.constant(np.arange(N) - N/2 - 0.5,dtype=tf.float32),[batchsize]), [batchsize,N])
    x_adjust,y_adjust = tf.constant((A+1)/2,dtype=PRECISION), tf.constant((B+1)/2,dtype=PRECISION)
    stride_adjust = tf.constant((max(A,B)-1)/(N-1))
    epsilon = tf.constant(1e-20)
    # multiply by 224 because these are (0,1) values.
    if tf.__version__[0] == '0':
        unpack,sub,mul,batch_matmul = tf.unpack,tf.sub,tf.mul,tf.batch_matmul
        concat = lambda x,y: tf.concat(x,y)
    else:
        unpack,sub,mul,batch_matmul = tf.unstack,tf.subtract,tf.multiply,tf.matmul
        concat = lambda x,y: tf.concat(y,x)
    min_ys,max_ys,min_xs,max_xs = unpack(tf.transpose(bbox))
    patches,filts,attention_numbers,attention_boxes = [],[],[],[]
    pixmax = tf.reduce_max(headout_full,[0,2,3],keep_dims=False)
    try:
        inv = tf.inv
    except:
        inv = lambda x:1/x
    for i in range(numfilts-1):
        # these are vectors over the batch.
        if baseline_t in ['full','fixed_pos','attentiononly']: # I accidentally used non-zero weights for 'fixed' baseline_t so stuck with that name.
            gp_X,gp_Y,sigmasqp, stridep, intensityp = unpack(tf.matmul(weights[f'attention_{i}'],tf.transpose(h_dec)) + tf.expand_dims(biases[f'attention_{i}'],1))
        elif baseline_t in ['biasonly','fixed-biasonly']: #
            # this times zero is necessary because weights must not do anything but they must seem to exist to tensorflow, because i'm doing fancy stuff with gradient
            # which doesn't handle missing parameters well.
            gp_X,gp_Y,sigmasqp, stridep, intensityp = unpack(0 * tf.matmul(weights[f'attention_{i}'],tf.transpose(h_dec)) + tf.expand_dims(biases[f'attention_{i}'],1))
        invsigmasq = inv(tf.exp(sigmasqp) + epsilon) #epsilon to avoid division by zero errors.
        if prevent_zoomout:
            stridep = tf.minimum(stridep,tf.constant(0,dtype=PRECISION))
        stride = tf.expand_dims(stride_adjust * tf.exp(stridep),1)
        # decided to not use intensity.
        #intensity = tf.exp(intensityp)
        g_X =  x_adjust * (gp_X + tf.constant(1,dtype=PRECISION)) 
        g_Y =  y_adjust * (gp_Y + tf.constant(1,dtype=PRECISION))
        # here, g_X and g_Y have draw-paper meanings. They are abolute coordinates in (0,224) x (0,224)
        if relative_arch:
            rel_X = (max_xs - min_xs)/2 + g_X-(img_s/2) # an adjusted g_X to be relative to object candidate.
            rel_Y = (max_ys - min_ys)/2 + g_Y-(img_s/2) # so g_X being 112 is like zero.
            # if too far to the left or right, place at the corresonding border.
            g_X = tf.maximum(tf.minimum(rel_X,img_s-border),border) #at least 'border' pixel away from edge.
            g_Y = tf.maximum(tf.minimum(rel_Y,img_s-border),border)
        # determines mean location at row i
        mu_X = tf.expand_dims(g_X,1) + range_const * stride
        # determines mean location at column j
        mu_Y = tf.expand_dims(g_Y,1) + range_const * stride
        # the raw versions are not normalized.
        expscale = tf.reshape(tf.constant(-1/2,dtype=PRECISION) * invsigmasq,[batchsize,1,1])
        F_X = tf.exp(expscale * tf.square(sub(identical_cols,tf.expand_dims(mu_X,2))))
        F_Y = tf.exp(expscale * tf.square(sub(identical_cols,tf.expand_dims(mu_Y,2))))
        X_norm = tf.maximum(epsilon,tf.reduce_sum(F_X,2,keep_dims=True))
        Y_norm = tf.maximum(epsilon,tf.reduce_sum(F_Y,2,keep_dims=True))
        F_X = F_X / X_norm
        F_Y = F_Y / Y_norm
        # this "multiplication" is duplication, to make it work with channels. it is NOT scaling.
        # intensity_shaped = tf.expand_dims(tf.expand_dims(intensity,0),2),3)
        filtered = batch_matmul(batch_matmul(tf.identity(numchan * [F_X]),headout_full),tf.transpose(numchan * [F_Y],perm=[0,1,3,2]))
        try:
            patches.append(tf.image.resize_images(tf.transpose(filtered,[1,2,3,0]),[img_s,img_s]))
        except:
            patches.append(tf.image.resize_images(tf.transpose(filtered,[1,2,3,0]),img_s,img_s))
        filts.append([F_X,F_Y])
        # stride=1 means no zoom so dy would be 224/4 or generally A/filtsize.
        patch_dy = 0.5 * (A/filtsize) * tf.squeeze(stride)
        patch_dx = 0.5 * (B/filtsize) * tf.squeeze(stride)
        attention_boxes.append([g_Y - patch_dy,g_X - patch_dx,g_Y + patch_dy,g_X + patch_dx])
        attention_numbers.append({'filtid':i,'gp_X' : gp_X,'gp_Y' : gp_Y,'sigmasqp' : sigmasqp,'stridep' : stridep,'intensityp' : intensityp})
    #make batchid is now the outermost dimension
    patches = tf.transpose(patches,perm=[1,0,2,3,4])
    return patches,tf.identity(filts),attention_numbers,attention_boxes

def arch_draw_video(X:tf.Tensor,Xfull:tf.Tensor,Xcfull:tf.Tensor,bbox_loc:tf.Tensor,parameters:parameters_t,dropout:tf.Tensor,\
                    batchsize:int,relative_arch:bool,distinct_decode:bool,loss_t:str,baseline_t:str,numfilts:int,include_center:bool,\
                    decode_arch,stop_grad,debug_items=None,devices=['/gpu:0'],keep_resolution=False,img_s=224,crop_reduce=2 ):
  conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
  conv1 = tf.nn.relu(conv1, 'relu1')
  pool1 = max_pool('pool1', conv1, k=1)
  conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
  conv2 = tf.nn.relu(conv2, 'relu2')
  pool2 = max_pool('pool2', conv2, k=2)
  conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
  conv3 = tf.nn.relu(conv3, 'relu3a')
  conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
  conv3 = tf.nn.relu(conv3, 'relu3b')
  pool3 = max_pool('pool3', conv3, k=2)
  conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
  conv4 = tf.nn.relu(conv4, 'relu4a')
  conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
  conv4 = tf.nn.relu(conv4, 'relu4b')
  pool4 = max_pool('pool4', conv4, k=2)
  conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
  conv5 = tf.nn.relu(conv5, 'relu5a')
  conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
  conv5 = tf.nn.relu(conv5, 'relu5b')
  pool5 = max_pool('pool5', conv5, k=2)
  pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
  dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
  dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']
  dense1 = tf.nn.relu(dense1, name='fc1') # Relu activation
  dense1 = tf.nn.dropout(dense1, _dropout)
  dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
  dense2 = tf.nn.dropout(dense2, _dropout)
  out = tf.matmul(dense2, _weights['out']) + _biases['out']
  return out

def arch_draw_shared(X:tf.Tensor,Xfull:tf.Tensor,Xcfull:tf.Tensor,bbox_loc:tf.Tensor,parameters:parameters_t,dropout:tf.Tensor,\
                    batchsize:int,relative_arch:bool,distinct_decode:bool,loss_t:str,baseline_t:str,numfilts:int,include_center:bool,\
                    decode_arch,stop_grad,stop_pos=3,debug_items=None,devices=['/gpu:0'],keep_resolution=False,img_s=224,crop_reduce=2):
    if tf.__version__[0] == '0':
        unpack,sub,mul,batch_matmul = tf.unpack,tf.sub,tf.mul,tf.batch_matmul
        concat = lambda x,y: tf.concat(x,y)
    else:
        unpack,sub,mul,batch_matmul = tf.unstack,tf.subtract,tf.multiply,tf.matmul
        concat = lambda x,y: tf.concat(y,x)
    weights,biases = parameters
    if decode_arch == "alexnet":
        net_data = load("bvlc_alexnet.npy").item()
        decoder_out = decoder_alexnet
    elif decode_arch == "vggnet":
        decoder_out = decoder_vggnet
    with tf.device(devices[0 % len(devices)]):
        headout_full = arch_common.arch_head(weights,biases,Xfull,dropout,keys=None,keep_resolution=keep_resolution) #not artifically centered.
        s = headout_full.get_shape()[1].value #should be 224 if conv0, 56 if conv2.
        dout_full = decoder_out(Xcfull,weights,biases,dropout,distinct_decode,stop_grad,debug_items=debug_items,keep_resolution=keep_resolution,stop_pos=stop_pos) #yes artificially centered.
        # decoder output always includes conv3, hence this variable name.
        conv3_full = dout_full[1]
    with tf.device(devices[1 % len(devices)]):
        dout = decoder_out(X,weights,biases,dropout,distinct_decode,stop_grad,debug_items=debug_items,keep_resolution=keep_resolution,stop_pos=stop_pos)
        dlist = [dout[0],dout_full[0]] if relative_arch else [dout,dout_full,bbox_loc]
        h_dec = concat(1,dlist) #using kwargs because different tensorflow versions appear to want different positional orderings.
        conv3 = dout[1]
        _,conv3_h,conv3_w,_ = [x.value for x in conv3.get_shape()]
        crop_shape = [conv3_h//crop_reduce,conv3_w//crop_reduce]
    with tf.device(devices[2 % len(devices)]):
        if numfilts > 1:
            ctxouts,filtout,attentionout,attention_boxes = ctx_draw(weights,biases,tf.transpose(headout_full,perm=[3,0,1,2]),h_dec,bbox_loc,dropout,batchsize,relative_arch,baseline_t,numfilts,debug_items=debug_items)
        else:
            attention_boxes,filtout,attentionout,ctxouts = [],None,None,None #
    if include_center:
        featmaps = [tf.image.resize_bilinear(conv3,crop_shape)]
    else:
        featmaps = []
    with tf.device(devices[3 % len(devices)]):
        for attention_box in attention_boxes: #iterating over filters.
            upsample_prop = img_s / conv3_h
            ys,xs,dys,dxs = [tf.expand_dims(x,1)/upsample_prop for x in attention_box]
            crbox = concat(1,[ys,xs,ys+dys,xs+dxs])
            # I need to "slice" the batch and the bbox_trans here.
            # need to make the featuremap crops some consistent size, 112x112 sounds okay.
            featmaps.append(tf.image.crop_and_resize(conv3,crbox,np.arange(batchsize),crop_shape))
    # need to concatenate featmaps here.
    with tf.device(devices[4 % len(devices)]):
        # batch needs to be outermost, infer the last dimension.
        nf = numfilts if include_center else numfilts - 1
        flat_correct = tf.reshape(tf.transpose(featmaps,perm=[1,0,2,3,4]),[batchsize * nf,conv3_h//2,conv3_w//2,-1])
        ok = ['similarity','classify'] if loss_t == 'dual' else ['similarity']
        out = arch_common.arch_tail(weights,biases,flat_correct,dropout,output_keys=ok,leaky=True,apply_pos='conv' + str(stop_pos),keep_resolution=keep_resolution)
        # all this really is necessary to get the (batchsize,numfilts,featsize) shape. The ways of lining the data up correctly that looked simpler were wrong in their output.
        out['similarity'] = tf.reshape(out['similarity'],[batchsize,nf,-1]) #-1 means infer shape.
        if loss_t == "dual":
            out['classify'] = tf.reshape(out['classify'],[batchsize,nf,-1]) #-1 means infer shape.
    return out,filtout,ctxouts,attentionout,headout_full,attention_boxes

def arch_draw_distinct(X:tf.Tensor,Xfull:tf.Tensor,Xcfull:tf.Tensor,bbox_loc:tf.Tensor,parameters:parameters_t,dropout:parameters_t,batchsize:int,relative_arch,distinct_decode,loss_t,baseline_t,numfilts,include_center,decode_arch,stop_grad,debug_items=None,devices=['/gpu:0'],img_s=224,stop_pos=3):
    '''
    bbox_loc should be on a [0,1] scale in both x and y.
    The shape of output is (batchsize,numfilts,numfeats) [whereas the output of ctx_draw is a partial transpose of this, so I apply that transpose here].
    '''
    weights,biases = parameters
    if tf.__version__[0] == '0':
        unpack,sub,mul,batch_matmul = tf.unpack,tf.sub,tf.mul,tf.batch_matmul
        concat = lambda x,y: tf.concat(x,y)
    else:
        unpack,sub,mul,batch_matmul = tf.unstack,tf.subtract,tf.multiply,tf.matmul
        concat = lambda x,y: tf.concat(y,x)
    if decode_arch == "alexnet":
        net_data = load("bvlc_alexnet.npy").item()
        decoder_out = decoder_alexnet
    elif decode_arch == "vggnet":
        decoder_out = decoder_vggnet
    with tf.device(devices[0 % len(devices)]):
        headout_full = arch_common.arch_head(weights,biases,Xfull,dropout,keys=None) #not artifically centered.
        s = headout_full.get_shape()[1].value #should be 224 if conv0, 56 if conv2.
        dout_full = decoder_out(Xcfull,weights,biases,dropout,distinct_decode,stop_grad,debug_items=debug_items,stop_pos=stop_pos) #yes artificially centered.
    with tf.device(devices[1 % len(devices)]):
        dout = decoder_out(X,weights,biases,dropout,distinct_decode,stop_grad,debug_items=debug_items,stop_pos=stop_pos)
        dlist = [dout[0],dout_full[0]] if relative_arch else [dout,dout_full,bbox_loc]
        h_dec = concat(1,dlist)
    with tf.device(devices[2 % len(devices)]):
        ctxouts,filtout,attentionout,attention_boxes = ctx_draw(weights,biases,tf.transpose(headout_full,perm=[3,0,1,2]),h_dec,bbox_loc,dropout,batchsize,relative_arch,baseline_t,numfilts,debug_items=debug_items)
        if include_center:
            patches = concat(1,[tf.expand_dims(X,1),ctxouts])
        else:
            numfilts -= 1 #since we don't include it, but our main shape arithmetic assumes the last is the center.
            patches = ctxouts
        ok = ['similarity','classify'] if loss_t == 'dual' else ['similarity']
        flat_patches = tf.reshape(patches,[batchsize * numfilts,img_s,img_s,3])
    with tf.device(devices[3 % len(devices)]):
        out = arch_common.arch_tail(weights,biases,flat_patches,dropout,output_keys=ok,leaky=True,apply_pos='conv' + str(stop_pos))
    # all this really is necessary to get the (batchsize,numfilts,featsize) shape. The ways of lining the data up correctly that looked simpler were wrong in their output.
        out['similarity'] = tf.reshape(out['similarity'],[batchsize,numfilts,-1]) #-1 means infer shape.
        if loss_t == "dual":
            out['classify'] = tf.reshape(out['classify'],[batchsize,numfilts,-1]) #-1 means infer shape.
    return out,filtout,patches,attentionout,headout_full,attention_boxes

def random_patch(Xfull:np.ndarray,bbox,img_s=224):
    dx,dy = (bbox[3]-bbox[2]),(bbox[1]-bbox[0])
    minx = 0 if (dx >= (img_s-1)) else random.choice(list(range(0,int(img_s-dx))))
    miny = 0 if (dy >= (img_s-1)) else random.choice(list(range(0,int(img_s-dy))))
    maxy,maxx = (miny+dy),(minx+dx)
    return Xfull[int(miny):int(maxy),int(minx):int(maxx)],[int(miny),int(maxy),int(minx),int(maxx)]

def rect_union(a:bbox_t,b:bbox_t) -> bbox_t:
    '''
    next two functions lifted from http://answers.opencv.org/question/90455/how-to-perform-intersection-or-union-operations-on-a-rect-in-python/.
    '''
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def rect_intersection(a:bbox_t,b:bbox_t) -> bbox_t:
    '''
    next two functions lifted from http://answers.opencv.org/question/90455/how-to-perform-intersection-or-union-operations-on-a-rect-in-python/.
    '''
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return (0,0,0,0)
    return (x, y, w, h)

def iou_area(box1:bbox_t,box2:bbox_t) -> float:
    inter,uni = rect_intersection(box1,box2),rect_union(box1,box2) 
    inter_area,uni_area = (inter[2]-inter[0]) * (inter[3] - inter[1]),(uni[2]-uni[0]) * (uni[3] - uni[1])
    if uni_area == 0:
        return 0
    else:
        return abs(inter_area / uni_area)

def quantify_patches(hyperparams,Xs:np.ndarray,Xfulls:np.ndarray,imgnames:List[str],bboxs:np.ndarray,posts,nickname:str,splitid:int,img_s=224):
    '''
    Quantify the saliency.
    Quantify the distance to the nearest object from the ground truth.
    Quantify the intersection over union with the nearest object from the ground truth.

    Xs:
    Xfulls:
    imgnames:
    bboxs: format of bboxs here: ymin,ymax,xmin,xmax
    posts:
    nickname:
    splitid:
   
    Returns 
    '''
    sals,sals_rand = [], []
    iou_range = (0,1) # thankfully, a real 0 to 1 quantity.
    iou_bins = 16
    N = len(Xs)
    assert(N == len(Xfulls) == len(imgnames) == len(bboxs))
    assert(bboxs.shape == (N,4))
    sal_range = [0,5]
    closest_range = [0,1]
    sal_bins = 16
    closest_bins = 16 
    for X in Xs:
        sals.append(arch_common.signature_saliency(X))
    rs = [random_patch(Xfulls[i],bboxs[i]) for i in range(N)]
    for i,Xfull in enumerate(Xfulls):
        sals_rand.append(arch_common.signature_saliency(imresize(rs[i][0],(img_s,img_s)))) #zero for not needing the box info itself.
    dists,rdists = [],[]
    ious,rious = [],[]
    pik = readsql(f"select height,width,patchname from perfect_isknown NATURAL JOIN imgsize WHERE splitid = {splitid}",hyperparams).set_index('patchname')
    for i,imgname in enumerate(imgnames):
        try:
            gtboxes = readsql(f"select * from perfect_bbox where imgname = (select imgname from perfect_bbox where patchname = '{imgname}')",hyperparams)
            bbox = bboxs[i]
            ymin,ymax,xmin,xmax=bbox
            dy,dx = (bbox[1] - bbox[0]),(bbox[3] - bbox[2])
            assert(len(bbox) == 4)
            c = bbox[0] + dy/2,bbox[2] + dx/2
            r = rs[i][1][0] + (rs[i][1][1] - rs[i][1][0])/2,rs[i][1][2] + (rs[i][1][3] - rs[i][1][2])/2
            box_ious,box_rious=[],[]
            for _,gtbox in gtboxes.iterrows():
                gh,gw = pik.ix[imgname][['height','width']]
                gtbox['miny'] *= img_s/gh
                gtbox['maxy'] *= img_s/gh
                gtbox['minx'] *= img_s/gw
                gtbox['maxx'] *= img_s/gw
                dists.append(min(euclidean(c,[gtbox['miny'],gtbox['minx']]), euclidean(c,[gtbox['maxy'],gtbox['minx']]),
                                 euclidean(c,[gtbox['miny'],gtbox['maxx']]), euclidean(c,[gtbox['maxy'],gtbox['maxx']])))
                rdists.append(min(euclidean(r,[gtbox['miny'],gtbox['minx']]), euclidean(r,[gtbox['maxy'],gtbox['minx']]),
                                 euclidean(r,[gtbox['miny'],gtbox['maxx']]), euclidean(r,[gtbox['maxy'],gtbox['maxx']])))
                gbox= [gtbox['miny'],gtbox['minx'],gtbox['maxy'],gtbox['maxx']]
                rebox = [ymin,xmin,ymax,xmax] #permutation which  iou expects.
                yrand = random.randrange(0,int(img_s-dy))
                xrand = random.randrange(0,int(img_s-dx))
                # preserving width and height.
                randbox = [yrand,xrand,yrand+dy,xrand+dx]
                box_ious.append(iou_area(gbox,np.array(rebox)))
                box_rious.append(iou_area(gbox,np.array(randbox)))
            ious.append(max(box_ious))
            rious.append(max(box_rious))
        except:
            print(f"Failed to get 'closer' information on imgname={imgname}")
    return {'saliency' : [{'selected' : True, 'histo' : np.histogram(np.array(sals).flatten(),bins=sal_bins,range=sal_range)},
                          {'selected' : False, 'histo' : np.histogram(np.array(sals_rand).flatten(),bins=sal_bins,range=sal_range)}],
            'closest' : [{'selected' : True, 'histo' : np.histogram(np.array(dists)/img_s,bins=closest_bins,range=closest_range)},
                         {'selected' : False, 'histo' : np.histogram(np.array(rdists)/img_s,bins=closest_bins,range=closest_range)}],
            'iou' : [{'selected' : True, 'histo' : np.histogram(ious,bins=iou_bins,range=iou_range)},
                     {'selected' : False, 'histo' : np.histogram(rious,bins=iou_bins,range=iou_range)}]}

def draw_switch(hyperparams) -> Callable:
    if hyperparams.compute_t == 'shared':
        arch_draw = arch_draw_shared
    elif hyperparams.compute_t == 'full':
        arch_draw = arch_draw_distinct
    return arch_draw
