import numpy as np
import tensorflow as tf
import arch_common
from utils import *

def arch_patch(X,Xfull,Xcfull,parameters,dropout,batchsize,loss_t,ctxop,debug_items=None,devices=['/gpu:0']):
    with tf.device(devices[0]):
        assert(ctxop in ["above_below","patches","expand"])
        weights,biases = parameters
        if ctxop in ["above_below","patches"]:
            imgs = Xcfull
        elif ctxop == "expand":
            imgs = X
        headout_full = arch_common.arch_head(weights,biases,imgs,dropout,keys=None)
        ok = ['similarity','classify'] if loss_t == 'dual' else ['similarity']
        out = arch_common.arch_tail(weights,biases,headout_full,dropout,output_keys=ok,leaky=True)
        if ctxop == "above_below":
            out['similarity'] = tf.concat(1,tf.split(0,2,tf.expand_dims(out['similarity'],1)))
        if loss_t == "dual":
            out['classify'] = tf.concat(1,tf.split(0,2,tf.expand_dims(out['classify'],1)))
    return out,None,None,None,headout_full #nones to match the interface of other arch_< > functions.

def refine_single(hyperparams,feed:feed_t,Xfull_placeholder,Xfull,X_placeholder,X,Xcfull_placeholder,Xcfull,bboxs,img_s=224) -> dict:
    '''
    For the static patch-based methods, make the adjustments here.
    '''
    if hyperparams.ctxop == 'above_below':
        feed[X_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,0:112,0:img_s,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,112:img_s,0:img_s,:]]])
        feed[Xcfull_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,0:112,0:img_s,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,112:img_s,0:img_s,:]]])
        del(feed[Xfull_placeholder])
    elif hyperparams.ctxop == "patches":
        feed[Xcfull_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,0:100,0:120,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,140:210,0:80,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,90:170,170:img_s-1,:]]])
        feed[X_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,0:100,0:120,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,140:210,0:80,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,90:170,170:img_s-1,:]]])
    elif hyperparams.ctxop == "expanded": #making X an expanded version of itself using bbox and Xfull information.
        for i,x in enumerate(Xfull):
            bb = bboxs[i]
            dh,dh = bb[1] - bb[0], bb[3] - bb[2]
            expanded_bbox = [bb[0] - min(0.5 * dh,bb[0]),bb[1] + min(0.5 * dh,img_s - bb[1]),
                             bb[2] - min(0.5 * dw,bb[2]),bb[3] + min(0.5 * dw,img_s - bb[3])]
            feed[X_placeholder][i] = x[expanded_bbox] 
    return feed
    
def refine_feed(hyperparams,feed,X_placeholder,Xp_placeholder,Xfull_placeholder,Xpfull_placeholder,Xcfull_placeholder,Xcpfull_placeholder,X,Xp,Xfull,Xpfull,Xcfull,Xcpfull,y_placeholder,equal,ya_placeholder,yb_placeholder,ya,yb,bboxs,bboxsp,img_s=224) -> dict:
    '''
    For the static patch-based methods, make the adjustments here.
    '''
    if hyperparams.ctxop == 'above_below':
        feed[Xcfull_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,0:img_s/2,0:img_s,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,img_s/2:img_s,0:img_s,:]]])
        feed[Xcpfull_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,0:img_s/2,0:img_s,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xcpfull[:,img_s/2:img_s,0:img_s,:]]])
        feed[X_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,0:img_s/2,0:img_s,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,img_s/2:img_s,0:img_s,:]]])
        feed[Xp_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in Xp[:,0:img_s/2,0:img_s,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xp[:,img_s/2:img_s,0:img_s,:]]])
        del(feed[Xfull_placeholder])
        del(feed[Xpfull_placeholder])
        if ya is not None and yb is not None:
            nclasses = ya.shape[1]
            feed[ya_placeholder] = onehot(np.tile(np.argmax(ya,axis=1),2),nclasses)
            feed[yb_placeholder] = onehot(np.tile(np.argmax(yb,axis=1),2),nclasses)
    elif hyperparams.ctxop == "patches":
        feed[Xcfull_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,0:100,0:120,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,140:210,0:80,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xcfull[:,90:170,170:img_s-1,:]]])
        feed[Xcpfull_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in Xcpfull[:,0:100,0:120,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xcpfull[:,140:210,0:80,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xcpfull[:,90:170,170:img_s-1,:]]])
        feed[X_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,0:100,0:120,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,140:210,0:80,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in X[:,90:170,170:img_s-1,:]]])
        feed[Xp_placeholder] = np.vstack([[img_as_float(imresize(x,(img_s,img_s))) for x in Xp[:,0:100,0:120,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xp[:,140:210,0:80,:]],[img_as_float(imresize(x,(img_s,img_s))) for x in Xp[:,90:170,170:img_s-1,:]]])
        if ya is not None and yb is not None:
            nclasses = ya.shape[1]
            feed[y_placeholder] = onehot(np.tile(np.argmax(ya,axis=1),3),nclasses)
            feed[yp_placeholder] = onehot(np.tile(np.argmax(yb,axis=1),3),nclasses)
    elif hyperparams.ctxop == "expand":
        for i,x in enumerate(Xfull):
            bb = bboxs[i]
            dh,dw = bb[1] - bb[0], bb[3] - bb[2]
            expanded_bbox = [bb[0] - min(0.5 * dh,bb[0]),bb[1] + min(0.5 * dh,img_s-1 - bb[1]),
                             bb[2] - min(0.5 * dw,bb[2]),bb[3] + min(0.5 * dw,img_s-1 - bb[3])]
            feed[X_placeholder][i] = img_as_float(imresize(x[expanded_bbox[0]:expanded_bbox[1],expanded_bbox[2]:expanded_bbox[3]],(img_s,img_s)))
        for i,x in enumerate(Xpfull):
            bb = bboxsp[i]
            dh,dw = bb[1] - bb[0], bb[3] - bb[2]
            expanded_bbox = [bb[0] - min(0.5 * dh,bb[0]),bb[1] + min(0.5 * dh,img_s-1 - bb[1]),
                             bb[2] - min(0.5 * dw,bb[2]),bb[3] + min(0.5 * dw,img_s-1 - bb[3])]
            feed[Xp_placeholder][i] = img_as_float(imresize(x[expanded_bbox[0]:expanded_bbox[1],expanded_bbox[2]:expanded_bbox[3]],(img_s,img_s)) )
    return feed
