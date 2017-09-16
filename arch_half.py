import tensorflow as tf
import subprocess
from arch_draw import arch_draw
import pickle
from arch_common import initialize,arch_args

args,params = arch_args()
splitid = int(args.splitid)
split = params.possible_splits[splitid]
nclasses,nickname = len(split['known']),args.nickname
with tf.Session( ) as sess:
    batchsize = 1
    X_placeholder = tf.placeholder(dtype=tf.float32, shape=[batchsize, 224,224,3],name="X")
    Xfull_placeholder = tf.placeholder(dtype=tf.float32, shape=[batchsize, 224,224,3],name="X")
    y_placeholder = tf.placeholder(dtype=tf.float32, shape=[batchsize],name="eq")
    bbox_loc = tf.placeholder(dtype=tf.float32,shape=[batchsize,4],name="bbox_loc")
    dropout = tf.placeholder(tf.float32,name="dropout")
    parameters = initialize(params.M,args,nclasses,initialization=params.initialization)
    arch = arch_draw(X_placeholder,Xfull_placeholder,bbox_loc,parameters,dropout,batchsize,True,False,"contrastive")
    logdir = '/tmp/drawhalf'
    subprocess.call(['mkdir',logdir])
    writer = tf.train.SummaryWriter(logdir,sess.graph,flush_secs=360)
