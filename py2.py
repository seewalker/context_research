'''
Alex Seewald 2016
aseewald@indiana.edu

This project is done in python 3, however caffe and coco modules require python 2.7. 
This file contains the bits and pieces of code which are tied to python 2.
'''

import sys
import os
import math
import itertools
import numpy as np
import sqlite3
import psycopg2
import random
import cPickle
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from scipy.stats import pearsonr
from scipy.ndimage.filters import gaussian_filter
from params import params
import time
import params as p
from skimage.io import imread, imsave
from scipy.misc import imresize
import subprocess
import constants
#sys.path.append(constants.caffe_root + "/python")
#from caffe.proto import caffe_pb2
#import caffe
from pycocotools.coco import COCO
from pycocotools.mask import frPyObjects, decode, iou
from objectGraph import myGt, poisson
#import caffe_arch_templates.detection as detection
from sklearn.cluster import SpectralClustering

cmap = mpl.cm.Reds
dbt = "sqlite"

def visualize_coocur(co=None):
    '''
    Possibly avoid loading the co-occurence if this function is being called from somewhere with that data.
    '''
    if co == None:
        co = cPickle.load(open('co.pkl','rb'))
    sigmas = [0,1,3]
    correlations = pd.DataFrame(columns=['catpair1','catpair2','sigma','corr','pvalue'])
    numskip,numtotal = 0,0
    for key in co.keys():
        catpair1,catpair2 = key
        bnames = [params.root('results/occur/{}_{}_{}.png'.format(catpair1,catpair2,sigma)) for sigma in sigmas]
        if all([os.path.exists(bname) for bname in bnames]):
            continue
        for i,sigma in enumerate(sigmas):
            smooth = gaussian_filter(co[key],sigma)
            fig,ax=plt.subplots()
            sns.heatmap(smooth,cmap=cmap,cbar=True,vmin=0,ax=ax,xticklabels=False,yticklabels=False)
            ax.hlines(smooth.shape[0]/2,0,smooth.shape[1])
            ax.vlines(smooth.shape[1]/2,0,smooth.shape[0])
            plt.savefig(bnames[i])
            plt.close() 
            for key2 in co.keys():
                smooth2 = gaussian_filter(co[key2],sigma)
                overlap = np.sum(smooth.flatten() * smooth2.flatten())
                print(overlap)
                if overlap < 5 and numtotal > 0:
                    print("Skipping {},{}. Proportion of pairs skipped: {}".format(key,key2,numskip/float(numtotal)))
                    numskip += 1
                numtotal += 1
                corr,pvalue = pearsonr(smooth,smooth2)
                correlations.loc[len(correlations)] = [key,key2,sigma,corr,pvalue]
        correlations.to_pickle('paircorr.pkl')

def scene_cluster(occur_mat,row_labels):
    if occur_mat.shape[0] > len(row_labels):
        occur_mat = occur_mat[0:len(row_labels),0:len(row_labels)] # accidentally had zero block at end of co-occurence.
    nclusts = [3,4,5,6,7]
    clusts = {k : {} for k in nclusts}
    for num_clusters in nclusts:
        clusts[num_clusters]['meta'] = {}
        clusts[num_clusters]['split'] = {}
        clusts[num_clusters]['split']['known'] = []
        clusts[num_clusters]['split']['unknown'] = []
        clust = SpectralClustering(n_clusters=num_clusters)
        pred = clust.fit_predict(occur_mat)
        for clusterid in np.unique(pred):
            matching_cats = np.array(row_labels)[np.where(pred == clusterid)[0]]
            known = set(random.sample(list(matching_cats),len(matching_cats)/2))
            unknown = set(matching_cats) - known
            clusts[num_clusters]['meta'][clusterid] = matching_cats
            clusts[num_clusters]['split']['known'].append(known)
            clusts[num_clusters]['split']['unknown'].append(unknown)
    cPickle.dump(clusts,open('clust_splits.pkl','wb'))

def normalized_coocur(co_occurence):
    frequencies = np.sum(co_occurence,axis=1)
    co_occurence = (co_occurence.T / frequencies).T 
    occur_map = sns.heatmap(co_occurence,cmap=cmap,cbar=True,vmin=0,xticklabels=False,yticklabels=False)
    plt.savefig(params.root('results/normalized_co_occurence.png'))
    plt.close()

def add_class(tmpdirname,classifier_t,num_images,splitid):
    #caffe.set_device(2)
    meanfile = params.root("cnn/{}/val_mean_{}.npy".format(classifier_t,splitid))
    #elif classifer_t == "vggnet":
        #meanfile = params.root("cnn/vggnet/mean.npy")
        #if not os.path.exists(meanfile):
        #    meanpix = np.array([[[103.939, 116.779, 123.68]]]) #assuming BGR.
        #    np.save(meanfile,np.repeat(np.repeat(meanpix,224,0),224,1))
        #assert(False), "unknown classifier {}".format(classifier_t)
    caffe.set_mode_gpu()
    testiter = constants.cnn_snapshot_period
    while os.path.exists(params.root("cnn/{}/model_train_{}_iter_{}.caffemodel".format(classifier_t,splitid,testiter))):
        testiter += constants.cnn_snapshot_period
    print("testiter: " + str(testiter))
    net = caffe.Net(params.root("cnn/{}/deploy_{}.prototxt".format(classifier_t,splitid)),
                    params.root("cnn/{}/model_train_{}_iter_{}.caffemodel".format(classifier_t,splitid,testiter - constants.cnn_snapshot_period)),
                    caffe.TEST)
    net.blobs['data'].reshape(num_images,3,constants.cnn_w[classifier_t],constants.cnn_h[classifier_t]) #setting the batch size.
    if not os.path.exists(meanfile):
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(params.root("cnn/{}/val_mean_{}.binaryproto".format(classifier_t,splitid)), 'rb' ).read()
        blob.ParseFromString(data)
        arr = np.array(caffe.io.blobproto_to_array(blob))
        np.save(meanfile, arr[0])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # According to https://gist.github.com/ksimonyan/211839e770f7b538e2d8, do BGR ordering with vggnet. This is also true with alexnet.
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data',np.load(meanfile))
    transformer.set_raw_scale('data',255)
    transformer.set_channel_swap('data',(2,1,0))
    for can_counter in range(num_images):
        # This is already assumed to be of proper shape (Alexnet wants 200x200, vgg wants 224 x 224)
        net.blobs['data'].data[can_counter] = transformer.preprocess('data',caffe.io.load_image(tmpdirname + "/" + str(can_counter) + ".jpg"))
    net.forward()
    outname = tmpdirname + '/distr.npy'
    subprocess.call(['rm',outname])
    np.save(outname, net.blobs['prob'].data)

def write_embedding(tmpdir,num_images):
    '''
    num_images excludes 'row.jpg', which will always be there.
    '''
    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = caffe.Net(params.root("cnn/siamese_deploy.prototxt"),
                    params.root("cnn/siamese_iter_97.caffemodel"),
                    caffe.TEST)
    def prepare(imgname):
        img = imresize(imread(tmpdir + "/" + imgname, as_grey=True), (constants.cnn_w, constants.cnn_h))
        return img.reshape((1,constants.cnn_w,constants.cnn_h))
    net.blobs['data'].reshape(num_images+1,1,constants.cnn_w,constants.cnn_h) #setting the batch size.
    raw_data = np.zeros((num_images + 1, 1, constants.cnn_w, constants.cnn_h))
    raw_data[0] = prepare("row.jpg")
    for i in range(num_images):
        raw_data[i+1] = prepare("{}.jpg".format(str(i)))
    caffe_in = raw_data * 0.00390625
    out = net.forward_all(data=caffe_in)
    np.save(tmpdir + '/embedding.npy', out['feat'])
    print("wrote embedding.")

def mksiamese():
    train = np.loadtxt(params.root("cnn/labels_train"), delimiter=" ",dtype=np.bytes_)
    val = np.loadtxt(params.root("cnn/labels_val"), delimiter=" ",dtype=np.bytes_)
    traindb = leveldb.LevelDB(params.root("cnn/siamese_traindb"))
    valdb = leveldb.LevelDB(params.root("cnn/siamese_valdb"))
    train_counter = 0
    for imgName, label in train:
        imgName, label = imgName.decode("utf-8"), label.decode("utf-8")
        partnerids = [random.randint(0, train.shape[0] - 1) for i in range(constants.num_siamese_pairs)]
        partners = [train[partnerid] for partnerid in partnerids]
        if train_counter % 200 == 0:
            print("{} of the way done".format(train_counter / float(train.shape[0])))
        img = imread(imgName,as_grey=True)
        img = imresize(img,(constants.siamese_w, constants.siamese_h))
        for partner, partner_label in partners:
            partner, partner_label = partner.decode("utf-8"), partner_label.decode("utf-8")
            partner_img = imread(partner,as_grey=True)
            partner_img = imresize(partner_img,(constants.siamese_w, constants.siamese_h))
            datum = caffe_pb2.Datum()
            datum.channels = 2
            datum.height = img.shape[0]
            datum.width = img.shape[1]
            if (img.dtype == partner_img.dtype == np.dtype('uint8')):
                datum.data = img.tostring() + partner_img.tostring()
            elif img.dtype == partner_img.dtype == np.dtype('float32'):
                pass
            else:
                print("dtypes did not match")
                sys.exit(1)
            if label == partner_label:
                datum.label = 1
            else:
                datum.label = 0
            datum_str = datum.SerializeToString()
            traindb.Put(str(train_counter), datum_str)
            train_counter += 1
    val_counter = 0
    for imgName, label in val:
        imgName, label = imgName.decode("utf-8"), label.decode("utf-8")
        partnerids = [random.randint(0, val.shape[0] - 1) for i in range(constants.num_siamese_pairs)]
        partners = [val[partnerid] for partnerid in partnerids]
        if val_counter % 200 == 0:
            print("{} of the way done".format(val_counter / float(val.shape[0])))
        img = imread(imgName,as_grey=True)
        img = imresize(img,(constants.siamese_w, constants.siamese_h))
        for partner, partner_label in partners:
            partner, partner_label = partner.decode("utf-8"), partner_label.decode("utf-8")
            partner_img = imread(partner,as_grey=True)
            partner_img = imresize(partner_img,(constants.siamese_w, constants.siamese_h))
            datum = caffe_pb2.Datum()
            datum.channels = 2
            datum.height = img.shape[0]
            datum.width = img.shape[1]
            if (img.dtype == partner_img.dtype == np.dtype('uint8')):
                datum.data = img.tostring() + partner_img.tostring()
            elif img.dtype == partner_img.dtype == np.dtype('float32'):
                pass
            else:
                print("dtypes did not match")
                sys.exit(1)
            if label == partner_label:
                datum.label = 1
            else:
                datum.label = 0
            datum_str = datum.SerializeToString()
            valdb.Put(str(val_counter), datum_str)
            val_counter += 1

def save_RCNN_candidates(thresh):
    '''
    This didn't get fully developed.
    '''
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    im = cv2.imread(im_file)
    img = imread( )
    scores, boxes = im_detect(net, im)
    cls_ind = 1 #indicating foreground.
    foreground_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)] #what is the 4?
    foreground_scores = scores[:, cls_ind]
    cls_ind = 0 #indicating background.
    background_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)] #what is the 4?
    background_scores = scores[:, cls_ind]
    for response in range(boxes.shape[0]):
        if foreground_scores[response] > thresh:
            candidate = img[foreground_boxes[response]]
            imsave( )
    for response in range(boxes.shape[0]):
        if background_scores[response] > thresh:
            candidate = img[background_boxes[response]]
            imsave( )

def cocoPosNeg( ):
    '''
    Makes caffe database with yes/no to isobject.
    Everything made iwth cocoExtractPatches and no border is positive.
    '''
    if not os.path.exists(params.root("train_patches")):
        cocoExtractPatches("train",False)
    annfile = params.root("annotations/instances_{}2014.json".format("train"))
    coco = COCO(annfile)
    catIds = coco.getCatIds()
    catNames = [cat['name'] for cat in coco.loadCats(catIds)]
    annIds = coco.getAnnIds(catIds=catIds) #is there a cleaner way to request 'all'?
    anns = coco.loadAnns(annIds)
    subprocess.call(["mkdir",params.root("cnn")])
    subprocess.call(["mkdir",params.root("train_negative")])
    current_imgid = None
    timeout = 1000 #very high number, hopefully we have as many negative examples.
    with open(params.root("cnn/labels_detect_binary")):
        # I should re-write this with pandas groupby.
        for image_id, df in pd.DataFrame(anns).groupby('image_id'):
            imgname = "{}/COCO_{}2014_".format(params.root(t + "_images"),t) + "0" * (12 - len(imgid)) + imgid + ".jpg"
            print(imgname)
            img = imread(imgname)
            w,h = img.shape[0],img.shape[1]
            for rowid,row in df.iterrows():
                bb = row['bbox']
                x,y,dx,dy = bb
                img[y:(y+dy),x:(x+dx)] = -1 #shouldn't be any negative pixels, so this placeholder can be used to mark annotated regions.
            neg = 0
            for t in range(timeout):
                if len(negatives) == len(annotations):
                    break
                xpos,ypos = random.randrange(w - 40),random.randrange(h - 40)
                w,h = random.randrange(200), random.randrange(200)
                if np.where(img[ypos:(ypos+h),xpos:(xpos+w)] == -1)[0].size > 0:
                    continue
                fname = "_".join(neg,imgid)
                imsave(parmas.root("train_negative/{}".format(fname)),img[ypos:(ypos+h),xpos:(xpos+w)])
                neg += 1

def expand_candidate(bbox,imgshape,decr=0.02,timeout=2):
    '''
    Returns a new bbox with extra surrounding image data to incorperate context information.
    If this box would leave bounds of image, simply use the original bounding box.
    If trying to expand takes longer than 1 seconds, 
    '''
    t0 = time.time()
    miny,maxy,minx,maxx = bbox
    dy,dx = maxy-miny,maxx-minx
    expand_y = constants.relative_expand_max * dy
    expand_x = constants.relative_expand_max * dx
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
        if time.time() - t0 > timeout:
            return [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])],(0,0)
    return expanded,(expand_y,expand_x)
   
def cocoNumObj( ):
    for t in ["train","val"]:
        annfile = params.root("annotations/instances_{}2014.json".format(t))
        coco = COCO(annfile)
        catIds = coco.getCatIds()
        catNames = [cat['name'] for cat in coco.loadCats(catIds)]
        annIds = coco.getAnnIds(catIds=catIds) #is there a cleaner way to request 'all'?
        anns = coco.loadAnns(annIds)
        
        dosql(" ")
     
def cocoExtractPatches(t,with_border,xl=True,do_caffelabels=False,do_sql=True,adding_sql=False):
    justAll = False #if true, does not do labels for each split as well.
    assert(t == "train" or t == "val")
    if not with_border:
        assert(not xl)
    xl = "xlctx" if xl else "ctx"
    annfile = params.root("annotations/instances_{}2014.json".format(t))
    coco = COCO(annfile)
    catIds = coco.getCatIds()
    catNames = [cat['name'] for cat in coco.loadCats(catIds)]
    annIds = coco.getAnnIds(catIds=catIds) #is there a cleaner way to request 'all'?
    anns = coco.loadAnns(annIds)
    if not os.path.exists(params.root("cnn")):
        subprocess.call(["mkdir",params.root("cnn")])
    if not os.path.exists(params.root("anns_val")):
        cPickle.dump(anns,open(params.root("anns_val"),'wb'))
    if do_sql:
        insert("CREATE TABLE IF NOT EXISTS perfect_bbox (patchname TEXT, imgname TEXT, miny INT, maxy INt, minx INT, maxx INT, isexpanded INT, isxl INT, PRIMARY KEY(patchname,isexpanded,isxl))")
    existing = {}
    for i, split in enumerate(params.possible_splits):
        name = params.root("cnn/labels_{}_{}".format(t,i))
        if os.path.exists(name):
            existing[i] = np.genfromtxt(name,usecols=0,dtype=str)
        else:
            existing[i] = []
    for scene_t,split in params.illustration_splits.items():
        name = "{}/labels_{}_{}".format(params.root("cnn"),t,scene_t)
        if os.path.exists(name):
            existing[scene_t] = np.genfromtxt(name,usecols=0,dtype=str)
        else:
            existing[scene_t] = []
    expand_xs,expand_ys = [], []
    count = 0
    for seq, ann in enumerate(anns):
        count += 1
        imgid = str(ann['image_id'])
        print(seq,imgid,float(seq)/len(anns),t,with_border)
        objname = catNames[catIds.index(ann['category_id'])]
        if with_border: 
            if not os.path.exists(params.root(t + "_{}patches".format(xl))):
                subprocess.call(["mkdir", params.root(t + "_{}patches".format(xl))])
            patchname = "{}/{}_{}_{}.jpg".format(params.root(t + "_{}patches".format(xl)),imgid, objname.replace(' ','_'), seq)
        else:
            if not os.path.exists(params.root(t + "_patches")):
                subprocess.call(["mkdir", params.root(t + "_patches")])
            patchname = "{}/{}_{}_{}.jpg".format(params.root(t + "_patches"),imgid, objname.replace(' ','_'), seq)
        if adding_sql:
            if len(readsql("SELECT * FROM perfect_bbox WHERE patchname = '{}' AND isxl = {}".format(patchname,int(xl == "xlctx")))) > 0:
                continue
        if not os.path.exists(patchname) or adding_sql:
            imgname = "{}/COCO_{}2014_".format(params.root(t + "_images"),t) + "0" * (12 - len(imgid)) + imgid + ".jpg"
            img = imread(imgname)
            w,h = img.shape[0],img.shape[1]
            bb = ann['bbox']
            x,y,dx,dy = bb
            if do_sql:
                insert("INSERT INTO perfect_bbox VALUES ('{}','{}',{},{},{},{},{},{})".format(patchname,imgname,y,y+dy,x,x+dx,0,int(xl == "xlctx")),logfile='/data_b/aseewald/wal2.sql')
            if with_border:
                (ymin,ymax,xmin,xmax), (expand_y,expand_x) = expand_candidate([y,y+dy,x,x+dx],img.shape)
                xmin,xmax,ymin,ymax = int(math.ceil(xmin)),int(math.ceil(xmax)),int(math.ceil(ymin)),int(math.ceil(ymax))
                if do_sql:
                    insert("INSERT INTO perfect_bbox VALUES ('{}','{}',{},{},{},{},{},{})".format(patchname,imgname,ymin,ymax,xmin,xmax,1,int(xl == "xlctx")),logfile='/data_b/aseewald/wal2.sql')
                expand_ys.append(expand_y)
                expand_xs.append(expand_x)
            else:
                xmin,xmax = x,x+dx
                ymin,ymax = y,y+dy
                xmin,xmax,ymin,ymax = int(math.ceil(xmin)),int(math.ceil(xmax)),int(math.ceil(ymin)),int(math.ceil(ymax))
            if adding_sql:
                continue
            patch = img[ymin:ymax,xmin:xmax]
            if (patch.shape[0] < 2) or (patch.shape[1] < 2):
                continue
            imsave(patchname,patch)
            if count % 100 == 0:
                print("Average amount of y expansion: {}".format(np.mean(expand_ys)))
                print("Average amount of x expansion: {}".format(np.mean(expand_xs)))
                print("{} of the way done".format(count/len(anns)))
            if not do_caffelabels:
                continue
            labelfile = open("{}/labels_{}".format(params.root("cnn"), t),"a")
            labelfile.write("{} {}\n".format(patchname, ann['category_id']))
        for i, split in enumerate(params.possible_splits):
            if objname in split['known'] and patchname not in existing[i]:
                local_id = split['known'].index(objname)
                name = "{}/labels_{}_{}".format(params.root("cnn"), t, str(i)) if not with_border else "{}/labels_ctx_{}_{}".format(params.root("cnn"), t, str(i)) 
                with open(name,"a") as labelfile:
                    labelfile.write("{} {}\n".format(patchname, local_id))
        for scene_t,split in params.illustration_splits.items():
            if objname in split['known'] and patchname not in existing[scene_t]:
                local_id = split['known'].index(objname)
                name = "{}/labels_{}_{}".format(params.root("cnn"), t, str(scene_t)) if not with_border else "{}/labels_ctx_{}_{}".format(params.root("cnn"), t, str(scene_t))
                with open(name,"a") as labelfile:
                    labelfile.write("{} {}\n".format(patchname,local_id))

def train_detector( ):
    with open(params.root("cnn/detector."),'w') as detect:
        detect.write()
    

def motivation( ):
    '''
    Computes the average distance of a pixel to the nearest ground truth in the coco dataset.
    The non-uniformity of this information is motivation for this project.
    '''
    d = '/data/aseewald/COCO/val_images'
    imgnames = os.listdir(d)
    s = 200
    nearestImgs = np.zeros((s,s,len(imgnames)))
    centroidImg = np.zeros((s,s))
    annfile = params.root("annotations/instances_val2014.json")
    coco = COCO(annfile)
    catIds = coco.getCatIds()
    catNames = [cat['name'] for cat in coco.loadCats(catIds)]
    cats = coco.loadCats(catIds)
    co_occurence = np.zeros((np.max(catIds) + 1,np.max(catIds) + 1)) #catids seem to be non-contiguous, so just make extra space and 
    if not os.path.exists('co.pkl'):
        relative_locations = {catpair : np.zeros((2 * s,2 * s)) for catpair in itertools.product(catNames,catNames)}
        for k, imgname in enumerate(imgnames):
            print(imgname)
            img = imread(d + '/' + imgname)
            imgid = int(imgname.split('_')[2][:-4])
            annIds = coco.getAnnIds(imgIds=imgid, catIds=catIds)
            # co-occurence
            anns = coco.loadAnns(annIds)
            pairs = itertools.product(anns,anns)
            for (ann1,ann2) in pairs:
                if ann1 == ann2: # do not include (self,self) in the co-occurence counting.
                    continue
                objname1, objname2 = catNames[catIds.index(ann1['category_id'])], catNames[catIds.index(ann2['category_id'])]
                center1 = [math.floor(ann1['bbox'][1] + ann1['bbox'][0]) / 2, math.floor((ann1['bbox'][3] + ann1['bbox'][2]) / 2)]
                center1 = [center1[0] * (s / float(img.shape[0])), center1[1] * (s / float(img.shape[1]))]
                center2 = [math.floor(ann2['bbox'][1] + ann2['bbox'][0]) / 2, math.floor((ann2['bbox'][3] + ann2['bbox'][2]) / 2)]
                center2 = [center2[0] * (s / float(img.shape[0])), center2[1] * (s / float(img.shape[1]))]
                # to fix slight rounding/resize problems...
                if center1[0] >= s:
                    center1[0] = s - 1
                if center1[1] >= s:
                    center1[1] = s - 1
                if center2[0] >= s:
                    center2[0] = s - 1
                if center2[1] >= s:
                    center2[1] = s - 1
                relcenter = (int(center1[0] - center2[0] + s), int(center1[1] - center2[1] + s)) #+s so that the coordinate system is not zero-centered, but zero-bordered.
                co_occurence[catIds.index(ann1['category_id']),catIds.index(ann2['category_id'])] += 1
                relative_locations[objname1,objname2][relcenter[0],relcenter[1]] += 1
            # plot relative locations of '<cat>' relative to '<cat>', after I find the most associated cats.
            #  center stuff.
            centers = []
            for ann in anns:
                center = [math.floor(ann['bbox'][1] + ann['bbox'][0]) / 2, math.floor((ann['bbox'][3] + ann['bbox'][2]) / 2)]
                center = [center[0] * (s / float(img.shape[0])), center[1] * (s / float(img.shape[1]))]
                if center[0] >= s:
                    center[0] = s - 1
                if center[1] >= s:
                    center[1] = s - 1
                centers.append(center)
                centroidImg[center] += 1
            if centers == []:
                continue
            for i in range(0,s):
                for j in range(0,s):
                    nearestImgs[i,j,k] = np.min(np.sum(np.abs(np.array(centers) - [i,j]), axis=1))
        cPickle.dump(relative_locations, open('co.pkl','wb'))
        occur_map = sns.heatmap(co_occurence,cmap=cmap,cbar=True,vmin=0,xticklabels=False,yticklabels=False)
        plt.savefig(params.root('results/co_occurence.png'))
        plt.close()
        cPickle.dump(co_occurence,open('co_mat.pkl','wb'))
        nearestImg = np.mean(nearestImgs,axis=2)
        sns.heatmap(nearestImg,cmap=cmap,cbar=True,xticklabels=False,yticklabels=False,vmin=0)
        plt.savefig(params.root('results/nearestImgheat.png'))
        plt.close()
        sns.heatmap(centroidImg,cmap=cmap,cbar=True,xticklabels=False,yticklabels=False,vmin=0)
        plt.savefig(params.root('results/centroidImgheat.png'),vmin=0)
        plt.close()
    else:
        print("loading pickle...")
        relative_locations = cPickle.load(open('co.pkl','rb'))
        print("pickled loaded")
    for catpair in relative_locations.keys():
        occur = relative_locations[catpair]
        fig,ax = plt.subplots()
        sns.heatmap(occur,cmap=cmap,cbar=True,vmin=0,ax=ax,xticklabels=False,yticklabels=False)
        ax.hlines(occur.shape[0]/2,0,occur.shape[1])
        ax.vlines(occur.shape[1]/2,0,occur.shape[0])
        plt.savefig(params.root('results/occur/{}.{}.{}.png'.format(catpair[0],catpair[1],0)))
        plt.close()
        fig,ax = plt.subplots()
        sns.heatmap(gaussian_filter(relative_locations[catpair],sigma=2),cmap=cmap,cbar=True,vmin=0,ax=ax,xticklabels=False,yticklabels=False)
        ax.hlines(occur.shape[0]/2,0,occur.shape[1])
        ax.vlines(occur.shape[1]/2,0,occur.shape[0])
        plt.savefig(params.root('results/occur/{}.{}.3.png'.format(catpair[0],catpair[1],2)))
        plt.close()
        print("Finished ",str(catpair))
    return relative_locations

def dataset_info():
    insert("CREATE TABLE IF NOT EXISTS numobjects (imgname TEXT, isval INT, num INT)")
    for d in [params.root('val_images'),params.root('train_images')]:
        imgnames = os.listdir(d)
        annfile = params.root("annotations/instances_val2014.json")
        coco = COCO(annfile)
        catIds = coco.getCatIds()
        catNames = [cat['name'] for cat in coco.loadCats(catIds)]
        cats = coco.loadCats(catIds)
        for k, imgname in enumerate(imgnames):
            print(imgname)
            imgid = int(imgname.split('_')[2][:-4])
            annIds = coco.getAnnIds(imgIds=imgid, catIds=catIds)
            anns = coco.loadAnns(annIds)
            insert("INSERT INTO numobjects VALUES('{}',{},{})".format(imgname,int(d == params.root('val_images')),len(anns)))
        
def point_in_poly(x,y,poly):
    n = len(poly)
    inside = False
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def readsql(query_stmt,whichdb="default",sync_on_difference=True):
    assert(whichdb in ["default","sqlite","postgres"])
    if "sqlite" in constants.dbtypes and (whichdb in ["sqlite","default"]):
        # make a readonly connection with some weird syntax.
        conn = sqlite3.connect(params.read_db,timeout=300)
        liteout = pd.read_sql(query_stmt,conn)
        conn.close()
        if (whichdb != "default") or len(constants.dbtypes) == 1: return liteout
    if "postgres" in constants.dbtypes and (whichdb in ["postgres","default"]):
        conn = psycopg2.connect(**params.pg)
        pgout = pd.read_sql(query_stmt,conn)
        conn.close()
        if whichdb != "default" or len(constants.dbtypes) == 1: return pgout
    if whichdb == "default" and len(constants.dbtypes) > 1:
        if not liteout.equals(pgout):
            print("Warning dbs do not match on query {}".format(query_stmt))
            if sync_on_difference:
                print("Syncing dbs now")
                syncdbs()
            return pgout
    else:
        return pgout
    
def insert(istr,logfile=None):
    '''
    Can't use utils.dosql because that relies on python3. Do the more simpleminded thing here.
    '''
    if logfile != None:
        with open(logfile,'a') as f:
            f.write(istr + ";\n")
        return
    if "postgres" in constants.dbtypes:
        pgconn = psycopg2.connect(**params.pg)
        pgcursor = pgconn.cursor()
        pgcursor.execute(istr)
        pgconn.commit()
        pgconn.close()
    if "sqlite" in constants.dbtypes:
        conn = sqlite3.connect(params.db,timeout=100)
        cursor = conn.cursor()
        cursor.execute(istr)
        conn.commit()
        conn.close()

def encode_imgname(imgname):
    '''
    For space-heavy pixgt, save gigabytes by having lower space usage.
    '''
    parts = os.path.split(imgname)
    if parts[0] == '':
        return imgname.replace('COCO_','').replace('_000000','').replace('train2014','t').replace('val2014','v')
    else:
        return os.path.join(parts[0],parts[1].replace('COCO_','').replace('_000000','').replace('train2014','t').replace('val2014','v'))

def decode_imgname(imgname):
    '''
    For space-heavy pixgt, save gigabytes by having lower space usage.
    '''
    parts = os.path.split(imgname)
    if parts[0] == '':
        return 'COCO_' + imgname.replace('t','train2014_000000').replace('v','val2014_000000')
    else:
        return os.path.join(parts[0],'COCO_' + parts[1].replace('t','train2014_000000').replace('v','val2014_000000'))

def inner(tup,as_tsv=True):
    commit_inside=False
    imgName,coco,catNames,catIds,istrain,tsv = tup
    psconn = psycopg2.connect(**params.pg)
    if len(pd.read_sql("SELECT * FROM pixgt WHERE imgname = '{}' LIMIT 1".format(encode_imgname(imgName)),psconn)) > 0:
        psconn.close()
        print("Already done with imgName={}, continuing".format(imgName))
        return
    psconn.close()
    t0 = time.time()
    print("Started {}".format(t0))
    imgid = int((imgName.split('_')[2]).split('.')[0])
    d = 'train_images' if istrain else 'val_images'
    img = imread(params.root('{}/{}.jpg'.format(d,imgName)))
    l = len(coco.imgToAnns.keys())
    if imgid not in coco.imgToAnns.keys():
        print("imgName={},imgid={} not one of the {} coco.imgToAnns.keys()".format(imgName,imgid,l))
        return
    anns = coco.imgToAnns[imgid]
    for ann in anns:
        objname = catNames[catIds.index(ann['category_id'])]
        segmentations = ann['segmentation']
        tests, label = [], []
        if type(segmentations) == dict:
            mymask = np.squeeze(decode(frPyObjects([segmentations],img.shape[0],img.shape[1])))
            tests.append(lambda x: mymask[x] == 1)
        else:
            for segmentation in segmentations:
                if type(segmentation) == list:
                    # the reshape is to get list of pairs instead of the flattened form COCO provides.
                    segmentation = np.array(segmentation)
                    tests.append(lambda x: point_in_poly(x[1],x[0],segmentation.reshape(segmentation.size / 2 ,2)))
        assert(dbt in ["sqlite","postgres"])
        todo = []
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for i, test in enumerate(tests):
                    if test((y,x)):
                        todo.append([encode_imgname(imgName),str(y),str(x),objname,str(int(istrain))])
        if as_tsv:
            tsv.writelines(['\t'.join(recs) for recs in todo])
        else:
            if commit_inside:
                if dbt == "sqlite":
                    cursor.executemany("INSERT INTO pixgt_prime VALUES(?,?,?,?,?)",todo)
                elif dbt == "postgres":
                    cursor.executemany("INSERT INTO pixgt_prime VALUES(%s,%s,%s,%s,%s)",todo )
                conn.commit()
                print("Finished {}".format(time.time() - t0),imgName)
            else:
                print("Finished {}".format(time.time() - t0),imgName)
                return todo
    
def dense_gt(multi=False,as_tsv=True):
    insert("CREATE TABLE IF NOT EXISTS pixgt(imgname TEXT,y INT,x INT,category TEXT,istrain INT)")
    insert("CREATE TABLE IF NOT EXISTS pixgt_prime(imgname TEXT,y INT,x INT,category TEXT,istrain INT)")
    tsv = open('/fast-data/aseewald/pixgt_{}.tsv'.format(time.time()),'a')
    for annfile in reversed([params.root("annotations/instances_val2014.json"),params.root("annotations/instances_train2014.json")]):
    #for annfile in [params.root("annotations/instances_val2014.json"),params.root("annotations/instances_train2014.json")]:
        t = 0
        istrain = annfile == params.root("annotations/instances_train2014.json")
        coco = COCO(annfile)
        catIds = coco.getCatIds()
        catNames = [cat['name'] for cat in coco.loadCats(catIds)]
        imgnames = params.train_names() if istrain else params.val_names()
        try:
            done = pd.read_csv("/data/aseewald/imgnames.csv")
        except:
            done = []
        random.shuffle(imgnames)
        if dbt == "sqlite":
            conn = sqlite3.connect('moregt.db',timeout=500)
        elif dbt == "postgres":
            conn = psycopg2.connect(**params.pg)
        cursor = conn.cursor()
        if multi:
            pool = multiprocessing.Pool(1)
            pool.map(inner,reversed([(imgname,coco,catNames,catIds,istrain,tsv) for imgname in imgnames]))
        else:
            todo = []
            count = 0
            for imgname in imgnames:
                if encode_imgname(imgname) not in done:
                    new = inner((imgname,coco,catNames,catIds,istrain,tsv))
                    if new is None:
                        continue
                    if not as_tsv:
                        todo = todo + new
                        if count % 20 == 19:
                            if dbt == "sqlite":
                                cursor.executemany("INSERT INTO pixgt_prime VALUES(?,?,?,?,?)",todo)
                            elif dbt == "postgres":
                                cursor.executemany("INSERT INTO pixgt_prime VALUES(%s,%s,%s,%s,%s)",todo)
                            todo = []
    if len(todo) > 0: #this is a moment where it is silly python does not have macros.
        if dbt == "sqlite":
            cursor.executemany("INSERT INTO pixgt_prime VALUES(?,?,?,?,?)",todo)
        elif dbt == "postgres":
            cursor.executemany("INSERT INTO pixgt_prime VALUES(%s,%s,%s,%s,%s)",todo)
    t0 = time.time()
    #insert("CREATE INDEX piximgname ON pixgt(imgname)")
    print("Making indexes took {} seconds".format(time.time() - t0))

def add_gt_coco( ):
    '''
    Inserts to ground_truth table given the 
    '''
    annfile = params.root("annotations/instances_val2014.json")
    coco = COCO(annfile)
    catIds = coco.getCatIds()
    catNames = [cat['name'] for cat in coco.loadCats(catIds)]
    centroids = pd.read_sql("SELECT imgname, canid, y, x FROM candidate_centroid",sqlite3.connect(params.db))
    for imgName in params.val_names():
        print(imgName)
        if not os.path.exists(params.root('val_candidates/{}_0_{}.pkl.gz'.format(imgName,params.candidate_method))):
            print("Skipping {}, does not exist".format(imgName))
            continue
        imgid = int((imgName.split('_')[2]).split('.')[0])
        img = imread(params.root('val_images/{}.jpg'.format(imgName)))
        if imgid not in coco.imgToAnns.keys():
            continue
        anns = coco.imgToAnns[imgid]
        for ann in anns:
            objname = catNames[catIds.index(ann['category_id'])]
            segmentations = ann['segmentation']
            tests, label = [], []
            if type(segmentations) == dict:
                mymask = np.squeeze(decode(frPyObjects([segmentations],img.shape[0],img.shape[1])))
                tests.append(lambda x: mymask[x] == 1)
            else:
                for segmentation in segmentations:
                    if type(segmentation) == list:
                        # the reshape is to get list of pairs instead of the flattened form COCO provides.
                        segmentation = np.array(segmentation)
                        tests.append(lambda x: point_in_poly(x[1],x[0],segmentation.reshape(segmentation.size / 2 ,2)))
            canid = 0
            coords = centroids[centroids['imgname'] == imgName]
            numcans = len(coords)
            for canid in range(numcans):
                for i, test in enumerate(tests):
                    coord = coords[ coords['canid'] == canid]
                    if coord['y'].values.size > 0 and coord['x'].values.size > 0:
                        if test((coord['y'].values[0],coord['x'].values[0])):
                            insert("INSERT INTO ground_truth VALUES('{}',{},'{}')".format(imgName,canid,objname))
                canid += 1

# def cocoIntersect(point, segmentation, imgshape):
#     gt = frPyObjects([segmentation], imgshape[0], imgshape[1])
#     mymask = decode(gt)
#     pt = frPyObjects([point], imgshape[0], imgshape[1])
#     return iou(gt,pt) != 0 


def initialize(num_angles,k_context):
    return (360 / num_angles) * np.ones(num_angles,k_context)
