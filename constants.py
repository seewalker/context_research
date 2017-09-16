'''
Alex Seewald
global parameters of the experiments which do not change.
'''
from collections import OrderedDict
import numpy as np

__author__ = "Alex Seewald"

# You may want to use just one or the other to avoid installing a database.
#dbtypes = ["sqlite","postgres"]
dbtypes = ["postgres"]

memcache_server = 'ultrasaurus.soic.indiana.edu'

assert("sqlite" in dbtypes or "postgres" in dbtypes)

# Hyperparameters of Appearance Based Features.
num_pyramid_clusters = 300
num_texton_clusters = 400
num_texton_responses = 20
max_keypoints = 8000 # ORB or SURF will not produce greater than this number of keypoints.
color_bins = 23 # for LAB color histograms.

# Hyperparameters related to Object Candidates
superpixel_amount = 50
num_objectness_boxes = 100

# if border is higher, segments are excluded if they have fewer than this many superpixels as a border.
border = 0
neighborhood_size = 20

# prop_val is only relevant for datasets without predefined train/val splits, like VOC2008.
prop_val = 0.4 #40% for testing, 60% for training.
prop_for_codebook = 0.25
prop_diagram = 0.0

# Proportion of added width or height to bounding boxes to get context.
relative_expand_max = 3.0
frame_expand_max = 0.2

weights_maxiter = 30

# object graph descriptor constants.
og_num_scales = 2
og_k = 4
og_timeout = 120
og_branching = 2
og_smoothing_std = 0.3
og_angle_sample = 10 #degrees

dtheta = 10
dtheta_partitioning = 45
num_context_clusters = 5

# for arch
negprop = 0.5
block_conv_w = 5

# for the paper.
hue_order,palette,papernames = zip(*[('DRAW4-dual','r','3+candidate dual non-shared'),\
                                     ('DRAW4-contrastive-shared','g','3+candidate'),\
                                     ('DRAW4-dual-shared-biasonly','#f7923e','3+candidate weird bias only'),\
                                     ('DRAW4-dual-shared-fixedbiasonly','b','3+candidate bias only'),\
                                     ('DRAW4-shared-fixedbiasonly','b','3+candidate bias only'),\
                                     ('DRAW4-nocenter','purple','3+no candidate'),\
                                     ('DRAW4-shared-attentiononly','salmon','training only attention'),\
                                     ('DRAW4dual-pascal','darkgreen','pascal'),\
                                     ('DRAW4-fixed-shared','#745620',''),\
                                     ('DRAW4-dual-shared-early','#185379','3+candidate k=2'),
                                     ('DRAW4-noctx','#003185','Siamese only (low resolution)'),\
                                     ('DRAW4-dual-shared','#418035','3+candidate dual'),\
                                     ('DRAW4-dual-shared-keep-stopgrad','#582422','3+candidate dual full resolution (grad stopped)'),\
                                     ('DRAW4-dual-shared-stopgrad','#b734fa','3+candidate dual (grad stopped)'),\
                                     ('DRAW5-nocenter','palegreen','4+no candidate'),\
                                     ('DRAW5-shared-nocenter','palegreen','4+no candidate'),\
                                     ('DRAW5-dual-shared','tan','4+candidate dual'),\
                                     ('DRAW3-dual-shared','#927510','2+candidate'),\
                                     # baselines
                                     ('conv-block-blur','#e6b5ea','M x M blur'),\
                                     ('above-below','#d57391','siamese above below'),\
                                     ('vanilla-vggnet','gold','appearance only'),\
                                     ('vanilla-vgg','gold','appearance only'),\
                                     ('random','lightgray','random network'),\
                                     ('None','orange','None'),\
                                     # greedy
                                     ('above_below','magenta','object graph'),\
                                     ('vanilla-embed','teal','[5] style embedding'),\
                                     ('embed','teal','[5] style embedding'),\
                                     ('left_right','k','horizontal object-graph'),\
                                     ('arb','sage','random receptive fields'),\
                                     ('euclidean','tomato','greedy without metric'),\
                                     ('metric','pink','greedy')
                                     ])

voc_neutral = -1
voc2008_labels = OrderedDict({
'aeroplane' :  ('#800000', 0),
'bicycle' :  ('#008000', 1),
'bird' :  ('#808000', 2),
'boat' :  ('#000080', 3),
'bottle' :  ('#800080', 4),
'bus' :  ('#008080', 5),
'car' :  ('#808080', 6),
'cat' :  ('#400000', 7),
'chair' :  ('#C00000', 8),
'cow' :  ('#408000', 9),
'diningtable' :  ('#C08000', 10),
'dog' :  ('#400080', 11),
'horse' :  ('#C00080', 12),
'motorbike' :  ('#408080', 13),
'person' :  ('#C08080 ', 14),
'pottedplant' :  ('#004000', 15),
'sheep' :  ('#804000', 16),
'sofa' :  ('#00C000', 17),
'train' :  ('#80C000', 18),
'tvmonitor' :  ('#004080', 19)
})

def all_to_split(split,intlabel):
    '''
    split - a 
    intlabel - a class identifier which in the global, v.i.z non-split, scope
    '''
    for key, val in voc2008_labels.items():
        if val[1] == intlabel:
            return split['known'].index(key)

def split_to_all(split,intlabel):
    stringlabel = split['known'][intlabel]
    return voc2008_labels[stringlabel][1]
