'''
Alex Seewald 2016
aseewald@indiana.edu

Implementation of Object Graph, and closely related ideas.
'''
from __future__ import division
import math
import copy
import os
import tempfile
import subprocess
import time
import functools
import random
import heapq
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.io import imread, imsave
from scipy.spatial.distance import euclidean
from scipy.misc import imresize
from scipy.stats import entropy
import scipy.io
from hyperparams import chisquared
import constants

__author__ = "Alex Seewald"

sns.set_style("white")

def partition(p,l):
    return functools.reduce(lambda x, y: x[not p(y)].append(y) or x, l, ([], []))

def pool(method,vec):
    if method == 'mean':
        return [ [np.mean(vec[0:(i+1),j]) for j in range(vec.shape[1])] for i in range(vec.shape[0])]
    if method == 'max':
        return [ [np.max(vec[0:(i+1),j]) for j in range(vec.shape[1])] for i in range(vec.shape[0])]
    elif method == 'none':
        return vec


def mkuniform(group,segment,number):
    "Initially, no group should be made to have 0 size, however."
    number = math.floor(number)
    if len(group) == number:
        return group
    elif len(group) == 0:
        return [segment for i in range(number)]
    elif len(group) < number:
        return group + [random.choice(group) for i in range(number - len(group))]
    elif len(group) > number:
        return group[0:number]
def on_border(segment,superpixels):
    above, below = partition(lambda neighbor: neighbor.centroid[1] > segment.centroid[1], superpixels)
    q1 = partition(lambda neighbor: neighbor.centroid[0] > segment.centroid[0], above)
    q2 = partition(lambda neighbor: neighbor.centroid[0] < segment.centroid[0], above)
    q3 = partition(lambda neighbor: neighbor.centroid[0] < segment.centroid[0], below)
    q4 = partition(lambda neighbor: neighbor.centroid[0] > segment.centroid[0], below)
    return (len(q1) < constants.border or len(q2) < constants.border or len(q3) < constants.border or len(q4) < constants.border)
def above_below(segment,neighbors):
    above, below = partition(lambda neighbor: neighbor.centroid[1] > segment.centroid[1], neighbors)
    above, below = above[0:min(constants.neighborhood_size,len(above))], below[0:min(constants.neighborhood_size,len(below))]
    above, below = mkuniform(above,segment,constants.neighborhood_size), mkuniform(below,segment,constants.neighborhood_size)
    return np.array([neighbor.object_distribution for neighbor in above]), np.array([neighbor.object_distribution for neighbor in below])
def quadrants(segment, neighbors):
    above, below = partition(lambda neighbor: neighbor.centroid[1] > segment.centroid[1], neighbors)
    q1 = [sp for sp in above if sp.centroid[0] > segment.centroid[0]]
    q2 = [sp for sp in above if sp.centroid[0] < segment.centroid[0]]
    q3 = [sp for sp in below if sp.centroid[0] < segment.centroid[0]]
    q4 = [sp for sp in below if sp.centroid[0] > segment.centroid[0]]
    q1, q2, q3, q4 = mkuniform(q1,segment,constants.neighborhood_size / 2), mkuniform(q2,segment,constants.neighborhood_size / 2), mkuniform(q3,segment,constants.neighborhood_size / 2), mkuniform(q4,segment,constants.neighborhood_size / 2), 
    return np.array([neighbor.object_distribution for neighbor in q1]), np.array([neighbor.object_distribution for neighbor in q2]), np.array([neighbor.object_distribution for neighbor in q3]), np.array([neighbor.object_distribution for neighbor in q4]), 
def left_right(segment, neighbors):
    right, left = partition(lambda neighbor: neighbor.centroid[0] > segment.centroid[0], neighbors)
    right, left = right[0:min(constants.neighborhood_size,len(right))], left[0:min(constants.neighborhood_size,len(left))]
    right, left = mkuniform(right,segment,constants.neighborhood_size), mkuniform(left,segment,constants.neighborhood_size)
    return np.array([neighbor.object_distribution for neighbor in right]), np.array([neighbor.object_distribution for neighbor in left])
def bag(segment,neighbors):
    return [np.array([neighbor.object_distribution for neighbor in neighbors])]
def randomly(segment, neighbors):
    if len(neighbors) == 0:
        neighbors = [segment for i in range(2 * constants.neighborhood_size)]
    elif len(neighbors) < (2 * constants.neighborhood_size):
        neighbors = neighbors + random.sample(neighbors, 2 * constants.neighborhood_size - len(neighbors))
    random.shuffle(neighbors)
    groupA, groupB = [], []
    for i, neighbor in enumerate(neighbors):
        if i % 2 == 0:
            groupA.append(neighbor)
        else:
            groupB.append(neighbor)
    return np.array([neighbor.object_distribution for neighbor in groupA]), np.array([neighbor.object_distribution for neighbor in groupB])

def domain_adjust(angle):
    if angle < 0:
        return math.pi + (math.pi + angle)
    else:
        return angle

def separate(segment,superpixels,separators,distinguish):
    partitions = [ [] for i in range(len(separators))]
    stuff = [ [] for i in range(len(separators))]
    angled_sps = [ (sp, complex(sp.centroid[0] - segment.centroid[0], sp.centroid[1] - segment.centroid[1])) for sp in superpixels]
    angled_sps.sort(key = lambda x: domain_adjust(np.angle(x[1])))
    seps = [ (sp, complex(sp.centroid[0] - segment.centroid[0], sp.centroid[1] - segment.centroid[1])) for sp in separators]
    seps.sort(key = lambda x: domain_adjust(np.angle(x[1])))
    starting = angled_sps.index(seps[0])
    partitions[0].append(seps[0][0])
    stuff[0].append(seps[0][0].color_histogram)
    partition_id = 0
    for sp in (angled_sps[(starting+1):] + angled_sps[0:starting]):
        if sp in seps:
            partition_id += 1
        stuff[partition_id].append(sp[0].color_histogram)
        if (not distinguish) or (entropy(sp[0].object_distribution) <= 1.7) or sp in seps:
            partitions[partition_id].append(sp[0])
    if distinguish:
        return partitions, stuff
    else:
        return partitions

def evenly_distribute(separators,superpixels):
    distribution = [math.floor(len(superpixels) / len(separators)) for i in range(len(separators))]
    for j in range(len(superpixels) - np.sum(distribution)):
        distribution[j] = distribution[j] + 1
    return distribution

def obj_cost(separators,segment,superpixels):
    '''
    underlying hypothesis - it is good to split around a low-entropy segment.
    '''
    object_entropy = np.sum([entropy(sep.object_distribution) for sep in separators])
    max_object_entropy = len(separators) * entropy( (1.0 / segment.object_distribution.size) * np.ones_like(segment.object_distribution))
    return (object_entropy / max_object_entropy)

def grouping_cost(separators, segment, superpixels):
    '''
    underlying hypothesis - it is better to split in more-closely-even partitions.
    '''
    grouped = separate(segment,superpixels,separators,False)
    grouping_entropy = entropy([len(group) for group in grouped])
    max_grouping_entropy = entropy(evenly_distribute(separators,superpixels))
    return (max_grouping_entropy / grouping_entropy) #can this be learned?

def saliency_cost(separators, segment, superpixels):
    '''
    try to maximize saliency.
    '''
    grouped = separate(segment,superpixels,separators, False)
    saliency_entropy = entropy([np.sum([sp.saliency_count for sp in group]) for group in grouped])
    total_saliency = np.sum([sp.saliency_count for sp in superpixels])
    max_saliency_entropy = entropy((total_saliency / len(grouped)) * np.ones(len(grouped)))
    return (max_saliency_entropy / saliency_entropy)

def cost(separators,segment,superpixels):
    '''
    Idea - rather than mixing the cost ideas in the program, different object-graph-descriptor vectors could be defined

    '''
    return grouping_cost(separators, segment, superpixels) + saliency_cost(separators, segment, superpixels)

def search(initialization,segment,superpixels,costfn):
    '''
    Might be best to precompute distances between all superpixels.
    My paper could remark that there is no triangle-inequality-like property to object distributions.
    '''
    t = 0
    cost_0 = costfn(initialization, segment, superpixels)
    queue, visited = [(cost_0, initialization.tolist())], []
    while t < constants.og_timeout:
        if len(queue) == 0:
            break
        cost_t, candidate = heapq.heappop(queue)
        heapq.heappush(visited,(cost_t, candidate))
        index = random.randint(0,len(candidate) - 1)
        changer = candidate[index]
        for neighbor in heapq.nsmallest(constants.og_branching+1,superpixels, key=lambda x: euclidean(x.centroid, changer.centroid))[1:]:
            if neighbor in candidate:
                continue
            newcomer = candidate[0:index] + [neighbor] + candidate[(index+1):]
            cost_new = cost(newcomer,segment,superpixels)
            if (cost_new, newcomer) not in visited:
                heapq.heappush(queue,(cost_new, newcomer))
        t += 1
    cost_f, answer = heapq.heappop(visited)
    answer.sort(key=lambda sp: domain_adjust(np.angle(complex(sp.centroid[0] - segment.centroid[0], sp.centroid[1] - segment.centroid[1]))))
    return answer

def equi_angle_initialize(segment,superpixels):
    angles = np.array([domain_adjust(np.angle(complex(sp.centroid[0] - segment.centroid[0], sp.centroid[1] - segment.centroid[1]))) for sp in superpixels])
    ideal_angles = np.linspace(0,2 * math.pi,constants.og_k)
    nearest = np.zeros(constants.og_k, dtype=np.int)
    for i, angle in enumerate(ideal_angles):
        nearest[i] = np.argmin(np.abs(angles - angle))
    if random.random() < 0.0:
        print("Equi init diff: " + str(np.mean(np.abs(ideal_angles - angles[nearest]))))
        print(str(ideal_angles), str(angles[nearest]))
    return nearest

def visualize(segment,superpixels,field,separators,scale,hyperparams):
    '''

    '''
    tstamp = time.process_time()
    base = imread(hyperparams.root("val_images/{}.jpg").format(segment.imgName))
    for sp in superpixels:
        if sp in field:
            color = [0,0,0]
        else:
            color = [255,255,255]
        if not hasattr(sp,'edge_pixels'):
            break
        for edgepix in sp.edge_pixels:
            base[edgepix] = color
    plt.imshow(base)
    # Still confused about the possible transposing going on.
    plt.gca().add_patch(plt.Circle((int(segment.centroid[1]), int(segment.centroid[0])), 10, color='black'))
    plt.gca().add_patch(plt.Circle((int(segment.centroid[1]), int(segment.centroid[0])), 3, color='green'))
    for separator in separators:
        plt.arrow(int(segment.centroid[1]), int(segment.centroid[0]), int(separator.centroid[1] - segment.centroid[1]), int(separator.centroid[0] - segment.centroid[0]), color='r', head_width=7)
    plt.savefig(hyperparams.root("results/{}_{}_{}_{}.png".format(segment.imgName, str(scale), hyperparams.encode(), tstamp)))
    plt.close()
    plt.title("scale=" + str(scale))
    sns.heatmap(pd.DataFrame(data=pd.DataFrame(thing_vec[scale], columns=pd.Index(constants.voc2008_set3['known'], name='known classes'), index=pd.Index([str(i) for i in range(constants.og_k)], name='radial regions counterclockwise from origin'))))
    plt.savefig(hyperparams.root("results/heat{}_{}_{}_{}.png".format(segment.imgName, str(scale), hyperparams.encode(), tstamp)))
    plt.close()

def variable_field(distinguish, cost_type, segment, superpixels,hyperparams):
    '''
    if distinguish is true: do things and stuff separately.
    scales can mean incorperating a certain proportions of all superpixels.

    try to minimize area entropy and minimize.
        '''
    if cost_type == "all":
        costfn = cost
    elif cost_type == "nosal":
        costfn = grouping_cost
    elif cost_type == "nospuniform":
        costfn = saliency_cost
    thing_vec = np.zeros((constants.og_num_scales, constants.og_k, segment.object_distribution.size))
    if distinguish:
        stuff_vec = np.zeros((constants.og_num_scales, constants.og_k, 3 * constants.color_bins))
    for scale in range(constants.og_num_scales):
        field = superpixels[0:math.floor(len(superpixels) * ((scale+1) / constants.og_num_scales))]
        field.sort(key=lambda x: entropy(x.object_distribution))
        field = np.array(field)
        #initialization = field[0:constants.og_k]
        initialization = field[equi_angle_initialize(segment, field)]
        separators = search(initialization, segment, field, costfn)
        if random.random() < 0.01:
            init_angles = [domain_adjust(np.angle(complex(sp.centroid[0] - segment.centroid[0], sp.centroid[1] - segment.centroid[1]))) for sp in initialization]
            final_angles = [domain_adjust(np.angle(complex(sp.centroid[0] - segment.centroid[0], sp.centroid[1] - segment.centroid[1]))) for sp in separators]
            with open(hyperparams.root("results/init_vs_ideal.txt"), 'a') as log:
                log.write(str(np.mean(np.abs(np.array(init_angles) - np.array(final_angles)))))
        if distinguish:
            thing_partitions, stuff_partitions = separate(segment, field, separators, distinguish)
            for i, partition in enumerate(thing_partitions): # fill in nearby data if a receptive field is empty.
                if len(partition) == 0:
                    thing_partitions[i] = [segment]
                    stuff_partitions[i] = [segment.color_histogram]
            thing_vec[scale,:,:] = [np.mean([sp.object_distribution for sp in partition],axis=0) for partition in thing_partitions]
            stuff_vec[scale,:,:] = [np.mean(partition, axis=0) for partition in stuff_partitions]
        else:
            thing_partitions = separate(segment, field, separators, distinguish)
            for i, partition in enumerate(thing_partitions): # fill in nearby data if a receptive field is empty.
                if len(partition) == 0:
                    thing_partitions[i] = [segment]
            thing_vec[scale,:,:] = [np.mean([sp.object_distribution for sp in partition],axis=0) for partition in thing_partitions]
        if random.random() < constants.prop_diagram:
            visualize(segment,superpixels,field,separators,scale)
    if distinguish:
        return np.concatenate((thing_vec,constants.stuff_scale * stuff_vec), axis=2), separators
    else:
        return thing_vec, separators

def is_variable(ogfunction):
    "For some reason, functools does not seem to support function equality, so this is a hack"
    return (ogfunction != above_below) and (ogfunction != left_right) and (ogfunction != randomly) and (ogfunction != quadrants) and (ogfunction != bag)

def object_graph(ogfunction,segment,superpixels,include_appearance,pooltypes,visualize=False):
    '''
    '''
    superpixels.sort(key=(lambda x: euclidean(x.centroid, segment.centroid)))
    ogvectors = {}
    regions = ogfunction(segment,superpixels)
    if is_variable(ogfunction):
        regions, separators = regions
    for pooltype in pooltypes:
        ogvector = np.concatenate([pool(pooltype,region) for region in regions])
        if include_appearance:
            distr = segment.object_distribution
            if ogvector.shape[1] != (distr.size):
                ogvector = np.vstack([np.concatenate((segment.object_distribution, constants.stuff_scale * segment.color_histogram)), ogvector])
            else:
                ogvector = np.vstack([distr, ogvector])
        ogvectors[pooltype] = ogvector
    if is_variable(ogfunction):
        return ogvectors, [sep.centroid for sep in separators]
    else:
        return ogvectors

def myGt(mask):
    '''
    What I will train the CNN with. This isn't quite the label though, it is a mask of random walk values at pixels.
    This is useful to store because
    '''

    # This should maybe be modified to only apply random walk within object areas, and place
    # zeros in other locations.
    walkmap = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            walkmap[i,j] = randomWalk((i,j),mask)
    return walkmap

def climatlab(argstr):
    return subprocess.call(["matlab","-nojvm","-nodisplay","-nosplash","-r",argstr + ";quit"])

def poisson(mask,hyperparams):
    '''
    Makes a XYT space of just one time size.
    Why involve a CNN? Because the test images don't have the bounding boxes!
    '''
    volume = np.array([mask])
    subprocess.call(['rm','current_mat.mat', 'current_poisson.mat'])
    scipy.io.savemat('current_mat.mat',dict(x=volume,y=hyperparams.hs))
    try:
        climatlab("try, gmg; catch, disp('failed'), end")
        U = scipy.io.loadmat('current_poisson.mat')['out']
        return U
    except:
        raise(IOError)

def extract_train_angular(hyperparams):
    '''
    Iterate over files in train_field_masks, running sector_mask at random angles
    and at random locations.

    Will it be possible for me to not have testing periodically in the training process?


    Next big idea - I could look for a correlation in what the
    '''
    d = hyperparams.root("train_field_masks")
    gtmasks = os.listdir(d)
    labelfile = open(hyperparams.root("cnn/field_labels_train"),'a')
    for gtmask in gtmasks:
        gt = np.load(d + '/' + gtmask)[0] #because it stores the full volume.
        imgid = gtmask.split('_')[0]
        imgname = "{}/COCO_train2014_".format(hyperparams.root("train_images")) + "0" * (12 - len(imgid)) + imgid + ".jpg"
        print("working on " + imgname)
        img = imread(imgname)
        for i in range(constants.field_train_sample):
            border = 30
            random_x = random.randint(border, img.shape[0] - border)
            random_y = random.randint(border, img.shape[1] - border)
            random_angle = random.randint(0,360)
            random_radius = random.randint(50,500)
            # verify that random choices are within image counds.
            # here, using 'x' and 'y' in a way consistent to what I've already done, not with what makes sense.
            while (not (0 < random_x + random_radius * math.sin(math.radians(random_angle)) < img.shape[0])) or (not (0 < random_y + random_radius * math.cos(math.radians(random_angle)) < img.shape[1])):
                random_angle = random.randint(0,360)
                random_radius = random.randint(50,500)
            icpy = copy.deepcopy(img)
            gcpy = copy.deepcopy(gt)
            mask = sector_mask(gt.shape, (random_x,random_y), random_radius, (random_angle, random_angle + constants.dtheta))
            icpy[~mask] = 0
            gcpy[~mask] = 0
            xmin = min(random_x, random_x + random_radius * math.sin(math.radians(random_angle)))
            xmax = max(random_x, random_x + random_radius * math.sin(math.radians(random_angle)))
            ymin = min(random_y, random_y + random_radius * math.cos(math.radians(random_angle)))
            ymax = max(random_y, random_y + random_radius * math.cos(math.radians(random_angle)))
            icpy,gcpy = icpy[xmin:xmax,ymin:ymax], gcpy[xmin:xmax,ymin:ymax]
            # do the resizing before label generation so there is not the scale issue.
            try:
                icpy = imresize(icpy, (constants.field_w,constants.field_h))
                gcpy = imresize(gcpy, (constants.field_w,constants.field_h))
                gtval = round(np.sum(gcpy))
                slicename = "{}/{}_{}.jpg".format(hyperparams.root("train_slices"),imgid,i)
                imsave(slicename,icpy)
                labelfile.write('{} {}\n'.format(slicename, gtval))
            except ValueError:
                continue

def sector_mask(shape,center,radius,angle_range):
    """
    Coordinates I want (to be verified) is 0 degrees still means to the right (even though i'm using y and x like image coords).

    Here, angles are like in the plane I'm used to.
    """
    y,x = np.ogrid[:shape[0],:shape[1]]
    cy,cx = center
    tmin,tmax = np.deg2rad(angle_range)
    if tmax < tmin:
        tmax += 2*np.pi
    r2 = (y-cy)*(y-cy) + (x-cx)*(x-cx)
    theta = np.arctan2(y-cy,x-cx) - tmin
    theta %= (2*np.pi)
    anglemask = theta <= (tmax-tmin)
    if radius == float("inf"):
        circmask = np.ones_like(anglemask)
    else:
        circmask = r2 <= radius*radius
    return circmask*anglemask

def angular_partition(shape,center,radius,dtheta):
    assert(360 % dtheta == 0)
    return [sector_mask(shape,center,radius,(theta,theta+dtheta)) for theta in np.arange(0,360,dtheta)]

def trim(img):
    xtrim = 0
    while np.count_nonzero(img[xtrim,:]) == 0:
        xtrim += 1
    img = img[xtrim:]
    xtrim = img.shape[0] - 1
    while np.count_nonzero(img[xtrim,:]) == 0:
        xtrim -= 1
    img = img[:xtrim]
    ytrim = 0
    while np.count_nonzero(img[:,ytrim]) == 0:
        ytrim += 1
    img = img[:,ytrim:]
    ytrim = img.shape[0] - 1
    while np.count_nonzero(img[:,ytrim:]) == 0:
        ytrim -= 1
    img = img[:,:ytrim]

def evaluateAngles(img,segment,superpixels):
    '''
    Run the CNN on candidate image regions.
    Try to maximize superpixel entropy count and favorability according to the CNN.
    '''
    with tempfile.TemporaryDirectory() as tmpdirname:
        for angular_img in fieldSlices(img,segment,superpixels):
            imsave(tmpdirname + "/angle_" + str(i) + ".jpg",segment.img_patch)
        subprocess.call(["/home/aseewald/anaconda2/bin/python","eval_angle.py",tmpdirname,str(len(np.linspace(0,360,constants.og_angle_sample)))], stderr=open('err','w'))
        scores = np.load(tmpdirname + '/angles.npy')
    return scores

def fieldSlices(img,segment,superpixels):
    angular_imgs = [ ]
    for i, angle in enumerate(np.linspace(0,360,constants.og_angle_sample)):
        cpy = copy.deepcopy(img)
        local_radius = min(remaining(img,segment.centroid,angle), cutoff(img,))
        mask = sector_mask(img.shape, segment.centroid, local_radius, (angle, angle+constants.og_angle_sample))
        cpy[~mask] = 0
        cpy = trim(cpy)
        angular_imgs.append(cpy)
    return angular_imgs

def best_rotation_distance(ogvec_one, ogvec_two):
    '''
    An earlier iteration of this project involved
    '''
    seg_distance = np.sum(np.abs(ogvec_one[0] - ogvec_two[0]))
    ogvec_one, ogvec_two = ogvec_one[1:], ogvec_two[1:]
    min_ds = float("inf")
    for roll_amount in range(constants.og_k):
        ds = 0
        for scale in range(constants.og_num_scales):
            starting, ending = scale * constants.og_k, (scale+1) * constants.og_k
            ds += np.sum(np.abs(ogvec_one[starting:ending] - np.roll(ogvec_two[starting:ending],roll_amount,axis=0)))
        if ds < min_ds:
            min_ds = ds
    return seg_distance + min_ds

# this is not yet verified.
def closest_match_distance(ogvec_one, ogvec_two,sep_centroids_one,sep_centroids_two,seg_centroid_one,seg_centroid_two):
    seg_distance = np.sum(np.abs(ogvec_one[0] - ogvec_two[0]))
    match_distance = 0
    ogvec_one, ogvec_two = ogvec_one[1:], ogvec_two[1:]
    angles_one = np.array([domain_adjust(np.angle(complex(sep_centroid[0] - seg_centroid_one[0], sep_centroid[1] - seg_centroid_one[1]))) for sep_centroid in sep_centroids_one])
    angles_two = np.array([domain_adjust(np.angle(complex(sep_centroid[0] - seg_centroid_two[0], sep_centroid[1] - seg_centroid_two[1]))) for sep_centroid in sep_centroids_two])
    for i, angle in enumerate(angles_two):
        match = np.argmin(angles_one - angles_two)
        match_distance += np.sum(np.abs(ogvec_one[match] - ogvec_two[i]))
    return seg_distance + match_distance

def smoothed_distance(ogvec_one, ogvec_two):
    '''
    The smoothing happens within a scale.
    '''
    # I might need to enforce oddness of vector length somehow, so that the 'current' bin is the clear max after being smoothed.
    accumulator = 0
    smoother = signal.gaussian(M=constants.og_k,std=constants.og_smoothing_std)
    smoothed = np.zeros_like(ogvec_two)
    k,scales = constants.og_k,constants.og_num_scales
    for rowid in range(ogvec_two.shape[0]):
        for scale in range(scales):
            section = ogvec_two[rowid][k * scale:k * (scale+1)]
            smoothed = signal.convolve(section,signal.gaussian( ))
            accumulator += euclidean(ogvec_one[rowid][k * scale:k * (scale+1)],smoothed)
    return accumulator
