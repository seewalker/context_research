'''

'''
import math
import numpy as np
from scipy.spatial.distance import euclidean
from utils import *

__author__ = "Alex Seewald"

def grid_field_candidates(N:int,hyperparams) -> np.ndarray:
    '''
    these are relative distances from segment (object candidate) centroid, from -1 to 1 ('1' representing the image dimension)
    '''
    epsilon = 0.05 #smallest possible width.
    num = 0
    fields = []
    dosql("CREATE TABLE IF NOT EXISTS grid_field_candidates(ymin INT,ymax INT,xmin INT,xmax INT,id INT, trial INT, PRIMARY KEY(trial,id))",hyperparams)
    maxtrial = readsql("SELECT max(trial) FROM grid_field_candidates",hyperparams)['trial']
    if len(maxtrial) == 0:
        trial = 0
    else:
        trial = maxtrail.values[0] + 1
    while num < (10 * N):
        x_start = 1 - (2 * random.random())
        x_width = random.random()
        y_start = 1 - (2 * random.random())
        y_width = random.random()
        if (x_width > epsilon) and (y_width > epsilon) and (x_start + x_width < 1.0) and (y_start + y_width < 1.0):
            fields.append([y_start,y_start + y_width, x_start, x_start + x_width])
            num += 1
    fields = random.sample(fields,N)
    num = 0
    for y_start,y_end,x_start,x_end in fields:
        dosql(f"INSERT INTO grid_field_candidates VALUES ({y_start},{y_end},{x_start},{x_end},{num},{trial})",hyperparams)
        num += 1
    return np.array(fields)

def split_field_candidates(N:int,args,hyperparams) -> Tuple[np.ndarray,np.ndarray]:
    dosql("CREATE TABLE IF NOT EXISTS split_field_candidates(angle FLOAT,offset_y FLOAT,offset_x FLOAT,id INT, trial INT, nickname TEXT, PRIMARY KEY(trial,id,nickname))",hyperparams,whichdb="postgres")
    maxtrial = readsql(f"SELECT max(trial) FROM split_field_candidates WHERE nickname = '{args.pixnickname}'",hyperparams,whichdb="postgres").values[0][0]
    if maxtrial is None:
        trial = 0
    else:
        trial = maxtrail.values[0] + 1
    offsets = np.random.uniform(-1,1,size=(N,2))
    angles = np.random.uniform(0,2 * math.pi,size=N)
    for i in range(N):
        dosql(f"INSERT INTO split_field_candidates VALUES ({angles[i]},{offsets[i][0]},{offsets[i][1]},{i},{trial},'{args.pixnickname}')",hyperparams,whichdb="postgres")
    return (offsets, angles)

def spherical_field_candidates(N:int):
    dosql("CREATE TABLE IF NOT EXISTS depth_field_candidates(radius_min FLOAT, theta_min FLOAT, phi_min FLOAT, radius_max FLOAT, theta_max FLOAT, phi_max FLOAT, id INT, trial INT, PRIMARY KEY(trial,id))",hyperparams)
    maxtrial = readsql("SELECT max(trial) FROM grid_field_candidates",hyperparams)['trial']
    if len(maxtrial) == 0:
        trial = 0
    else:
        trial = maxtrail.values[0] + 1
    r_min = np.random.uniform(0,0.9)
    dr = np.random.uniform( )
    theta_min = np.random.uniform(0,2 * math.pi,size=N)
    dtheta = np.random.uniform(0.1,1.5 * math.pi,size=N)
    phi_min = np.random.uniform(0.1,math.pi,size=N)
    dphi = np.random.uniform(0,.75 * math.pi,size=N)
    for i in range(N):
        thetamin,thetamax = theta_min[i],theta_min[i]+dtheta[i] % (2 * math.pi)
        phimin,phimax = phi_min[i],phi_min[i]+dphi[i] % math.pi
        rmin,rmax = r_min[i],min(r_min[i] + dr[i],1.0)
        dosql(f"INSERT INTO depth_field_candidates({rmin},{thetamin},{phimin},{rmax},{thetamax},{phimax},{i},{trial})",hyperparams)

def xyd_field_candidates():
    maxtrial = readsql("SELECT max(trial) FROM grid_field_candidates",hyperparams)['trial']
    if len(maxtrial) == 0:
        trial = 0
    else:
        trial = maxtrail.values[0] + 1
    offsets = np.random.uniform(-1,1,size=(N,2))
    angles = np.random.uniform(0,2 * math.pi,size=N)
    depth_start = np.random.randn(0,0.9,size=N)
    depth_delta = np.random.randn(0.1,0.9,size=N)
    depths = np.minimum(depth_start + delta_depth,1.0)

def quadsplit_field_candidates(N=8000):
    offsets = np.random.normal(loc=0.0,scale=0.5,size=(N,2))
    angles1 = np.random.uniform(0,2 * math.pi,size=N)
    angles2 = np.random.uniform(0,2 * math.pi,size=N)
    return (offsets, angles1, angles2)
    
def grid_field_vector(candidate_centroid,sp_centroids,sp_distrs,imgshape,field_candidates) -> np.ndarray:
    '''
    does max pooling.
    '''
    h,w = imgshape
    candidate_rel = candidate_centroid['y'].values[0] / h, candidate_centroid['x'].values[0] / w
    numcat = sp_distrs[0].size
    vector = np.zeros((field_candidates.size,numcat))
    for i, field in enumerate(field_candidates):
        matching = []
        for j, sp_centroid in enumerate(sp_centroids):
            sp_rel = (sp_centroid[0] / h) - candidate_rel[0], (sp_centroid[1] / w) - candidate_rel[1]
            if (field[0] <= sp_rel[0] <= field[1]) and (field[2] <= sp_rel[1] <= field[3]):
                matching.append(sp_distrs[j])
        if matching:
            vector[i] = np.max(np.array(matching), axis=0)
        else:
            vector[i] = np.zeros(numcat)
    return vector.flatten()

def is_above(pt1:Tuple[int,int], pt2:Tuple[int,int], theta:float, epsilon=1e-7) -> bool:
    '''
    theta is in radians here.
    The image coordinate system has the origin in the top-left corner, whereas the bottom-left corner is more familiar.
    To avoid mistakes in that translation I'm working in the familiar coordiantes and "above" here does not mean 'above' is the image
    (but there is a bijection, so ultimately there is still meaning in the angles).
    '''
    r = (pt2[0] - pt1[0], pt2[1] - pt1[1])
    if r[1] == 0:
        r[1] = epsilon #prevening divide by zero errors.
    theta_r = math.atan(r[0] / r[1]) + math.pi # adding pi to put it in (0,2pi) range from (-pi,pi) range.
    above = theta > theta_r
    #aboveness.write('{}\n'.format(str(int(above))))
    return above

def split_field_vector(candidate_centroid,sp_centroids, sp_distrs,imgshape,offset_candidates,angle_candidates,scales=[1],pool_t="mean"):
    '''
    This is dense (for the sake of learning).
    does mean pooling.
    '''
    h,w = imgshape
    candidate_y, candidate_x = candidate_centroid['y'].values[0] , candidate_centroid['x'].values[0]
    assert(offset_candidates.shape[0] == angle_candidates.size)
    assert(len(sp_distrs) == sp_centroids.shape[0])
    assert(all([0 <= x <= 2 * math.pi for x in angle_candidates])) #ensure we are using radians.
    # between functions, my code is not consistent in the use of words for describing number of categories (the question being 'does None count'?).
    # but here it is what it should be.
    numcat = sp_distrs[0].size
    vector = np.zeros((angle_candidates.size,len(scales),numcat))
    num_superpixels = sp_centroids.shape[0]
    if pool_t == "mean":
        pool = np.mean
    elif pool_t == "max":
        pool = np.max
    else:
        assert(False), f"Unknown pool_t: {pool_t}"
    for i, offset_candidate in enumerate(offset_candidates):
        angle = angle_candidates[i]
        # describing the superpixels.
        relevant = list(zip(sp_centroids, sp_distrs))
        relevant.sort(key=lambda x: euclidean(x[0],offset_candidate))
        relevant_distr = np.array([x[1] for x in relevant])
        relevant_centroids = np.array([x[0] for x in relevant])
        relevant_distr_at_scale = np.array([relevant_distr[0:(math.floor(scale * len(sp_distrs)))] for scale in scales])
        relevant_centroids_at_scale = np.array([relevant_centroids[0:(math.floor(scale * len(sp_distrs)))] for scale in scales])
        for scaleid in range(len(scales)):
            above_distrs= []
            reldistrs = relevant_distr_at_scale[scaleid]
            relcenters = relevant_centroids_at_scale[scaleid]
            assert(relcenters.shape[0] == reldistrs.shape[0])
            # iterating over superpixels
            for j in range(0,relcenters.shape[0]):
                if is_above((candidate_y/h + offset_candidate[0], candidate_x/w + offset_candidate[1]), (relcenters[j][0] / h, relcenters[j][1] / w), angle):
                    above_distrs.append(reldistrs[j])
            if len(above_distrs) == 0:
                vector[i,scaleid] = np.zeros(numcat)
            else:
                vector[i,scaleid] = pool(above_distrs, axis=0)
    return vector.flatten()
