import sys
from arch_visualize import visualize_img
import pickle
import random
import os

currents = [
'imgwise_DRAW3-dual-shared_0_0_7_30_False.pkl',
]

currents = random.sample(currents,int(0.01 * len(currents)))

for current in currents:
    _,nick,tstep,_,_,trial,which = current.split('_')
    fname = '/data/aseewald/COCO/featmaps/' + current
    if os.path.exists(os.path.splitext(fname)[0] + '.jpg') or os.path.exists(os.path.splitext(fname)[0] + '.png'):
        continue
    y = pickle.load(open(fname,'rb'))
    try:
        visualize_img(*y)
    except:
        print("blah")
