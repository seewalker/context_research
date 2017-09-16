'''
Some visualizations for the paper:

The experiments take awhile to run, so
'''
import matplotlib.pyplot as plt
import constants
import fnmatch

def visualize_process(imgName,ns=None):
    '''
    have the split be one with relatively few known classes, so that the annotations can be pretty.
    column 1: draw the raw image.
    column 2: raw image background. segment boundaries overlayed, annotated with probability distributions.
    column 3: same as above, but with a different segmentation.
    column 4: superpixels
    column 5: some representation of the object graph descriptor.
    '''
    if not ns:
        ns = constants.
    else:
        ns = [n]
    for n in ns:
        # find matching segments
def visualize_objgraph(imgName):
    " How to best do this?"
    pass

def purity_completeness_curve( ):
    '''
    '''
    pass

def precision_recall_curve( ):
    '''
    Varying the known theshold - 
    '''
