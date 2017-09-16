# Introduction

This is the software that is necessary to produce the results published in ( ) and in ( ). The data-handling and evaluation aspects of each paper are fairly similar, but are not currently sufficiently abstract to be a python module themselves, so they are combined in this repository.

For an interactive visualization of the embeddings and clusterings in (), see (link)
For an interactive visualization of the embeddings and clusterings in (), see (link)

# Software Dependencies

Many of the dependencies are contained in `dependencies` directory. Other things you will need are:
* a postgres installation, and optionally sqlite3. Tested with postgres 9.5 and 9.6.
* python 3 and python 2.7, along with all libraries being imported by this code.
* tensorflow for python 3. Tested with version 0.11
* Matlab. Tested on r2015a.
* Opencv, if you want to make object candidates rather than using pre-computed ones.

If using greedy approach, a modified version of the lsml.py of metric_learn is needed. Simply overwrite lsml.py in the module's source with ./lsml_custom.py. This is a hack, and if it doesn't work with some new version of metric_learn, please bug me (aseewald@indiana.edu) about it.

# Hardware Needs

* Training the end-to-end systems is computation, not I/O bound, so it helps to have a GPU with as much memory and FLOPS as possible.
* Training the fully convolutional networks for greedy learning takes more time on I/O than computation reading in ground truth from the database, so if you have a large SSD it helps to put the database there.
* If doing greedy learning, it will help to have lots of RAM because we would like to keep big matrices in memory.

# How to Use

We provide some pre-computed data to start with.

In order to run this on an arbitrary dataset of unknown structure, it's a lot of work to get the data in the right form, and the code here is not currently automated to the point where you can press a "demo" or "go" button and process all the data and run the experiments. Parts of the work are captured in dataproc.py, and other parts have been done interactively.

Batchsizes and such system-dependent parameters are hard-coded a few places in the code, which may not work with your system if you have less resources, or you may want to increase these values if you have more resources.

For any questions about this project, contact Alex Seewald at aseewald@indiana.edu. I'd be glad to answer, and honored you are trying to replicate things!

# Table of Contents

* dataproc.py - code for organizing pre-processing of the data and making the reprsentations of object candidates given trained models.
* arch.py - the program that trains the end-to-end architectures.
* arch_block.py - architectural components in the blockwise variants.
* arch_draw.py - architectural components in the DRAW-style variants.
* arch_common.py - architectural components shared among the variants.
* greedy_learning.py - 
* affinity.py - makes design matrices and affinity matrices for clustering representations of unseen object candidates.
* evaluate.py - measures the performance in terms of purity.
* fully_conv.py - Code for training fully convolutional networks to do pixelwise prediction.
* py2.py - Unfortunately, parts of the python ecosystem (such as the COCO modules) are stuck in python 2.7. This file contains the bits and pieces thatmust be run with a python 2 interpreter.
* constants.py - Contains hyperparameters and design decisions of the experiments which the whole project must refer to.
* params.py - Defines a class `Params` which keeps track of parameters to a run of an experiment, along with the locations of the data.
* web - Contains the code and data necessary to run the graph hosted at: http://madthunder.soic.indiana.edu/~aseewald/objgraph.

More extensive documentation can be found in the scripts' docstrings, which you can view by executing:
In[N] import data
In[N+1] help(data)

## Expected Structure of a Data Directory:

    train_images - everything here is a full, raw image in the training set.
    validation_images - everything here is a full, raw image in the valing set, and there is a corresponding ground truth label for the file of similar name in 'ground_truth'.
    val_images - everything here is a full, raw image in the testing wet, and there is no ground truth information available.
    ground_truth - pixelwise ground truth images for full regions. An example of suitable information is PASCAL 2008s 'SegmentationClass' direcotry.
    train_superpixels - <name>_<number>. Whereas train_superpixelation consists of matricies of pixel labels, train_superpixels consists
                        of pickled python objects which store everything a Segment needs to have associated with it.
    val_superpixels - <name>_<number>. This is done for images in gt_test_images and for images in nogt_val_images.
    train_segments - <name>_<number>, like above.
    val_segments - <name>_number, like above. This is done for images in gt_val_images and for images in nogt_val_images.
    train_patches - These are rectangular patches of an image which are used to train the convolutional neural network. There need not be pixel-wise ground truth for these,
                    there must be bounding-box-wise ground truth. These are also used to train the siamese network.
    val_patches - To see how well the CNN does.
    cnn - Models and logs are stored here.
    kernels - kernels get stored here. There is sense in reusing them.

## Profiling

The directory profiles contains line_profiler output for typical examples of usage of my code.

## The Database

To avoid recomputing all the data when needed, certain things like {bounding boxes,CNN features,centroids} of superpixels are stored in a database.
constants.py has a variable `dbtypes` which determines whether just postgres or postgres along with redundant sqlite both will be used for storage (the choice of whether to make inserts redundant is switchable at the wrapper function dosql in utils.py).

Most of the tables are created as needed by relevant parts of the code, but if a table is missing somehow, its expected definition is in ./database_schema.
