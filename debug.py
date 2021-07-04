# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import socket
import socks
import allensdk
import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
import pprint
from tqdm import tqdm
from scipy import ndimage as nd
import imageio as io
import json
import os
import datetime
import pycocotools
from pycocotools.mask import encode
import pycocotools.coco as coco
from pycocotools.coco import COCO
import imantics
import cv2


# %%
my_cache_dir = '/media/seeker/sda2_1TB/nwb_data'
bc = bpc.VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=my_cache_dir)
experiment_table = bc.get_ophys_experiment_table()
with open('output.txt', 'wt') as out:                       
    print(experiment_table.index,file=out)
type(experiment_table.index)
    


# %%



# %%
dataset = imantics.Dataset('allen_342')
for i in tqdm(range(342)):
    ophys_experiment_id = experiment_table.index[i]
    #print('loading the {} file'.format(i))
    image=imantics.Image.from_path('/home/seeker/Swin-Transformer-Object-Detection/image/image_{}.png'.format(ophys_experiment_id))
    mask_array =cv2.imread('/home/seeker/Swin-Transformer-Object-Detection/mask/mask_{}.png'.format(ophys_experiment_id),cv2.IMREAD_GRAYSCALE)
    mask = imantics.Mask(mask_array)
    ann=imantics.Annotation.from_mask(mask,image,imantics.Category('cell'))
    dataset.add(ann)


# %%
json.dumps(ann.coco())


# %%
for i in tqdm(range(342)):
    ophys_experiment_id = experiment_table.index[i]
    dataset = bc.get_behavior_ophys_experiment(ophys_experiment_id)
    plt.imsave('../convert/image/image{}.png'.format(i),dataset.max_projection)


