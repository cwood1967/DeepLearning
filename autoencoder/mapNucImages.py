import hashlib
import sys

print(sys.executable)

import numpy as np
from skimage.io import imread

def image_to_hash(image_array):

    d = dict()
    for i in range(image_array.shape[0]):
        h = hashlib.md5(image_array[i]).hexdigest()
        d[h] = i

    return d

def map_images(d1, d2):
    mf = dict()
    for k, v in d2.items():
        if k in d1:
            mf[v] = d1[k]

    return mf

f = '/Users/cjw/Projects/cjw/cytoAE/FishCifTiff/Fish1_5.tif'
f2 = '/Users/cjw/Projects/cjw/cytoAE/FishCifTiff/Fish1_5_Nuc.tif'

x = imread(f)
x2= imread(f2)
d1 = image_to_hash(x)
d2 = image_to_hash(x2)

mf = map_images(d1, d2)

list(mf.items())[0:20]


