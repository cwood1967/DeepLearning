#%%

import sys
import os

try:
	os.chdir('/home/cjw/Code/DeepLearning/classifier')
	print(os.getcwd())
except:
	pass


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import umap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
#plt.style.use('ggplot')

style = "fivethirtyeight"
#%%
def umap_features(xdata):
    #cemb = umap.UMAP(n_neighbors=10, n_components=7, min_dist=.05).fit_transform(xdata)
    emb = umap.UMAP(n_neighbors=15, min_dist=.05).fit_transform(xdata)
    return emb



#%%
datadir = "Data/"
mmfiles = [datadir + a for a in os.listdir(datadir) if a.endswith('.mm')]
mmfiles
#%%
# hm = np.memmap(mmfiles[1], dtype=np.int32, shape=(4,))
# print(hm)

mm = np.memmap(mmfiles[0], dtype=np.float32, offset=0, shape=(-1, 64,64,5))
mm.shape
#%%
plt.imshow(mm[0,:,:,4])
print(mm[0, :,:,1].min())

#%%

mm[1:4].shape
#%%
#cpdir = "Checkpoints/all_0-1-2-3-4"
x = mm[:,:,:, [0,2,4]]
xm = x.mean(axis=(1,2), keepdims=True)
xs = x.std(axis=(1,2), keepdims=True)

x = (x - xm)/xs
x = x.reshape((-1, 64*64*3))

print(x.shape)
u = umap_features(x)

#%%
plt.figure(figsize=(8,8))
plt.scatter(u[:,0], u[:,1], s=4)

#%%
