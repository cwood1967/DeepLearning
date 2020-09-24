# %%
import os
import glob
import pickle
import numpy as np
import tifffile
from matplotlib import pyplot as plt
import pandas as pd
import shutil
# %%

topdir = "/Users/cjw/Dropbox/phago/"
idirs = glob.glob(topdir + "Snail_Phago*")
rdict = dict()
for d in idirs:
    print(f"{d}/*.tif")
    imagefiles = sorted(glob.glob(f"{d}/*.tif"))
    k0 = os.path.basename(d)
    ks = k0.split("_")
    k = ks[0] + ks[-1]
    rdict[k] = imagefiles
    print(k, len(imagefiles))    

# %%

csvs = glob.glob(f"{topdir}*.csv")
csvs
# %%
clist = list()
for c in csvs:
    _df = pd.read_csv(c, usecols=[0,1,2,3])
    k = os.path.basename(c).split('_')[0]
    _df['replicate'] = k
    clist.append(_df)

df = pd.concat(clist)
df.columns = ['ClusterID', 'EventID', 'FileName', 'index_in_file', 'replicate']
# %%

# %%

df430 = df[df.ClusterID == 27430].reset_index()
df442 = df[df.ClusterID == 27442].reset_index()
df430.shape, df442.shape
# %%
df442
# %%

def lookup(index, imagelist):
    strindex = str(index).join(["_", "_"])
    res =[z for z in imagelist if strindex in z]
    return res
# %%
rg = rdict['Snail4']
print(rg[0])
%timeit lookup(5, rg)
# %%
lookup(657, rg)
# %%

p = "Data/27442"
if not os.path.exists(p):
    os.makedirs(p)

index_dict = dict()
i = 0
for row in df442.itertuples():
    idx = row.index_in_file
    files = lookup(idx,rdict[row.replicate])
    if len(files) != 3:
        print(len(files), row)
        continue
    index_dict[i] = idx
    i += 1
    for xf in files:
        rep = row.replicate
        pp = f"{p}/{rep}"
        if not os.path.exists(pp):
            os.mkdir(pp)
        shutil.copy2(xf, pp)

# %%
index_dict
# %%

#with open("Data/27442/27442_Snail2_index.pkl", 'rb') as pkl:
#    index_dict = pickle.load(pkl)
index_dict
# %%
mh = np.memmap('Data/27430/27430_Snail2.mm', mode='r', shape=(4,), dtype=np.int32)
mhs = tuple(mh)
del mh
mm = np.memmap('Data/27430/27430_Snail2.mm', mode='r', shape=mhs,
               dtype=np.float32, offset=128)



# %%
i = np.random.randint(0, len(mm))
sa = lookup(index_dict[i], rdict['Snail2'])
plt.subplot(2,1,1)
plt.imshow(mm[i,:,:,0])
plt.subplot(2,1,2)
plt.imshow(tifffile.imread(sa[0]))
i
# %%
