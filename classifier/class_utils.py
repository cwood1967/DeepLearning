#%%
import numpy as np

#%%
def combine_channels(c1, c2, labels):
    nc = labels.shape[1]
    v1 = np.zeros((nc,), dtype=np.float32)
    v2 = np.zeros((nc,), dtype=np.float32)

    v1[c1] = 1.
    v2[c2] = 1.
    cx = labels.argmax(axis=1)

    wx1 = np.where(cx == c1)
    wx2 = np.where(cx == c2)
    new_labels = labels.copy()
    new_labels[wx2] = v1
    new_labels = np.delete(new_labels, c2, 1)

    return new_labels

print('Here we go!!')
mm = np.memmap('Data/labels.mm', dtype=np.float32, offset=0)
mm = mm.reshape((-1, 9))
labels = mm[:]
print(labels.shape)
new_labels = combine_channels(0, 8, labels)
new_labels = combine_channels(4, 7, new_labels)
print(new_labels.shape)


#%%
