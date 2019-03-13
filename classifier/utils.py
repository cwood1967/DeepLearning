import numpy as np
import pandas as pd
import tensorflow as tf

class Classifier:
    def __init__(self, datafile, labelsfile):

        _images = self.readmm(datafile)
        _labels = self.read_labels_mm(labelsfile, _images)
    
        self.images, self.labels = self.permute_data_and_labels(_images, _labels)
        self.set_ttv(.4, .2, .4)

    def readmm(self, datafile, w=64, nc=5):
        mm = np.memmap(datafile, dtype=np.float32)
        mm = mm.reshape((-1, w, w, nc))
        x = mm[:]
        del mm
        return x 

    def read_labels_mm(self, labelsfile, images):
        mm = np.memmap(labelsfile, dtype=np.float32)
        ns = images.shape[0]
        mm = mm.reshape((ns, -1))
        x = mm[:]
        del mm
        return x

    def permute_data_and_labels(self, data, labels):
        n = data.shape[0]
        perm = np.random.permutation(n)
        pdata = data[perm]
        plabels = labels[perm]
        return pdata, plabels

    def set_ttv(self, ptrain, ptest, pval):
        n = self.images.shape[0] 
        ntrain = int(ptrain*n)
        ntest = int(ptest*n)
        nval = n - ntrain - ntest
        self.train_images = self.images[:ntrain]
        self.test_images = self.images[ntrain:ntrain + ntest]
        self.val_images = self.images[ntrain + ntest:]

        self.train_labels = self.labels[:ntrain]
        self.test_labels = self.labels[ntrain:ntrain + ntest]
        self.val_labels = self.labels[ntrain + ntest:]

    def get_batch(self, x, y, n):
        xp, yp = self.permute_data_and_labels(x, y)
        return xp[:n], yp[:n]

    def network(self, images):
        ## just make a nice classification thing
        

datafile = '/ssd1/cjw/Data/cyto/Classifier/images.mm'
labelsfile = '/ssd1/cjw/Data/cyto/Classifier/labels.mm'
c = Classifier(datafile, labelsfile)

bx, by = c.get_batch(c.train_images, c.train_labels, 5)


