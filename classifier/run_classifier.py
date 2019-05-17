import os
try:
	os.chdir(os.path.join(os.getcwd(), 'classifier'))
	print(os.getcwd())
except:
	pass

import shutil
import sys
import time
import network

import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as metrics

datafile = 'Data/cc_images.mm'
labelsfile = 'Data/cc_6_labels.mm'

#cc = [0,3,7]
cc = [0, 1, 2, 3, 4, 5]
c = network.get_classifier(datafile, labelsfile, 32, 5, cc, channels=[0,1,3,4], ow=64)

try:
    shutil.rmtree('logs')
except:
    "couldn't delete"
    
time.sleep(4)
while os.path.exists('logs'):
    time.sleep(.1)

c.train(n_iter=40000, learning_rate=0.0006, droprate=0, l2f=.003, batchsize=128)

# run all validation images
vl, vsm, vlb, vcm = c.sess.run([c.loss, c.softmax, c.label_batch, c.confmat],
                      feed_dict={c.image_batch:c.val_images,
                                 c.label_batch:c.val_labels,
                                 c.is_training:False})


#tl.shape, np.argmax(vsm, axis=-1).shape
tls = np.argmax(c.val_labels, axis=-1)
vsms = np.argmax(vsm, axis=-1)


print(metrics.classification_report(tls, vsms))
print(metrics.accuracy_score(tls, vsms))

cm = metrics.confusion_matrix(tls, vsms)
#cm = cm/cm.sum(axis=1)
#import pandas as pd
#cmdf = pd.DataFrame(cm)
np.set_printoptions(precision=3)
print(cm)
print(cm.sum(axis=0, keepdims=True))
print(cm.sum(axis=1, keepdims=True))
print(cm.sum(axis=0).sum())
print(cm.sum(axis=1).sum())
print(cm.sum())
print()
print(metrics.confusion_matrix(tls, vsms))

all_loss, all_sm, _, _ = c.sess.run([c.loss, c.softmax, c.label_batch, c.confmat],
                      feed_dict={c.image_batch:c.images,
                                 c.label_batch:c.labels,
                                 c.is_training:False})


#%%
np.save('Data/all_pickle.pkl', all_sm)




