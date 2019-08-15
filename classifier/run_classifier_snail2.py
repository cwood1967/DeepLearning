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

datafile = '/home/cjw/Code/DeepLearning/classifier/Data/images.mm'
labelsfile = '/home/cjw/Code/DeepLearning/classifier/Data/labels.mm'

#cc = [0,3,7]
cc = list(range(7))

c = network.get_classifier(datafile, labelsfile, 32, 5, cc, channels=[0,2,4],
                          ow=64, combine=[[0, 8], [4, 7]])

try:
    shutil.rmtree('/scratch/cjw/logs')
    print('deleted logs')
except:
    print("couldn't delete")
    
time.sleep(4)
while os.path.exists('`logs'):
    shutil.rmtree('logs')
    time.sleep(.1)

c.train(n_iter=25000, learning_rate=0.0006, droprate=0.0, l2f=.004,
        batchsize=256)

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

all_images = c.orig_images
all_labels = c.orig_labels

xm = all_images.mean(axis=(1,2), keepdims=True)
sm = all_images.std(axis=(1,2), keepdims=True)
all_images = (all_images - xm)/sm

all_loss, all_sm, _, _ = c.sess.run([c.loss, c.softmax, c.label_batch, c.confmat],
                      feed_dict={c.image_batch:all_images,
                                 c.label_batch:all_labels,
                                 c.is_training:False})



np.set_printoptions(precision=3)
print(metrics.classification_report(all_labels.argmax(axis=-1), all_sm.argmax(axis=-1)))
print(metrics.accuracy_score(all_labels.argmax(axis=-1), all_sm.argmax(axis=-1)))
cm = metrics.confusion_matrix(all_labels.argmax(axis=-1), all_sm.argmax(axis=-1))

print(cm)

np.save('Data/all_pickle.pkl', all_sm)




