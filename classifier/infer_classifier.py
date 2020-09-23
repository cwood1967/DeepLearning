import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics

### where is the best model checkpoint
#cpdir = 'Checkpoints/Snail_Redo_for_metrics/best-2019-08-16-10-24/'
#checkpoint = 'best-checkpoint-9200'
cpdir = 'Checkpoints/Snail_Redo_for_metrics_7_classes/best-2019-08-16-10-46/'
checkpoint = 'best-checkpoint-21300'
if not cpdir.endswith('/'):
    cpdir += '/'
    
### where are thme images and labels
sval = cpdir.split("/")
valdir = "/".join(sval[0:-2]) + "/"
val_images = np.load(valdir + 'validation_images.npy')
val_labels = np.load(valdir + 'validation_labels.npy')
val_nums = val_labels.argmax(axis=1)

### load the Image 3C clusters into a dataframe
#df = pd.read_csv('Data/Snail/ClusterIDs.csv')
#print(df.head())

### read in the fdl coordinates, not all of the cells have fdl coords
#fdl = pd.read_csv('Data/Snail/FDL_coords.csv', delimiter=';')
#fdl['Y'] = -fdl['Y']
#print(fdl.head())

### join the fdl and df for a dataframe
#res = fdl.set_index('EventID').join(df.set_index('EventID'))
#res.head()

sess = tf.Session()
#### load the best model and do tf stuff
saver = tf.train.import_meta_graph(cpdir + checkpoint + '.meta')
saver.restore(sess, cpdir + checkpoint)

softmax = sess.graph.get_tensor_by_name('Softmax:0')
images = sess.graph.get_tensor_by_name('Placeholder:0')

vsm = sess.run(softmax, feed_dict={images:val_images})
vsm_nums = vsm.argmax(axis=1)

val_accuracy = metrics.accuracy_score(val_nums, vsm_nums)
print(val_accuracy)
val_cm = metrics.confusion_matrix(val_nums, vsm_nums)

val_report = metrics.classification_report(val_nums, vsm_nums)

print(val_report)
print(val_cm)

# plt.figure(figsize=(8,8))
# sns.scatterplot(x='X', y='Y', hue='ClusterID', palette='Set1',
#                 edgecolor='none', s=12, marker='o',
#                 legend="full", data=res)
# plt.savefig('InferPlots/fdl.pdf', bbox_inches='tight')

