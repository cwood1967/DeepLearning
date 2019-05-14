import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

## this might be funky ##
class Classifier:
    def __init__(self, datafile, labelsfile, w, nc, cnums,
                 offset=0, ow = 64, channels=[1,2,3],
                dtype=np.float32, label_offset=0):

        self.ow = ow
        self.w = w
        self.nc = nc
        self.nfilters = 8
        
        self.offset = offset
        self.channels = channels
        self.label_dtype = dtype
        self.label_offset = label_offset
        _images = self.readmm(datafile, w=ow)
        _labels = self.read_labels_mm(labelsfile, _images)
        self.images, self.labels = self.permute_data_and_labels(_images, _labels)

        self.label_nums = np.argmax(self.labels, axis=1)        

        cilist = list()
        cllist = list()
        #cnums = [0, 1, 6,  7]
        print("reducing classes")
        for i, c in enumerate(cnums):
            #print(i, c)
            cl = self.label_nums == c
            ci = self.images[cl]
            lc = np.zeros((ci.shape[0], len(cnums)), dtype=np.float32)
            ohv = np.zeros(len(cnums), dtype = np.float32)
            ohv[i] = 1
            lc[:] = ohv
            cilist.append(ci)
            cllist.append(lc)
            #print(lc)

        x_images = np.concatenate(cilist, axis=0)
        x_labels = np.concatenate(cllist, axis=0)        
        self.images, self.labels = self.permute_data_and_labels(x_images, x_labels)
        
        self.label_nums = np.argmax(self.labels, axis=1)        
        self.normalize()
        self.nclasses = self.labels.shape[1]
        self.nclusters = self.label_nums.max() + 1
        self.set_ttv(.8, .1, .1)
        #print(self.test_images.shape)

    def readmm(self, datafile, w=64, nc=5):
        mm = np.memmap(datafile, dtype=np.float32, offset=self.offset)
        mm = mm.reshape((-1, self.ow, self.ow, self.nc))
        crop0 = (self.ow - self.w)//2
        crop1 = (crop0 + self.w)
        if self.channels == -1:
            x = mm[:,crop0:crop1, crop0:crop1, :]
            x# = np.expand_dims(x, -1)
        else:
            x = mm[:,crop0:crop1, crop0:crop1, self.channels]
        del mm
        return x 

    def read_labels_mm(self, labelsfile, images):
        mm = np.memmap(labelsfile, dtype=self.label_dtype, offset=self.label_offset)
        ns = images.shape[0]
        mm = mm.reshape((ns, -1))
        x = mm[:]
        del mm

        return x

    def normalize(self, type=0):
        xm = self.images.mean(axis=(1,2), keepdims=True)
        sm = self.images.std(axis=(1,2), keepdims=True)
        self.images = (self.images - xm)/sm

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
        
        self.train_images = self.images[:ntrain]
        self.test_images = self.images[ntrain:ntrain + ntest]
        self.val_images = self.images[ntrain + ntest:]

        self.train_labels = self.labels[:ntrain]
        self.test_labels = self.labels[ntrain:ntrain + ntest]
        self.val_labels = self.labels[ntrain + ntest:]

        self.train_label_nums = self.label_nums[:ntrain]
        self.test_label_nums = self.label_nums[ntrain:ntrain + ntest]
        self.val_label_nums = self.label_nums[ntrain + ntest:]
        self.make_class_where()

    def make_class_where(self):

        self.class_where_train = list()
        self.class_where_test = list()
        self.class_where_val = list()
        
        for i in range(self.nclasses):
            wi = np.where(self.train_label_nums == i)
            self.class_where_train.append(wi)

        for i in range(self.nclasses):
            wi = np.where(self.test_label_nums == i)
            self.class_where_test.append(wi)

        for i in range(self.nclasses):
            wi = np.where(self.val_label_nums == i)
            self.class_where_val.append(wi)
            
            
            

    def balanced_set(self, x, y, yn,nsamples, cluster_num):
        # x is images
        # y is labels
        # yn is the label_nums

        # get the indices where label is cluster_num
        
        w = yn[cluster_num]
        #w = np.where(yn == cluster_num)
        # shuffle the contents of w
        # where returns a tuple, so need to index it to get the array
        np.random.shuffle(w[0])
        # get the first nsamples of nr
        nr = w[0][:nsamples]
        bx = x[nr]
        nrot  = np.random.randint(0, 4)
        bx = np.rot90(bx, nrot, axes=(1,2))
        by = y[nr]
        return bx, by
        

    def get_balanced_batch(self, x, y, yn, n):
        
        d = n//self.nclusters
        m = n % self.nclusters

        image_list = list()
        label_list = list()
        r = np.random.randint(0, self.nclusters)
        for i in range(self.nclusters):
            s = d
            if i == r:
                s += m
            abx, aby = self.balanced_set(x, y, yn, s, i)
            image_list.append(abx)
            label_list.append(aby)

        bx = np.concatenate(image_list, axis=0)
        by = np.concatenate(label_list, axis=0)

        kx = np.arange(bx.shape[0])
        np.random.shuffle(kx)
        bx = bx[kx]
        bx += .2*np.random.rand()
        by = by[kx]
        return bx, by
                                  
    def get_batch(self, x, y, n):
        xp, yp = self.permute_data_and_labels(x, y)
        xp = xp[:n]
        yp = yp[:n]
        xp += .2*np.random.standard_normal(size=xp.shape)
        return xp, yp



    def get_regularizer(self, scale=1.):
        return tf.contrib.layers.l2_regularizer(scale)
    
    def dnet_block(self, x, nf, k, drate, is_training=True, droprate=0):
        
        h = tf.layers.conv2d(x, nf, k, strides=1,
                             padding='same', dilation_rate=drate,
                             kernel_initializer=None,
                             kernel_regularizer=self.get_regularizer(),
                             activation=None)

        
        h = tf.nn.leaky_relu(h)

        if is_training:
            h = tf.keras.layers.SpatialDropout2D(rate=droprate).apply(h)
        

        h = tf.concat([x, h], -1)
        
        h1 = tf.layers.conv2d(h, nf, k, strides=1,
                             padding='same', dilation_rate=drate,
                             kernel_initializer=None,
                             use_bias=False,
                             kernel_regularizer=self.get_regularizer(),
                             activation=None)

        h1 = tf.nn.leaky_relu(h1)
        if is_training:
            h1 = tf.keras.layers.SpatialDropout2D(rate=droprate).apply(h1)

        h = tf.concat([h, h1], -1)

        h = tf.layers.conv2d(h, nf, k, strides=2,
                             padding='same', dilation_rate=drate,
                             kernel_initializer=None,
                             kernel_regularizer=self.get_regularizer(),
                             use_bias=False,
                             activation=None)

        if is_training:
            h = tf.keras.layers.SpatialDropout2D(rate=droprate).apply(h)     
        h = tf.nn.leaky_relu(h)

        return h

    def create_network(self, batch, is_training=True, droprate=0):
        ## just make a nice classification thing
        layers = list()
        layers.append(batch)
        
        h = self.dnet_block(batch, 8, 3, 1, droprate=droprate)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat1')

        h = self.dnet_block(h, 16, 3, 1, droprate=droprate)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat2')

        h = self.dnet_block(h, 32, 3, 1)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat4')

        #h = self.dnet_block(h, 128, 3, 1)
        #layers.append(h)
        #h = tf.concat(layers, -1, name='concat8')
        print(h)
        h = tf.layers.flatten(h)
#         h = tf.layers.dense(h, 500,
#                             kernel_regularizer=self.get_regularizer())
#         h = tf.nn.leaky_relu(h)

#         h = tf.layers.dense(h, 100,
#                             kernel_regularizer=self.get_regularizer())
        
#         h = tf.nn.leaky_relu(h)
        
        h = tf.layers.dense(h, self.nclasses ,
                            kernel_initializer=tf.constant_initializer(value=0.0),
                            bias_initializer=tf.constant_initializer(value=1./self.nclasses),
                             kernel_regularizer=self.get_regularizer())        
        
        self.logits = h
        self.softmax = tf.nn.softmax(h)

    def create_loss(self, labels, l2f=0):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=labels)
        loss = tf.reduce_mean(loss, axis=(-1))
        l2_loss = tf.losses.get_regularization_loss()
        #print("loss before reduction", loss)
        self.loss = tf.reduce_mean(loss) + l2f*l2_loss
        self.l2_loss = l2_loss
        #print(self.loss)

    def create_opt(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                     name='adam_opt').minimize(self.loss)

    def create_placeholders(self):
        sizeC = self.images.shape[-1]
        self.image_batch = tf.placeholder(tf.float32, shape=(None, self.w, self.w, sizeC))
        self.label_batch = tf.placeholder(tf.float32, shape=(None, self.nclasses))
        self.learning_rate = tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, shape=())
        
    def create_accuracy(self, y, y_):
        cpred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(cpred, tf.float32))
        p = tf.argmax(y, 1)
        p_ = tf.argmax(y_, 1)
        #print("##########", p, p_)
        _, self.accuracy_score = tf.metrics.accuracy(p_, p)
        _, self.precision = tf.metrics.precision(p_, p)
        _, self.recall = tf.metrics.recall(p_, p)        
        self.confmat = tf.math.confusion_matrix(p_, p)
                                                    
    def train(self, n_iter=10000, learning_rate=0.001, droprate=0, l2f=0):
        tf.reset_default_graph()
        self.create_placeholders()
        self.create_network(self.image_batch, is_training=self.is_training, droprate=droprate)
        self.create_loss(self.label_batch, l2f=l2f)
        self.create_accuracy(self.softmax, self.label_batch)
        self.create_opt()
        print("***********************")


        sess = tf.Session()
        self.sess = sess
        xinit = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(xinit)
        #sess.run(tf.global_variables_initializer())
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('accuracy_score', self.accuracy_score)
        tf.summary.scalar('precision', self.precision)
        tf.summary.scalar('recall', self.recall)                 
 #       tf.summary.scalar('zl2 loss', self.l2_loss)
        tf.summary.histogram('logits', self.softmax)
        tf.summary.histogram('clusters', tf.argmax(self.softmax, 1))
        tf.summary.histogram('truth', tf.argmax(self.label_batch, 1))
        tf.summary.image('image', self.image_batch)
        #print(tf.reduce_mean(self.softmax, axis=-1))
        #print(tf.reduce_mean(self.softmax, axis=0))        

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test')
        
        for i in range(n_iter):
            if i % 100 == 0:
                learning_rate -= .0002*learning_rate
                #print('learning rate set to ', learning_rate)
            if i >= 0:
                bx, by = self.get_balanced_batch(self.train_images,
                                             self.train_labels,
                                             self.class_where_train, 128)

            _, xl = sess.run([self.opt, self.loss],
                             feed_dict={self.image_batch:bx, self.label_batch:by,
                                        self.learning_rate:learning_rate,
                                        self.is_training:True})
            
            
            if i % 100 == 0:
                tb, tl = self.get_balanced_batch(self.test_images,
                                                 self.test_labels,
                                                 self.class_where_test,
                                                 128)
                vl, _, _, vcm = sess.run([self.loss, self.softmax, self.label_batch, self.confmat],
                              feed_dict={self.image_batch:self.test_images,
                                         self.label_batch:self.test_labels,
                                         self.is_training:False})

                
                summary = sess.run(merged, feed_dict={self.image_batch:bx, self.label_batch:by,
                                                       self.is_training:False})
                
                test_summary = sess.run(merged, feed_dict={self.image_batch:tb,
                                                           self.label_batch:tl, self.is_training:False})
            if i % 100 == 0:
                train_writer.add_summary(summary, i)
                test_writer.add_summary(test_summary, i)                
            if i % 1000 == 0:
                print(i, xl, vl)

        ''' run the final test'''
        tb, tl = self.get_balanced_batch(self.val_images,
                                         self.val_labels,
                                         self.class_where_test,
                                         1024)

        vl, vsm, vlb, vcm = sess.run([self.loss, self.softmax, self.label_batch, self.confmat],
                              feed_dict={self.image_batch:tb,
                                         self.label_batch:tl,
                                         self.is_training:False})

        print(vcm)
'''###### end of Classifier #######'''

def test_err(x):
    print(x)
    
def get_classifier(datafile, labelsfile, w, nc, cc, offset=0, ow=65,
                   channels=[1,2,3], dtype=np.float32, label_offset=0):
    
    c = Classifier(datafile, labelsfile, w, nc, cc, offset=offset,
                   ow=ow, channels=channels, dtype=dtype,
                   label_offset=label_offset)
    return c

if __name__ == '__main__':
    if sys.platform == 'darwin':
        datapre = '/Users/cjw/'
    else:
        datapre = ''

    parser = argparse.ArgumentParser(description='Train the classifier.')
    parser.add_argument('--cc', type=str)
    args = parser.parse_args()
    cc = np.array(args.cc.split(','), dtype=np.int32)
    print("Train on these", cc)

    datafile = datapre + 'Data/cc_images.mm'
    labelsfile = datapre + 'Data/cc_labels.mm'
    c = Classifier(datafile, labelsfile, 32, 5, cc)
    c.train(n_iter=10000, learning_rate=0.0008)
