import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

class Classifier:
    def __init__(self, datafile, labelsfile, w, nc):

        self.w = w
        self.nc = nc
        self.nfilters = 8
        _images = self.readmm(datafile)
        _labels = self.read_labels_mm(labelsfile, _images)

    
        self.images, self.labels = self.permute_data_and_labels(_images, _labels)
        self.label_nums = np.argmax(self.labels, axis=1)
        c8 = self.label_nums == 8
        c3 = self.label_nums == 3
        ic8 = self.images[c8]
        ic3 = self.images[c3]

        lc8 = np.zeros((ic8.shape[0], 2), dtype=np.float32)
        lc3 = np.zeros((ic3.shape[0], 2), dtype=np.float32)
        lc8[:] = [0, 1]
        lc3[:] = [1, 0]

        x_images = np.concatenate([ic8, ic3], axis=0)
        x_labels = np.concatenate([lc8, lc3], axis=0)        
        self.images, self.labels = self.permute_data_and_labels(x_images, x_labels)
        #print(self.label_nums)
        
        self.normalize()
        self.nclasses = self.labels.shape[1]
        self.set_ttv(.8, .1, .1)
        print(self.test_images.shape)

    def readmm(self, datafile, w=64, nc=5):
        mm = np.memmap(datafile, dtype=np.float32)
        mm = mm.reshape((-1, w, w, nc))
        x = mm[:,16:48, 16:48, [0, 2,4]]
        del mm
        return x 

    def read_labels_mm(self, labelsfile, images):
        mm = np.memmap(labelsfile, dtype=np.float32)
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
        nval = n - ntrain - ntest
        self.train_images = self.images[:ntrain]
        self.test_images = self.images[ntrain:ntrain + ntest]
        self.val_images = self.images[ntrain + ntest:]

        self.train_labels = self.labels[:ntrain]
        self.test_labels = self.labels[ntrain:ntrain + ntest]
        self.val_labels = self.labels[ntrain + ntest:]

    def get_batch(self, x, y, n):
        xp, yp = self.permute_data_and_labels(x, y)
        xp += .2*np.random.randn()
        return xp[:n], yp[:n]

    def dnet_block(self, x, nf, k, drate):
        h = tf.layers.conv2d(x, nf, k, strides=2,
                             padding='same', dilation_rate=drate,
                             kernel_initializer=None,
                             activation=None)

        h = tf.nn.relu(h)
        h = tf.layers.dropout(h, rate=.5)
        h = tf.layers.conv2d(h, nf, k, strides=1,
                             padding='same', dilation_rate=drate,
                             kernel_initializer=None,
                             activation=None)
        
        h = tf.layers.dropout(h, rate=.5)        
        h = tf.nn.relu(h)
        return h

    def create_network(self, batch):
        ## just make a nice classification thing
        layers = list()
        layers.append(batch)
        ns = 8
        h = self.dnet_block(batch, 4, 3, 1)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat1')

        h = self.dnet_block(h, 4, 3, 1)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat2')

        h = self.dnet_block(h, 8, 3, 1)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat4')

        h = self.dnet_block(h, 16, 3, 1)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat8')
        print(h)
        h = tf.layers.flatten(h)
        h = tf.layers.dense(h, 500)
        h = tf.nn.relu(h)

        h = tf.layers.dense(h, 100)
        h = tf.nn.relu(h)
        
        h = tf.layers.dense(h, self.nclasses)        
        
        self.logits = h
        self.softmax = tf.nn.softmax(h)

    def create_loss(self, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=labels)
        loss = tf.reduce_sum(loss, axis=(-1))
        print("loss before reduction", loss)
        self.loss = tf.reduce_mean(loss)
        print(self.loss)

    def create_opt(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                     name='adam_opt').minimize(self.loss)

    def create_placeholders(self):
        self.image_batch = tf.placeholder(tf.float32, shape=(None, self.w, self.w, 3))
        self.label_batch = tf.placeholder(tf.float32, shape=(None, self.nclasses))
        self.learning_rate = tf.placeholder(tf.float32, shape=())

    def create_accuracy(self, y, y_):
        cpred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(cpred, tf.float32))
        
                                                    
    def train(self, learning_rate=0.001):
        tf.reset_default_graph()
        self.create_placeholders()
        self.create_network(c.image_batch)
        self.create_loss(self.label_batch)
        self.create_accuracy(self.softmax, self.label_batch)
        self.create_opt()
        print("***********************")


        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.histogram('logits', self.softmax)
        tf.summary.histogram('clusters', tf.argmax(self.softmax, 1))
        tf.summary.histogram('truth', tf.argmax(self.label_batch, 1))
        print(tf.reduce_mean(self.softmax, axis=-1))
        print(tf.reduce_mean(self.softmax, axis=0))        
        #tf.summary.scalar('test_loss', self.loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test')
        m_writer = tf.summary.FileWriter('logs/m')
        
        for i in range(20000):
            bx, by = self.get_batch(self.train_images, self.train_labels, 256)
            _, xl, xmg = sess.run([self.opt, self.loss, merged],
                             feed_dict={self.image_batch:bx, self.label_batch:by, self.learning_rate:learning_rate})
            train_writer.add_summary(xmg, i)
            if i % 50 == 0:
                tb, tl = self.get_batch(self.test_images, self.test_labels, 256)
                vl = sess.run(self.loss,feed_dict={self.image_batch:tb, self.label_batch:tl})
                #summary = sess.run(merged, feed_dict={self.image_batch:bx, self.label_batch:by})
                test_summary = sess.run(merged, feed_dict={self.image_batch:tb, self.label_batch:tl})
                test_writer.add_summary(test_summary, i)                
                print(i, xl, vl)

if sys.platform == 'darwin':
    datapre = '/Users/cjw/'
else:
    datapre = '/ssd1/cjw/'

datafile = datapre + 'Data/cyto/Classifier/images.mm'
labelsfile = datapre + 'Data/cyto/Classifier/labels.mm'
c = Classifier(datafile, labelsfile, 32, 5)
c.train(learning_rate=0.002)
