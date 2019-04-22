import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

## this might be funky ##
class Classifier:
    def __init__(self, datafile, labelsfile, w, nc):

        self.w = w
        self.nc = nc
        self.nfilters = 8
        _images = self.readmm(datafile)
        _labels = self.read_labels_mm(labelsfile, _images)
        self.images, self.labels = self.permute_data_and_labels(_images, _labels)

        self.label_nums = np.argmax(self.labels, axis=1)        

        cilist = list()
        cllist = list()
        cnums = [3, 7]
        print("reducing classes")
        for i, c in enumerate(cnums):
            print(i, c)
            cl = self.label_nums == c
            ci = self.images[cl]
            lc = np.zeros((ci.shape[0], len(cnums)), dtype=np.float32)
            ohv = np.zeros(len(cnums), dtype = np.float32)
            ohv[i] = 1
            lc[:] = ohv
            cilist.append(ci)
            cllist.append(lc)
            print(lc)
        # c3 = self.label_nums == 5
        # ic8 = self.images[c8]
        # ic3 = self.images[c3]

        # lc8 = np.zeros((ic8.shape[0], 2), dtype=np.float32)
        # lc3 = np.zeros((ic3.shape[0], 2), dtype=np.float32)
        # lc8[:] = [0, 1]
        # lc3[:] = [1, 0]

        x_images = np.concatenate(cilist, axis=0)
        x_labels = np.concatenate(cllist, axis=0)        
        self.images, self.labels = self.permute_data_and_labels(x_images, x_labels)
        
        self.label_nums = np.argmax(self.labels, axis=1)        
        self.normalize()
        self.nclasses = self.labels.shape[1]
        self.nclusters = self.label_nums.max() + 1
        self.set_ttv(.8, .1, .1)
        print(self.test_images.shape)

    def readmm(self, datafile, w=64, nc=5):
        mm = np.memmap(datafile, dtype=np.float32)
        mm = mm.reshape((-1, w, w, nc))
        x = mm[:,16:48, 16:48, [1, 2, 3]]
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
    
    def dnet_block(self, x, nf, k, drate, is_training=True):
        
        h = tf.layers.conv2d(x, nf, k, strides=1,
                             padding='same', dilation_rate=drate,
                             kernel_initializer=None,
                             kernel_regularizer=self.get_regularizer(),
                             activation=None)

        h = tf.nn.leaky_relu(h)
        if is_training:
            h = tf.layers.dropout(h, rate=1)
            
        h = tf.layers.conv2d(h, nf, k, strides=2,
                             padding='same', dilation_rate=drate,
                             kernel_initializer=None,
                             kernel_regularizer=self.get_regularizer(),
                             activation=None)

        if is_training:
            h = tf.layers.dropout(h, rate=1)        
        h = tf.nn.leaky_relu(h)
        return h

    def create_network(self, batch, is_training=True):
        ## just make a nice classification thing
        layers = list()
        layers.append(batch)
        ns = 8
        h = self.dnet_block(batch, 16, 3, 1)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat1')

        h = self.dnet_block(h, 32, 3, 1)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat2')

        h = self.dnet_block(h, 64, 3, 1)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat4')

        h = self.dnet_block(h, 128, 3, 1)
        layers.append(h)
        #h = tf.concat(layers, -1, name='concat8')
        print(h)
        h = tf.layers.flatten(h)
        h = tf.layers.dense(h, 500)
        h = tf.nn.leaky_relu(h)

        h = tf.layers.dense(h, 100,
                            kernel_regularizer=self.get_regularizer())
        
        h = tf.nn.leaky_relu(h)
        
        h = tf.layers.dense(h, self.nclasses ,
                             kernel_regularizer=self.get_regularizer())        
        
        self.logits = h
        self.softmax = tf.nn.softmax(h)

    def create_loss(self, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=labels)
        loss = tf.reduce_sum(loss, axis=(-1))
        l2_loss = tf.losses.get_regularization_loss()
        print("loss before reduction", loss)
        self.loss = tf.reduce_mean(loss) + l2_loss
        self.l2_loss = l2_loss
        print(self.loss)

    def create_opt(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                     name='adam_opt').minimize(self.loss)

    def create_placeholders(self):
        self.image_batch = tf.placeholder(tf.float32, shape=(None, self.w, self.w, 3))
        self.label_batch = tf.placeholder(tf.float32, shape=(None, self.nclasses))
        self.learning_rate = tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, shape=())
        
    def create_accuracy(self, y, y_):
        cpred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(cpred, tf.float32))
        
                                                    
    def train(self, learning_rate=0.001):
        tf.reset_default_graph()
        self.create_placeholders()
        self.create_network(c.image_batch, is_training=self.is_training)
        self.create_loss(self.label_batch)
        self.create_accuracy(self.softmax, self.label_batch)
        self.create_opt()
        print("***********************")


        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('l2 loss', self.l2_loss)
        tf.summary.histogram('logits', self.softmax)
        tf.summary.histogram('clusters', tf.argmax(self.softmax, 1))
        tf.summary.histogram('truth', tf.argmax(self.label_batch, 1))
        tf.summary.image('image', self.image_batch)
        print(tf.reduce_mean(self.softmax, axis=-1))
        print(tf.reduce_mean(self.softmax, axis=0))        
        tf.summary.scalar('test_loss', self.loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test')
        m_writer = tf.summary.FileWriter('logs/m')
        
        for i in range(200000):
            if i % 100 == 0:
                learning_rate -= .0002*learning_rate
                print('learning rate set to ', learning_rate)
                
            bx, by = self.get_balanced_batch(self.train_images,
                                             self.train_labels,
                                             self.class_where_train, 128)

            _, xl = sess.run([self.opt, self.loss],
                             feed_dict={self.image_batch:bx, self.label_batch:by,
                                        self.learning_rate:learning_rate,
                                        self.is_training:True})

            if i % 200 == 0:
                tb, tl = self.get_balanced_batch(self.test_images,
                                                 self.test_labels,
                                                 self.class_where_test,
                                                 128)
                vl = sess.run(self.loss,feed_dict={self.image_batch:tb,
                                                   self.label_batch:tl,
                                                   self.is_training:False})
                
                summary = sess.run(merged, feed_dict={self.image_batch:bx, self.label_batch:by,
                                                       self.is_training:False})
                
                test_summary = sess.run(merged, feed_dict={self.image_batch:tb,
                                                           self.label_batch:tl, self.is_training:False})
                train_writer.add_summary(summary, i)
                test_writer.add_summary(test_summary, i)                
                print(i, xl, vl)

if sys.platform == 'darwin':
    datapre = '/Users/cjw/'
else:
    datapre = '/ssd1/cjw/'

datafile = datapre + 'Data/cyto/Classifier/images.mm'
labelsfile = datapre + 'Data/cyto/Classifier/labels.mm'
c = Classifier(datafile, labelsfile, 32, 5)
c.train(learning_rate=0.0008)
