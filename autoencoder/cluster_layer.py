import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

from matplotlib import pyplot as plt 

from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import KLD
from tensorflow.layers import InputSpec

import autoencoder.adversarial as aae
import autoencoder.utils as utils

class cluster_layer(Layer):

    def __init__(self, k=10, weights=None, **kwargs):
        """initialize this layer
        Parameters:
        k : integer
            The number of clusters
        weights : Tensor
            size is(latent space size X number of clusters)
         
        """
        super(cluster_layer, self).__init__(**kwargs)
        self.k = k

        if weights is None:
            self.init_with_weights = None
        else:
            self.init_with_weights = weights
            
        self.input_spec = InputSpec(ndim=2)


    def call(self, inputs):
        """ in a dense network, this would be a matrix mul or dot
        product, but here it is going to be a student's t test of
        each of the 'points' to the cluster centers and give a soft
        probability for the prediected cluster
        """
        '''
        x = tf.expand_dims(inputs, axis=1)
        qn = 1./(1.0 + (tf.norm((x - self.clusters), axis=2)))
        
        nrlz = tf.reduce_sum(qn, axis=1)
        qnt = tf.transpose(qn)
        qt = tf.divide(qnt, nrlz)
        q = tf.transpose(qt)
        print("CALL ", qn.shape, q.shape)
        '''
        '''try to use softmax'''
        '''
        x = tf.expand_dims(inputs, axis=1)
        kern = x - self.clusters
        q = 1 - tf.nn.softmax(tf.norm(kern, axis=2))
        '''
        '''try '''
        
        x = tf.expand_dims(inputs, axis=1)
        kern = x - self.clusters
        kern = kern*kern
        kern = tf.reduce_sum(kern, axis=2)
        kern = 1 + kern
        kern = 1./kern
        q = tf.transpose(tf.transpose(kern)/tf.reduce_sum(kern, axis=1))
        print("CALL ",  q.shape, x.shape)
        return q
        
        
    def build(self, input_shape):
        """ create the variables
        - the input, the trainable weights,
        - the output, other stuff?
        """
        print("building cluster layer", input_shape)
        self.input_spec = InputSpec(dtype=tf.float32,
                                    shape=(None, input_shape[1]))

        print('after input_spec')
        print(self.k, input_shape[1], type(input_shape[1].value), self.dtype)
        self.clusters = self.add_variable(
            name='clusters',
            shape=[self.k, input_shape[1].value],
            initializer=tf.truncated_normal_initializer(stddev=1.04),
            dtype=self.dtype,
            trainable=True)

        print(self.clusters.shape)
        print(self.init_with_weights.shape)
        if self.init_with_weights is not None:
            self.clusters = tf.assign(self.clusters, self.init_with_weights)
            del self.init_with_weights
            
        self.built = True
        print('done building')
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k)


def cluster_target(q):
    w = (q*q)/q.sum(axis=0) #sum over batch/sample axis, left with n clusters
    res = (w.T/w.sum(axis=1)).T
    return res

def tf_cluster_target(q):
    with tf.variable_scope('cluster'):
        qq = tf.square(q)
        qn = tf.reduce_sum(q, axis=0)
        w = tf.divide(qq, qn)
        wt = tf.transpose(w)
        ws = tf.reduce_sum(w, axis=1)
        p = tf.transpose(tf.divide(wt, ws))
    return p

def cluster_loss(images, encoder, decoder, cluster):
    ''' need reconstruction loss and cluster loss'''
    with tf.variable_scope('cluster'):
        r1 = tf.reduce_sum(tf.square(images- decoder), axis=(1,2,3))
        rloss = tf.reduce_mean(r1)
        p = tf_cluster_target(cluster)
        xentropy = -tf.reduce_sum(p * tf.log(cluster + 0.00001))
        entropy = -tf.reduce_sum(p * tf.log(p + 0.00001))

        closs = xentropy - entropy

        kd = tf.reduce_sum(tf.keras.losses.KLD(p, cluster))
        loss = rloss + kd #+ closs
    return loss, rloss
    
def cluster_train(trained):

    ## Z is the latent space from a sampling of images
    ## k i s the number of clusters

    ### setup the initial clusters

    w = trained.params['width']
    sample_images = utils.get_sample(trained.mmdict, trained.df,
                                     12000, w,
                                     trained.params['nchannels'],
                                     channels=trained.params['channels'])


    print(sample_images.shape)
#    images = tf.placeholder(tf.float32, (None, w, w, trained.params['nchannels']))
#    print(images)
    Z = trained.encoder.eval({trained.images:sample_images}, session=trained.sess)

    k = trained.params['nclusters']
    #kmeans = KMeans(n_clusters=k, n_init=20)
    dbscan = DBSCAN(eps=1.2)
    db = dbscan.fit(Z)
    print(db)
    core = dbscan.core_sample_indices_
    nc = len(core)
    print("N cores", nc)
    
    k = nc
    #km = kmeans.fit_predict(Z)
    
    print("original clusters")
    print(np.histogram(db.labels_, k)[0])
    #plt.hist(km, bins=20)
    #plt.show()
    km_last = np.copy(db)
    #iweights = kmeans.cluster_centers_
    iweights = Z[core]
    plt.hist(np.reshape(iweights, (-1)))
    plt.show()
    with tf.variable_scope("cluster"):
        acluster = cluster_layer(k=nc, weights=iweights)
        cluster = acluster.apply(trained.encoder)

        '''use this to calc the z and reconstruction when ready for batches'''
        '''but do this by running the encoder and decoder with sess.run and 
            a feed dict through an optimizer, like usual
        '''

        closs, rloss = cluster_loss(trained.images, trained.encoder, trained.decoder, cluster)

        copt = tf.train.AdamOptimizer(1*trained.params['learning_rate']).minimize(
            closs)

    tvar = tf.trainable_variables()
    
    #cvars = [a for a in tvar if 'cluster' in a.name]
    cvars = tf.global_variables(scope='cluster')
    #print(cvars)
    trained.sess.run(tf.variables_initializer(cvars))
    #trained.load()
    
    for i in range(12000):

        batch = utils.get_sample(trained.mmdict, trained.df,
                                 trained.params['batchsize'], w,
                                 trained.params['nchannels'],
                                 channels=trained.params['channels'])
        #print(batch.shape, trained.images)
        trained.sess.run(copt, feed_dict={trained.images:batch})
        if i % 1000 == 0:
            zloss, zrloss,_ = trained.sess.run([closs, rloss, copt],
                                               feed_dict={trained.images:batch})
            print(i, zloss, zrloss)
            ze = trained.encoder.eval({trained.images:sample_images}, session=trained.sess)
            #print(np.histogram(ze, 10)[0])
            qx = cluster.eval({trained.encoder:ze}, session=trained.sess)
            kmp = np.argmax(qx, axis=1)
            print(np.histogram(kmp, nc)[0])
            
            #print(np.sum(qx[0:5,:], axis=1))
            
            
            delta_label = np.sum(kmp != km_last).astype(np.float32) / kmp.shape[0]
            km_last = kmp
            print(delta_label)
            
    res = cluster.eval({trained.images:batch}, session = trained.sess)
    print(res, res.shape)
    bZ = trained.encoder.eval({trained.images:sample_images}, session=trained.sess)
    bR = trained.decoder.eval({trained.encoder:Z}, session=trained.sess)
    return cluster
    
     
     
     
     
    


