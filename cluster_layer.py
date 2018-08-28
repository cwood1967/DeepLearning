import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans

from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import KLD
from tensorflow.layers import InputSpec


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

        x = tf.expand_dims(inputs, axis=1)
        qn = 1./(1.0 + tf.norm((x - self.clusters), axis=2))

        q = qn/tf.reduce_sum(qn)
        return q
        
        
    def build(self, input_shape):
        """ create the variables
        - the input, the trainable weights,
        - the output, other stuff?
        """

        self.input_spec = InputSpec(dtype=tf.float32,
                                    shape=(None, input_shape[1]))

        self.clusters = self.add_variable(
            name='clusters',
            shape=[self.k, input_shape[1]],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=self.dtype,
            trainable=True)

        print(self.clusters.shape)
        print(self.init_with_weights.shape)
        if self.init_with_weights is not None:
            self.clusters = tf.assign(self.clusters, self.init_with_weights)
            del self.init_with_weights
            
        self.built = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k)


def cluster_target(q):
    w = (q*q)/q.sum(axis=0) #sum over batch/sample axis, left with n clusters
    res = (w.T/w.sum(axis=1)).T
    return res


    
    


