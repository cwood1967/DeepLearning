import tensorflow as tf
import numpy as np

def sampling(z_mean, z_log_sigma, batchsize, length):
    eps = tf.random_normal(shape=(batchsize, length), mean=0., stddev=4.)
    return z_mean + tf.exp(z_log_sigma) * eps


def leaky_relu(x, alpha=0.2):
    """
    Calculate activation with leaky Relu
    Parameters
    ----------
    x : tensor
    alpha : float

    Returns
    -------
    Activation value
    """
    return tf.maximum(x * alpha, x)


def clipped_u(x, clip=2.):
    return tf.clip_by_value(x, -clip, clip)


def get_init():
    """
    Create a layer initializer
    Returns
    -------
    Normal initializer
    """
    return tf.truncated_normal_initializer(stddev=.04) #.04


def get_reg():
    return tf.contrib.layers.l2_regularizer(.01)


def dud():
    def __enter__():
        return 1

    def exit():
        pass

def dropout(x, is_train, rate):
    """
    Perform dropout on the input tensor
    Parameters
    ----------
    x : tensor
        Input tensor
    is_train : boolean
        True if training
    rate : float
        Keep probability

    Returns
    -------

    """
    if is_train:
        x = tf.nn.dropout(x, rate)
    return x

def layer_conv2d(x, nfilters, size, strides, padding, name,
                 droprate, is_train, activation=None):
    """
    Create a 2D convolutional layer
    
    Parameters
    ----------
    x : Tensor
    nfilters : int
        Number of filter layer
    size: int
        Size of the filter kernel
    strides : int
        Stride length of the kernel
    padding : str
        How to do the padding on the edges
    name : str
        name of the tensorflow variable
    droprate : float
        keep probability
    is_train : boolean
        True is doing training
    activation : function
        activation function

    Returns
    -------
    z : tensor
    """

    z = tf.layers.conv2d(x, nfilters, size, strides=strides,
                         padding=padding, kernel_initializer=get_init(),
                         name=name, activation=activation)
    z = leaky_relu(z)
    z = dropout(z, is_train, droprate)
    return z


def encoder(images, latent_size, droprate=0.7, is_train=True,
            nfilters=None):
    """
    Build the encoder part of th neural network
    
    Parameters
    ----------
    images : numpy.array
        batch of images
    latent_size : int
        size of the final layer
    droprate : float
        keep probability
    is_train : boolean
        running as training or inference
    nfilters : list
        list of (stack size, filter size)

    Returns
    -------
    he : Tensor
        fully connected layer of size (batchsize, layer_layer_size)
        
    """
    print('Encoder', is_train)
    # images = tf.placeholder(tf.float32, (None, height, width, nchannels))

    """create the model using the images"""
    if nfilters is None:
        k = 64 * np.asarray([1, 2, 4, 8], dtype=np.int32)
        k = [(64, 5), (128, 3), (256, 3), (512, 3)]
    else:
        k = nfilters

    if 1 == 1:
        layers = list()
        layers.append(images)
        for i, ki in enumerate(k):
            """Use the last element on the layers list"""
            hc = layer_conv2d(layers[-1], ki[0], ki[1], 2, "same",
                               "filter_{:02d}".format(i),
                               droprate, is_train, activation=None)
            layers.append(hc)

        h = tf.contrib.layers.flatten(layers[-1])
        he = tf.layers.dense(h, latent_size, kernel_initializer=get_init(),
                             activation=None,
                             name='latent_space')
        print(layers, he)
    return he

def layer_upconv(x, nfilters, size, strides,
                 padding, tname, droprate, is_train):
    """
    Create a layer for the decoder, up-scale the tensor
    
    Parameters
    ----------
    x : tensor
    nfilters : int
        number of filter layers
    size : int
        size of the filter kernel
    strides : int
        length of convolution stride
    padding : str
        Type of padding
    tname : str
        variable name
    droprate : float
        Keep probability
    is_train : boolean
        True if for training

    Returns
    -------
    z : tensor
    
    """

    z = tf.layers.conv2d_transpose(x, nfilters, size, strides=strides,
                                     padding=padding,
                                     activation=None,
                                     kernel_initializer=get_init())

    z = leaky_relu(z)
    z = dropout(z, is_train, droprate)
    return z

def decoder(z, nchannels=2, width=64, droprate=.7, is_train=True,
            nfilters=None):

    if nfilters is None:
        #ks = 32 * np.asarray([1, 2, 4, 8], dtype=np.int32)
        k = [(256, 3), (128, 3), (64, 3), (32, 5)]
    else:
        k = nfilters

    ## size of the first "image"
    isize = width // int(np.exp2(len(k) - 0))
    print("isize: ", isize, width)

    #     with tf.variable_scope("decoder", reuse=(not is_train)):
    if 1 == 1:
        layers = list()
        dh = tf.layers.dense(z, isize * isize * k[0][0], activation=None,
                             kernel_initializer=get_init())
        dh = leaky_relu(dh)
        dh = dropout(dh, is_train, droprate)
        layers.append(dh)



        for i, ki in enumerate(k):
            if i == 0:
                dh = tf.reshape(layers[-1], (-1, isize, isize, k[0][0]))
                layers.append(dh)
            else:
                tname = "upconv_{:02d}".format(i)
                dh  = layer_upconv(layers[-1], ki[0], ki[1], 2, "same",
                                   tname, droprate, is_train)
                layers.append(dh)

        dh0 = tf.layers.conv2d_transpose(layers[-1], nchannels, 5, strides=2,
                                         padding='same',
                                         activation=None,
                                         kernel_initializer=get_init(),
                                         name='decoder_out')

        sdh0 = tf.nn.sigmoid(dh0)  # , name='decoder_image')
        print(layers, dh0)
    return sdh0


def ae_loss(images, sdh0):
    """
    Calculate the loss, just [input image] - [decoded image]
    Parameters
    ----------
    images : tensor
        Input images
    sdh0 : tensor
        Decoded images
    
    Returns
    -------
    loss : float
        The reduced loss over all samples
    """
#    xloss = -tf.reduce_sum(images * tf.log(sdh0) +
#                          (1 - images) * tf.log(1 - sdh0), (1, 2, 3))

    rloss = tf.reduce_sum(tf.square(images - sdh0), (1, 2, 3))
    loss = tf.reduce_mean(rloss)
    return loss


def model_opt(loss, learning_rate):
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=.5).minimize(loss)
    return opt


