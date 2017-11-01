import tensorflow as tf
import numpy as np

def sampling(z_mean, z_log_sigma, batchsize, length):
    eps = tf.random_normal(shape=(batchsize, length), mean=0., stddev=4.)
    return z_mean + tf.exp(z_log_sigma) * eps


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x * alpha, x)


def clipped_u(x, clip=2.):
    return tf.clip_by_value(x, -clip, clip)


def get_init():
    return tf.truncated_normal_initializer(stddev=.01)


def get_reg():
    return tf.contrib.layers.l2_regularizer(.01)


def dud():
    def __enter__():
        return 1

    def exit():
        pass

def dropout(x, is_train, rate):
    if is_train:
        x = tf.nn.dropout(x, rate)
    return x

def layer_conv2d(x, nfilters, size, strides, padding, name,
                 droprate, is_train, activation=None):

    z = tf.layers.conv2d(x, nfilters, size, strides=strides,
                         padding=padding, kernel_initializer=get_init(),
                         name=name, activation=activation)
    z = leaky_relu(z)
    z = dropout(z, is_train, droprate)
    return z


def encoder(images, latent_size, droprate=0.7, is_train=True,
            nfilters=None):
    print('Encoder', is_train)
    # images = tf.placeholder(tf.float32, (None, height, width, nchannels))
    ## create the model using hte images
    if nfilters is None:
        k = 64 * np.asarray([1, 2, 4, 8], dtype=np.int32)
        k = [(64, 5), (128, 3), (256, 3), (512, 3)]
    else:
        k = nfilters

    #     with tf.variable_scope("encoder", reuse=(not is_train)):
    if 1 == 1:
        layers = list()
        layers.append(images)
        for i, ki in enumerate(k):
            hc = layer_conv2d(layers[-1], ki[0], ki[1], 2, "same",
                               "filter_{:02d}".format(i),
                               droprate, is_train, activation=None)
            layers.append(hc)

            # hc1 = tf.layers.conv2d(images, k[0], 5, strides=2, padding="same",
            #                        activation=None,
            #                        kernel_initializer=get_init(), name='filter_1')
            # hc1 = leaky_relu(hc1)
            # hc1 = dropout(hc1, is_train, droprate)
            #
            # hc2 = tf.layers.conv2d(hc1, k[1], 3, strides=2, padding="same",
            #                        activation=None,
            #                        kernel_initializer=get_init(), name='filter_2')
            # hc2 = leaky_relu(hc2)
            # hc2 = dropout(hc2, is_train, droprate)
            #
            # hc3 = tf.layers.conv2d(hc2, k[2], 3, strides=2, padding="same",
            #                        activation=tf.nn.elu,
            #                        kernel_initializer=get_init(), name='filter_3')
            # hc3 = leaky_relu(hc3)
            # hc3 = dropout(hc3, is_train, droprate)
            #
        h = tf.contrib.layers.flatten(layers[-1])
        he = tf.layers.dense(h, latent_size, kernel_initializer=get_init(),
                             activation=None,
                             name='latent_space')
        print(layers, he)
    return he

def layer_upconv(x, nfilters, size, strides,
                 padding, tname, droprate, is_train):

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

        dh4 = tf.reshape(dh, (-1, isize, isize, k[0][0]))
        layers.append(dh4)

        for i, ki in enumerate(k):
            tname = "upconv_{:02d}".format(i)
            dh  = layer_upconv(layers[-1], ki[0], ki[1], 2, "same",
                               tname, droprate, is_train)
            layers.append(dh)

        # dh3 = tf.layers.conv2d_transpose(dh4, k[2], 3, strides=2,
        #                                  padding='same',
        #                                  activation=None,
        #                                  kernel_initializer=get_init())
        # dh3 = leaky_relu(dh3)
        # dh3 = dropout(dh3, is_train, droprate)
        #
        # dh2 = tf.layers.conv2d_transpose(dh3, k[1], 3, strides=2,
        #                                  padding='same',
        #                                  activation=None,
        #                                  kernel_initializer=get_init())
        # dh2 = leaky_relu(dh2)
        # dh2 = tf.nn.dropout(dh2, .7)
        #
        dh0 = tf.layers.conv2d_transpose(layers[-1], nchannels, 5, strides=2,
                                         padding='same',
                                         activation=None,
                                         kernel_initializer=get_init(),
                                         name='decoder_out')

        sdh0 = tf.nn.sigmoid(dh0)  # , name='decoder_image')
        print(layers, dh0)
    return sdh0


def ae_loss(images, sdh0, nchannels=1, latent_size=32, is_train=True,
             width=64):

#    xloss = -tf.reduce_sum(images * tf.log(sdh0) +
#                          (1 - images) * tf.log(1 - sdh0), (1, 2, 3))

    rloss = tf.reduce_sum(tf.square(images - sdh0), (1, 2, 3))
    loss = tf.reduce_mean(rloss)
    return loss


def model_opt(loss, learning_rate):
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=.5).minimize(loss)
    return opt


