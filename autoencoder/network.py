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

def encoder(images, latent_size, droprate=0.7, is_train=True):
    print('Encoder', is_train)
    # images = tf.placeholder(tf.float32, (None, height, width, nchannels))
    ## create the model using hte images
    k = 64 * np.asarray([1, 2, 4, 8], dtype=np.int32)
    #     with tf.variable_scope("encoder", reuse=(not is_train)):
    if 1 == 1:
        hc1 = tf.layers.conv2d(images, k[0], 5, strides=2, padding="same",
                               activation=None,
                               kernel_initializer=get_init(), name='filter_1')
        hc1 = leaky_relu(hc1)
        hc1 = dropout(hc1, is_train, droprate)

        hc2 = tf.layers.conv2d(hc1, k[1], 3, strides=2, padding="same",
                               activation=None,
                               kernel_initializer=get_init(), name='filter_2')
        hc2 = leaky_relu(hc2)
        hc2 = dropout(hc2, is_train, droprate)

        hc3 = tf.layers.conv2d(hc2, k[2], 3, strides=2, padding="same",
                               activation=tf.nn.elu,
                               kernel_initializer=get_init(), name='filter_3')
        hc3 = leaky_relu(hc3)
        hc3 = dropout(hc3, is_train, droprate)

        h = tf.contrib.layers.flatten(hc3)
        he = tf.layers.dense(h, latent_size, kernel_initializer=get_init(),
                             activation=None,
                             name='latent_space')

    return he


def decoder(z, nchannels=2, width=64, droprate=.7, is_train=True):
    isize = width // 8
    print(isize, width)
    k = 32 * np.asarray([1, 2, 4, 8], dtype=np.int32)
    #     with tf.variable_scope("decoder", reuse=(not is_train)):
    if 1 == 1:
        dh = tf.layers.dense(z, isize * isize * k[3], activation=None,
                             kernel_initializer=get_init())
        dh = leaky_relu(dh)
        dropout(dh, is_train, droprate)

        dh4 = tf.reshape(dh, (-1, isize, isize, k[3]))
        dh3 = tf.layers.conv2d_transpose(dh4, k[2], 3, strides=2,
                                         padding='same',
                                         activation=None,
                                         kernel_initializer=get_init())
        dh3 = leaky_relu(dh3)
        dh3 = dropout(dh3, is_train, droprate)

        dh2 = tf.layers.conv2d_transpose(dh3, k[1], 3, strides=2,
                                         padding='same',
                                         activation=None,
                                         kernel_initializer=get_init())
        dh2 = leaky_relu(dh2)
        dh2 = tf.nn.dropout(dh2, .7)

        dh0 = tf.layers.conv2d_transpose(dh2, nchannels, 5, strides=2,
                                         padding='same',
                                         activation=None,
                                         kernel_initializer=get_init(),
                                         name='decoder_out')

        sdh0 = tf.nn.sigmoid(dh0)  # , name='decoder_image')

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


