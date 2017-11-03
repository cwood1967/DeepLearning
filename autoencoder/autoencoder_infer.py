import sys
import warnings
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import network
import utils

def infer(mmdict, df, params):
    tf.reset_default_graph()

    width = params['width']
    height = params['height']
    nchannels = params['nchannels']
    channels = params['channels']
    #nepochs = params['nepochs']
    batchsize = params['batchsize']
    #learning_rate = params['learning_rate']
    #restore = params['restore']
    latent_size = params['latent_size']
    enc_sizes = params['enc_sizes']
    dec_sizes = params['dec_sizes']

    images = tf.placeholder(tf.float32, (None, height, width, nchannels))
    z = tf.placeholder(tf.float32, (None, latent_size))

    enc = network.encoder(images, latent_size, droprate=.7, is_train=True,
                          nfilters=enc_sizes)
    sdd = network.decoder(enc, nchannels=nchannels, width=width, droprate=.7,
                          is_train=True, nfilters=dec_sizes)

    loss = network.ae_loss(images, sdd, nchannels=nchannels,
                           latent_size=latent_size, width=width)

    # test_batch, _, _ = utils.getbatch(mmdict, df, len(df) // batchsize,
    #                                   batchsize, width, nchannels,
    #                                   channels=channels)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('a128/autoencoder-128-100126.meta')
    saver.restore(sess, tf.train.latest_checkpoint('a128/'))



