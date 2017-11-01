
import sys
import warnings
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt



# print(sys.path)
import network
import utils


def check_gpu():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def display(sess, x1, x2, x3, x4, x5, x6, p1, p2, p3, p4):
    f = plt.figure(figsize=(12, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(x1)
    plt.subplot(1, 6, 2)
    plt.imshow(x2)

    plt.subplot(1, 6, 3)
    plt.imshow(x3)
    plt.subplot(1, 6, 4)
    plt.imshow(x4)

    plt.subplot(1, 6, 5)
    plt.imshow(x5)
    plt.subplot(1, 6, 6)
    plt.imshow(x6)

    plt.show()
    plt.plot(p1)
    plt.plot(p2)
    plt.plot(p3, color='red')
    plt.plot(p4, color='black')
    plt.show()

''' need to have the data file, a csv file with x, y, etc
to create the dataframe
'''


def train(mmdict, df, params):
    """Train the autoencoder neural network and save the results
    
    Arguments:
        mmdict -- dictionary of memory map files
        params -- dictionay of parameters
    """

    tf.reset_default_graph()

    width = params['width']
    height = params['height']
    nchannels = params['nchannels']
    channels = params['channels']
    nepochs = params['nepochs']
    batchsize = params['batchsize']
    learning_rate = params['learning_rate']
    restore = params['restore']
    latent_size = params['latent_size']

    images = tf.placeholder(tf.float32, (None, height, width, nchannels))
    z = tf.placeholder(tf.float32, (None, latent_size))

    enc = network.encoder(images, latent_size, droprate=.7, is_train=True)
    sdd = network.decoder(enc, nchannels=nchannels, width=width, droprate=.7,
                          is_train=True)

    loss = network.ae_loss(images, sdd, nchannels=nchannels,
                           latent_size=latent_size, width=width)

    opt = network.model_opt(loss, learning_rate)

    test_batch, _, _ = utils.getbatch(mmdict, df, len(df) // batchsize,
                                      batchsize, width, nchannels,
                                      channels=channels)

    samp = 26

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if restore:
            pass
        else:
            counter = 0
            for e in range(nepochs):
                start = 0
                ib = 0
                while start < len(df) // batchsize - 1:
                    batch, wells, rownums = utils.getbatch(mmdict,
                                                           df, start, batchsize,
                                                           width, nchannels,
                                                           channels=channels)

                    # aenc = sess.run(enc,
                    #                 feed_dict={images:batch})

                    asdd, aenc, az, _ = sess.run([sdd, enc, loss, opt],
                                                 feed_dict={images: batch})

                    ni = np.random.randint(0, batchsize)

                    if ib % 1000 == 0:
                        sdh0r = asdd[ni]
                        test_he = sess.run(enc, feed_dict={images: test_batch})
                        test_sdd = sess.run(sdd, feed_dict={enc: test_he,
                                                            images: test_batch})

                        print('Epoch: ', e, 'Iteration: ', ib, 'Loss: ', az)
                        display(sess, sdh0r[:, :, 0], batch[ni, :, :, 0],
                                test_sdd[ni, :, :, 0], test_batch[ni, :, :, 0],
                                test_sdd[23, :, :, 0], test_batch[23, :, :, 0],
                                test_he[ni], aenc[ni],
                                test_he[23], test_he[0])
                    ib += 1
                    counter += 1
                    start += 1
                    if ib > 2000000:
                        break
                saver.save(sess, 'autoencoder-128x', global_step=counter)
    print("Done")

if __name__ == '__main__':

    datadir = "/Users/cjw/Projects/cjw/yeastAE/Data/yeast/"
    datafile = datadir + "plate15_cells.csv"
    p_df = utils.read_data_file(datafile)
    mmfilename = datadir + "mmplate15-1.mm"
    mm = np.memmap(mmfilename, dtype='int32', mode='r',
                   shape=(4,))
    mmshape = mm.shape
    xshape = (mm[0], mm[1], mm[2], mm[3])
    del mm

    m2 = np.memmap(mmfilename, dtype='float32', offset=128,
                   mode='r', shape=xshape)
    p_mmdict = {"mmplate15-1.mm": m2}

    p_width = 32
    p_height = 32
    p_nchannels = 1
    p_channels = [3]
    p_nepochs = 2
    p_batchsize = 32
    p_learning_rate = 0.0004
    p_restore = False
    p_latent_size = 128

    params = dict()

    params['width'] = p_width
    params['height'] = p_height
    params['nchannels'] = p_nchannels
    params['channels'] = p_channels
    params['nepochs'] = p_nepochs
    params['batchsize'] = p_batchsize
    params['learning_rate'] = p_learning_rate
    params['restore'] = p_restore
    params['latent_size'] = p_latent_size

    train(p_mmdict, p_df, params)









