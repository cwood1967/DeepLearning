import time
import os

import tensorflow as tf
import numpy as np
import pandas as pd

import autoencoder.network as network
import autoencoder.utils as utils
import  autoencoder.adversarial as aae
from matplotlib import pyplot as plt

class aae_clustering():

    def __init__(self, params):
        self.width = params['width']
        self.height = params['height']
        self.nchannels = params['nchannels']
        self.channels = params['channels']
        self.nepochs = params['nepochs']
        self.batchsize = params['batchsize']
        self.learning_rate = params['learning_rate']
        self.restore = params['restore']
        self.latent_size = params['latent_size']
        self.enc_sizes = params['enc_sizes']
        self.dec_sizes = params['dec_sizes']
        self.droprate = params['droprate']
        self.stdev = params['stdev']
        self.denoise = params['denoise']
        self.slam = params['slam']
        self.intializer = network.get_init
        self.nclusters = params['nclusters']

    def d_initializer(self, stdev):
        """
        Create a layer initializer
        Returns
        -------
        Normal initializer
        """
        return tf.truncated_normal_initializer(stddev=stdev) #.04
    
    def create_encoder(self, images, is_train, reuse=False):
        
        image = images

        print('Encoder', is_train)
        print(image, image.shape)
        """create the model using the images"""

        k = self.enc_sizes
        sknum = ""
        layers = list()
        layers.append(image)
        with tf.variable_scope("encoder", reuse=reuse):
            for i, ki in enumerate(k):
                """Use the last element on the layers list"""
                hc = network.layer_conv2d(layers[-1], ki[0], ki[1], 2, "same",
                                   "filter{:s}_{:02d}".format(sknum, i), self.stdev,
                                   self.droprate, is_train, activation=None)
                layers.append(hc)

            h = tf.contrib.layers.flatten(layers[-1])
            h = network.leaky_relu(h)
            h = network.dropout(h, is_train, self.droprate)
            
            encoder_z = tf.layers.dense(h, self.latent_size,
                                 kernel_initializer=network.get_init(self.stdev),
                                 activation=None,
                                 name='latent_space' + sknum)

            # hy = tf.layers.dense(h, self.latent_size,
            #                      kernel_initializer=network.get_init(self.stdev),
            #                      activation=None,
            #                      name='clusters_1' + sknum)
            # hy = tf.tanh(hy)
            # hy = network.dropout(h, is_train, .6)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            hy = tf.layers.dense(h, self.nclusters, use_bias=False,
                                 kernel_initializer=network.get_init(0.01*self.stdev),
                                 activation=None,
                                 kernel_regularizer=regularizer,
                                 name='clusters' + sknum)

            encoder_y = tf.nn.softmax(hy)
            #print(layers, he)
        return encoder_z, encoder_y

    def create_decoder(self, encoder_z, encoder_y, is_train, reuse=False):

        sknum = ""
        k = self.dec_sizes
        isize = self.width // int(np.exp2(len(k) - 0))
        hshape = k[0][0]*isize*isize
        with tf.variable_scope("decoder", reuse=reuse):
            hy = tf.layers.dense(encoder_y, hshape,
                                 kernel_initializer=network.get_init(self.stdev),
                                 activation=None, 
                                 name="hy")

            hz = tf.layers.dense(encoder_z, hshape,
                                 kernel_initializer=network.get_init(self.stdev),
                                 activation=None,
                                 name="hz")

            h = tf.add(hz, hy)

            ### now start regular decode stuff
            h = network.leaky_relu(h)
            h = network.dropout(h, is_train, self.droprate)

            layers = list()
            layers.append(h)

            for i, ki in enumerate(self.dec_sizes):
                if i == 0:
                    dh = tf.reshape(h, (-1, isize, isize,
                                        self.dec_sizes[0][0]))
                else:
                    tname = "upconv_{:s}_{:02d}".format(sknum, i)
                    dh  = network.layer_upconv(layers[-1], ki[0], ki[1], 2, self.stdev, "same",
                                   tname, self.droprate, is_train)

                layers.append(dh)

            dh0 = tf.layers.conv2d_transpose(layers[-1], self.nchannels, 5, strides=2,
                                 padding='same',
                                 activation=None,
                                 kernel_initializer=network.get_init(self.stdev),
                                 name='decoder_out' + sknum)

            #dh0 = tf.minimum(dh0, 1)
            sdh0 = tf.sigmoid(dh0) #tf.nn.relu(dh0)

        return sdh0, dh0
    
            
    def create_discriminator(self, w, reuse=False, name="discriminator"):
        
        with tf.variable_scope(name, reuse=reuse):
            h1 = tf.layers.dense(w, 1000,
                                 kernel_initializer=self.d_initializer(self.stdev),
                                 activation=None,
                                 name="discrim01")

            h1 = network.leaky_relu(h1)

            h2 = tf.layers.dense(h1, 1000,
                                 kernel_initializer=self.d_initializer(self.stdev),
                                 activation=None,
                                 name="discrim02")

            h2 = network.leaky_relu(h2)

            last = tf.layers.dense(h2, 1,
                                 kernel_initializer=self.d_initializer(.5),
                                 activation=None,
                                 name="discrim03")

            return last

    
    def reconstruction_loss(self, images, decoder, sdecoder, encoder_y):

        cw = tf.constant([.1, 8., .1, .2], dtype=tf.float32)
        
        r = images - sdecoder
        #r1 = tf.reduce_sum(tf.square(r), axis=(1,2,3))
        r1 = tf.reduce_sum(tf.square(r), axis=(1,2)) 
        #r1 =tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder,
        #                                            labels=images)
        #b = tf.reduce_mean(encoder_y, axis=1)
        eloss = tf.reduce_max(tf.reduce_sum(encoder_y, axis=0))
        eloss1 = tf.reduce_min(tf.reduce_sum(encoder_y, axis=0))
        rloss = tf.reduce_mean(r1, axis=0)
        rloss = tf.reduce_sum(tf.multiply(cw, rloss))
        self.rloss = rloss
        self.eloss = eloss - eloss1
        #self.eloss1 = self.batchsize - eloss1
        return rloss, eloss

    def discriminator_loss(self, sample_z, encoder):
        smooth = 0.2
        sample_logits = self.create_discriminator(sample_z, name="discriminator")
        ae_logits = self.create_discriminator(encoder, reuse=True, name="discriminator")

        sample_labels = (1 - smooth)*tf.ones_like(sample_logits)
        ae_labels = smooth + tf.zeros_like(ae_logits)

        # trick the discriminator 
        gen_labels = (1 - smooth)*tf.ones_like(ae_logits)
        sce_sample =tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_logits,
                                                            labels=sample_labels)
        d_sample_loss = tf.reduce_mean(sce_sample)
        sce_ae = tf.nn.sigmoid_cross_entropy_with_logits(logits=ae_logits,
                                                         labels=ae_labels)
        d_ae_loss = tf.reduce_mean(sce_ae)
        d_loss = d_sample_loss + d_ae_loss

        sce_gen = tf.nn.sigmoid_cross_entropy_with_logits(logits=ae_logits,
                                                          labels=gen_labels)
        gen_loss = tf.reduce_mean(sce_gen)

        self.d_loss = d_loss
        self.gen_loss = gen_loss


    def cluster_loss(self, y, encoder_y):
        smooth = 0.
        y_logits = self.create_discriminator(y, name="cluster")
        aey_logits = self.create_discriminator(encoder_y, reuse=True, name="cluster")

        y_labels = (1 - smooth)*tf.ones_like(y_logits)
        aey_labels = smooth + tf.zeros_like(aey_logits)

        #trick the discriminator
        gen_labels = (1 - smooth)*tf.ones_like(aey_logits)
        sce = tf.nn.sigmoid_cross_entropy_with_logits

        sce_y = sce(logits=y_logits, labels=y_labels)
        c_y_loss = tf.reduce_mean(sce_y)

        sce_aey = sce(logits=aey_logits, labels=aey_labels)
        c_aey_loss = tf.reduce_mean(sce_aey)

        c_loss = c_y_loss + c_aey_loss

        sce_cgen = sce(logits=aey_logits, labels=gen_labels)
        c_gen_loss = tf.reduce_mean(sce_cgen)

        self.c_loss = c_loss
        self.c_gen_loss = c_gen_loss

    def opt(self):
        tvar = tf.trainable_variables()
        encvars = [a for a in tvar if 'encoder' in a.name]
        decvars = [a for a in tvar if 'decoder' in a.name]
        dzvars = [a for a in tvar if 'discriminator' in a.name]
        cyvars = [a for a in tvar if 'cluster' in a.name]
        aevars = encvars + decvars

        c_d_loss = self.d_loss + self.c_loss
        c_g_loss = self.gen_loss + self.c_gen_loss
        
        self.ae_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.rloss, var_list=aevars)
        self.e_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.eloss, var_list=aevars)
#        self.e1_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
#            self.eloss1, var_list=aevars)
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
            c_d_loss, var_list=dzvars + cyvars)
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
            c_g_loss, var_list=encvars)
        # self.c_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
        #     self.c_loss, var_list=cyvars)
        # self.gc_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
        #     self.c_gen_loss, var_list=encvars)

        return self.ae_opt, self.d_opt, self.g_opt, self.e_opt #, self.c_opt, self.gc_opt


def one_hot_batch(batchsize, nclusters):

    ohy = np.zeros((batchsize, nclusters), dtype=np.float32)

    ry = np.random.randint(0,nclusters,size=batchsize)
    for i in range(batchsize):
        ohy[i,ry[i]] = 1.0

    return ohy

        
def setup():
    esize = [(128,3), (256, 3), (512,3)]
    dsize = list(reversed(esize))

    params =dict()

    params['nclusters'] =18
    params['width'] = 32
    params['height'] = 32
    params['nchannels'] = 4
    params['channels'] = [0,1,3,4]
    params['nepochs'] = 20
    params['batchsize'] = 256
    params['learning_rate'] = 0.0003
    params['restore'] = False
    params['latent_size'] = 64
    params['enc_sizes'] = esize 
    params['dec_sizes'] = dsize
    params['droprate'] = 0.85
    params['stdev'] = 0.04
    params['denoise'] = False
    params['slam'] = 0

    return params

def cluster(niterations, datadir=None, params=None,
            display=False, display_int=500, report_int=50, title="AE_cluster"):
    
    if datadir is None:
        datadir = '/media/cjw/Data/cyto/mmCompensatedTifs/'
        
    if params is None:
        params = setup()

    tf1 = time.strftime("%Y-%m-%d-%H-%M-%S")
    tf2 = time.strftime("checkpoint-%Y-%m-%d-%H-%M-%S")
 
    savedir = '/media/cjw/Data/cyto/Checkpoints/' + tf1 + "_" + title + "/"
    savedir += tf2 + '/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    #os.mkdir(savedir)
    savename = savedir +  "autoencoder-{:d}x".format(params['latent_size'])

    print("Using data from:", datadir)
    print("Saving checkpoints to:", savename)
 
    
    mmdict, n_all_images = aae.create_mmdict(datadir)
    df = aae.create_df(mmdict)
    
    print(list(mmdict.keys()))
    print(df.head())
    w = params['width']

    tf.reset_default_graph()

    images = tf.placeholder(tf.float32, (None, w, w, params['nchannels'])) 
    sample_z = tf.placeholder(tf.float32, (None, params['latent_size']))
    y = tf.placeholder(tf.float32, (None, params['nclusters']))
    
    vn = aae_clustering(params)

    encoder_z, encoder_y = vn.create_encoder(images, True)
    sdecoder, decoder =vn.create_decoder(encoder_z, encoder_y, True)
    #vn.create_discriminator(sample_z)
    vn.reconstruction_loss(images, decoder, sdecoder, encoder_y)
    vn.discriminator_loss(sample_z, encoder_z)
    vn.cluster_loss(y, encoder_y) 
    ae, d, g, ee = vn.opt()

    saver = tf.train.Saver()

    test_images = utils.get_sample(mmdict, df, 16,
                                   params['width'], 
                                   params['nchannels'],
                                   channels=params['channels'])

    test_images2 = utils.get_sample(mmdict, df, 16,
                               params['width'], 
                               params['nchannels'],
                               channels=params['channels'])

    test_oh = one_hot_batch(params['batchsize'], params['nclusters'])
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step_counter = 0
        for i in range(niterations):
            #print(i)
            batch_images = utils.get_sample(mmdict, df, params['batchsize'],
                                            params['width'], 
                                            params['nchannels'],
                                            channels=params['channels'])
            
            batch_z = np.random.normal(0, 1, size=(params['batchsize'],
                                                   params['latent_size']))

            batch_y = one_hot_batch(params['batchsize'], params['nclusters'])
            
            ##batch_z = np.random.uniform(-10, 50, size=(params['batchsize'],
            ##                                       params['latent_size']))
            sess.run([ae, ee], feed_dict={images:batch_images})
            sess.run(d, feed_dict={images:batch_images,
                                   sample_z:batch_z, y:batch_y})
#            sess.run(c, feed_dict={images:batch_images, y:batch_y})
            sess.run(g, feed_dict={images:batch_images,
                                   sample_z:batch_z, y:batch_y})
#            sess.run(gc, feed_dict={images:batch_images, y:batch_y})
            sess.run([ae, ee], feed_dict={images:batch_images})

            step_counter += 1
            if i % report_int == 0:
                xd = vn.d_loss.eval({images:batch_images, sample_z:batch_z})
                xg = vn.gen_loss.eval({images:batch_images, sample_z:batch_z})
                xr = vn.rloss.eval({images:batch_images})
                xe = vn.eloss.eval({images:batch_images})
                xc = vn.c_loss.eval({images:batch_images, y:batch_y})
                xcg = vn.c_gen_loss.eval({images:batch_images, y:batch_y})
                txd = vn.d_loss.eval({images:test_images, sample_z:batch_z})
                txg = vn.gen_loss.eval({images:test_images, sample_z:batch_z})
                txr = vn.rloss.eval({images:test_images})
                txc = vn.c_loss.eval({images:test_images, y:test_oh}) 
                txcg = vn.c_gen_loss.eval({images:test_images, y:test_oh})
                
                test_image = np.expand_dims(test_images[ i % 16],axis=0)
                encoded_z = encoder_z.eval({images:test_images})
                encoded_y = encoder_y.eval({images:test_images})
                encoded_y_b = encoder_y.eval({images:batch_images})

                decoded = sdecoder.eval({encoder_z:encoded_z, encoder_y:encoded_y})
                decoded = np.squeeze(decoded[i % 16])
                xspace =  encoder_z.eval({images:batch_images})
                print(i, xd, xg, xr, xe, xc, xcg)
                print(i, txd, txg, txr, txc, txcg)                
                print(i, np.round(np.sum(encoded_y, axis=0)))
                print(i, np.round(np.sum(encoded_y_b, axis=0)))
                
                encoded_y2 = encoder_y.eval({images:test_images2})                
                print(np.argmax(encoded_y, axis=1))
                print(np.argmax(encoded_y2, axis=1))
                #print(batch_y)
                #print(encoded_y)
            if display and i % display_int == 0:
                plt.figure(figsize=(8,2))
                plt.subplot(2,6,1)
                plt.imshow(np.squeeze(test_image)[:,:,0])
                plt.subplot(2,6,2)
                plt.imshow(np.squeeze(test_image)[:,:,1])
                plt.subplot(2,6,3)
                plt.imshow(np.squeeze(test_image)[:,:,2])
                plt.subplot(2,6,4)
                plt.imshow(np.squeeze(test_image)[:,:,3])
                plt.subplot(2,6,7)
                plt.imshow(decoded[:,:,0])
                plt.subplot(2,6,8)
                plt.imshow(decoded[:,:,1])
                plt.subplot(2,6,9)
                plt.imshow(decoded[:,:,2])
                plt.subplot(2,6,10)
                plt.imshow(decoded[:,:,3])
                plt.subplot(2,3,3)
                plt.hist(batch_z.reshape((-1)), bins=25)
                plt.subplot(2,3,6)
                plt.hist(xspace.reshape((-1)), bins=25)
                plt.show()
                
            if i % 1000 == 0:
                saver.save(sess, savename, global_step=step_counter)
                
        saver.save(sess, savename, global_step=step_counter)  
        
    print("Done")
