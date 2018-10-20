from data_utils import SortedNumberGenerator
from os.path import join, basename, dirname, exists
import datetime
import argparse
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, TimeDistributed
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import RMSprop

from tqdm import tqdm
import random
from functools import partial
import numpy as np
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os
if not os.path.exists('models/'):
    os.makedirs('models/')

if not os.path.exists('fig'):
    os.makedirs('fig')


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def __init__(self, batch_size):
        super(RandomWeightedAverage, self).__init__()
        self.batch_size = batch_size
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def time_distributed(model, inputs):

    ''' alternative for keras.layer.TimeDistributed '''

    n_times = inputs.shape[1]
    outputs = []
    for i in range(n_times):
        outputs.append(model(inputs[:,i,...]))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class WGANGP():
    def __init__(self, args, encoder, cpc):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 3 if args.color else 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = args.code_size
        self.predict_terms = 4#args.predict_terms
        #self.terms = 4

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim, ))
        pred = Input(shape=(self.latent_dim, ))

        #z_disc_con = keras.layers.Lambda(lambda i: K.concatenate([i[0], i[1]], axis=-1))([z_disc, pred])
        z_disc_con = concatenate([z_disc, pred],-1)

        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc_con)


        ########## TimeDistributed doesn't work for self.generator ##########

        # z_discxx = Input(shape=(4, self.latent_dim))
        # tdistri = TimeDistributed(self.generator, input_shape=(4, self.latent_dim))
        # #fake_imgsx = tdistri(z_discxx)
        # imgsx = Input(shape=(4, self.img_rows, self.img_cols, self.channels))
        # out_imgs = TimeDistributed(self.critic, input_shape=(4, self.img_rows, self.img_cols, self.channels))(imgsx)
        # Noise input


        # z_disc = Input(shape=(self.predict_terms, self.latent_dim))
        # pred = Input(shape=(self.predict_terms, self.latent_dim))
        # z_disc_con = concatenate([z_disc, pred],-1)
        # # Generate image based of noise (fake sample)
        # fake_img = time_distributed(self.generator,z_disc_con)





        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(args.batch_size)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc, pred],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        pred = Input(shape=(self.latent_dim,))

        #z_gen_con = keras.layers.Lambda(lambda i: K.concatenate([i[0], i[1]], axis=-1))([z_gen, pred])
        z_gen_con = concatenate([z_gen, pred],-1)

        # Generate images based of noise
        img = self.generator(z_gen_con)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model

        z = encoder(img)

        tz = keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(z)
        tpred = keras.layers.Lambda(lambda z: K.expand_dims(z, 1))(pred)
        cpc_loss = cpc([tpred, tz])

        self.generator_model = Model(inputs=[z_gen, pred], outputs=[valid, cpc_loss, img])
        self.generator_model.compile(loss=[self.wasserstein_loss, 'binary_crossentropy', None], loss_weights=[args.gan_weight, 1.0, 0.0], optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim * 2))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim * 2,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)





def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings '''

    # x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x

def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)

    return x


def network_prediction(context, code_size, predict_terms, name='z'):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name=name+'_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs

        # dot_product = K.mean(20 - K.abs(y_encoded  - preds) * (y_encoded - preds), axis=-1)
        # dot_product = K.mean(K.l2_normalize(y_encoded, axis=-1) * K.l2_normalize(preds, axis=-1), axis=-1)
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension
    
        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)









def network_cpc(args, image_shape, terms, predict_terms, code_size, learning_rate):

    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)

    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input)

    # Loss
    cpc_layer = CPCLayer()
    dot_product_probs = cpc_layer([preds, y_encoded])


    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=[preds, dot_product_probs])


    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=[None, 'binary_crossentropy'],
        metrics=['binary_accuracy']
    )
    cpc_model.summary()


    encoder_model.trainable = False
    cpc_layer.trainable = False

    return (cpc_model, encoder_model, cpc_layer)


def train_model(args, batch_size, output_dir, code_size, lr=1e-4, terms=4, predict_terms=4, image_size=28, color=False):

    # Prepare data
    train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=True)

    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=True)

    channel = 3 if color else 1
    model, encoder, cpc = network_cpc(args, image_shape=(image_size, image_size, channel), terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=lr)
    gan = WGANGP(args, encoder, cpc)

    session = tf.Session()

    #All placeholders for Tensorboard
    train_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
    m1_1 = tf.summary.scalar('train/loss', train_loss_ph)

    val_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
    m1_2 = tf.summary.scalar('val/loss', val_loss_ph)

    train_acc_ph = tf.placeholder(shape=(), dtype=tf.float32)
    m1_3 = tf.summary.scalar('train/acc', train_acc_ph)

    val_acc_ph = tf.placeholder(shape=(), dtype=tf.float32)
    # val_acc_ph = K.print_tensor(val_acc_ph, message='val_acc_ph')
    m1_4 = tf.summary.scalar('val/acc', val_acc_ph)

    g_train_loss_critic_ph = tf.placeholder(shape=(), dtype=tf.float32)
    g_train_loss_critic_ph = tf.Print(g_train_loss_critic_ph, [g_train_loss_critic_ph])
    m2_1 = tf.summary.scalar('train/generator/critic_loss', g_train_loss_critic_ph)

    g_train_loss_cpc_ph = tf.placeholder(shape=(), dtype=tf.float32)
    m2_2 = tf.summary.scalar('train/generator/cpc_loss', g_train_loss_cpc_ph)

    d_train_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
    m2_3 = tf.summary.scalar('train/dis/loss', d_train_loss_ph)

    g_test_loss_critic_ph = tf.placeholder(shape=(), dtype=tf.float32)
    m3_1 = tf.summary.scalar('test/generator/critic_loss', g_test_loss_critic_ph)

    g_test_loss_cpc_ph = tf.placeholder(shape=(), dtype=tf.float32)
    m3_2 = tf.summary.scalar('test/generator/cpc_loss', g_test_loss_cpc_ph)

    d_test_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
    m3_3 = tf.summary.scalar('test/dis/loss', d_test_loss_ph)

    raw_train_image_ph = tf.placeholder(shape=(1,28,28,channel), dtype=tf.float32)
    # raw_train_image_ph = K.print_tensor(raw_train_image_ph, message='raw_train_image_ph')
    raw_train_image_ph = tf.cast((tf.clip_by_value(raw_train_image_ph, -1, 1) + 1) * 127, tf.uint8)
    m2_4 = tf.summary.image('train/raw', raw_train_image_ph)

    recon_train_image_ph = tf.placeholder(shape=(1,28,28,channel), dtype=tf.float32)
    recon_train_image_ph = tf.Print(recon_train_image_ph, [recon_train_image_ph])
    recon_train_image_ph = tf.cast((tf.clip_by_value(recon_train_image_ph, -1, 1) + 1) * 127, tf.uint8)
    m2_5 = tf.summary.image('train/recon', recon_train_image_ph)

    raw_test_image_ph = tf.placeholder(shape=(1,28,28,channel), dtype=tf.float32)
    raw_test_image_ph = tf.Print(raw_test_image_ph, [raw_test_image_ph])
    raw_test_image_ph = tf.cast((tf.clip_by_value(raw_test_image_ph, -1, 1) + 1) * 127, tf.uint8)
    m3_4 = tf.summary.image('test/raw', raw_test_image_ph)

    recon_test_image_ph = tf.placeholder(shape=(1,28,28,channel), dtype=tf.float32)
    recon_test_image_ph = tf.Print(recon_test_image_ph, [recon_test_image_ph])
    recon_test_image_ph = tf.cast((tf.clip_by_value(recon_test_image_ph, -1, 1) + 1) * 127, tf.uint8)
    m3_5 = tf.summary.image('test/recon', recon_test_image_ph)

    merged1 = tf.summary.merge([m1_1, m1_2, m1_3, m1_4])
    merged2 = tf.summary.merge([m2_1, m2_2, m2_3, m2_4, m2_5])
    merged3 = tf.summary.merge([m3_1, m3_2, m3_3, m3_4, m3_5])

    writer = tf.summary.FileWriter('./logs/train_' + args.name + '_' +datetime.datetime.now().strftime('%d_%H-%M-%S '))


    if len(args.load_name) > 0:
        model = keras.models.load_model(join(output_dir, args.load_name))

    else:
        print('Start Training CPC')
        for epoch in range(args.cpc_epochs // 1):

            avg0, avg1, avg2, avg3 = [], [], [], []
            for i in range(len(train_data) // 1):
                train_batch = next(train_data)
                train_result = model.train_on_batch(train_batch[0][:2], train_batch[1])
                avg0.append(train_result[0])
                avg2.append(train_result[2])
                sys.stdout.write(
                    '\r Epoch {}: training[{} / {}]'.format(epoch, i, len(train_data)))

            for i in range(len(validation_data) // 1):
                validation_batch = next(validation_data)
                validation_result = model.test_on_batch(validation_batch[0][:2], validation_batch[1])
                avg1.append(validation_result[0])
                avg3.append(validation_result[2])
                sys.stdout.write(
                    '\r Epoch {}: validation[{} / {}]'.format(epoch, i, len(validation_data)))

            print('\n%s' % ('-' * 40))
            print('Train loss: %.2f, Accuracy: %.2f \t Validation loss: %.2f, Accuracy: %.2f' % (100.0 * np.mean(avg0), 100.0 * np.mean(avg2), 100.0 * np.mean(avg1), 100.0 * np.mean(avg3)))
            print('%s' % ('-' * 40))

            summary = session.run(merged1,
                                feed_dict={train_loss_ph: np.mean(avg0),
                                            val_loss_ph: np.mean(avg1),
                                            train_acc_ph: np.mean(avg2),
                                            val_acc_ph: np.mean(avg3)
                                            })

            writer.add_summary(summary, epoch)
            writer.flush()
        

        # Saves the model
        # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
        model.save(join(output_dir, 'cpc_' + args.name + '.h5'))

        # Saves the encoder alone
        encoder = model.layers[1].layer
        encoder.save(join(output_dir, 'encoder_' + args.name + '.h5'))

    print('\nStart Training GAN')
    # Adversarial ground truths
    valid = -np.ones((batch_size, 1))
    fake =  np.ones((batch_size, 1))
    cpc_true = np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

    # self.critic_model = Model(inputs=[real_img, z_disc, pred],
    #                     outputs=[valid, fake, validity_interpolated])
    # self.generator_model = Model(inputs=[z_gen, pred], outputs=[valid, cpc_loss, img])

    for epoch in range(args.gan_epochs):


        avg0, avg1, avg2 = [], [], []
        for i in range(len(train_data) // 1):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            train_batch = next(train_data)

            preds, _ = model.predict(train_batch[0][:2], batch_size=batch_size)

            for _ in range(5):
                print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
                noise = np.random.normal(0, 1, (batch_size, args.code_size))
                image = train_batch[0][0][:, random.randint(0,3)]
                d_loss = gan.critic_model.train_on_batch([image, noise, preds[:,0]], [valid, fake, dummy])
                avg2.append(d_loss[0])


            image = train_batch[0][2][:, 0]
            g_loss = gan.generator_model.train_on_batch([noise, preds[:, 0]], [valid, cpc_true])
            avg0.append(g_loss[1])
            avg1.append(g_loss[2])

            _, _, recon = gan.generator_model.predict([noise, preds[:, 0]], batch_size=batch_size)
            sys.stdout.write(
                '\r Epoch {}: train[{} / {}]'.format(epoch, i, len(train_data)))

        print('\n%s' % ('-' * 40))
        print('Training -- Generator Critic loss: %.2f, Generator CPC loss: %.2f, Discriminator: %.2f' % (100.0 * np.mean(avg0), 100.0 * np.mean(avg1), 100.0 * np.mean(avg2)))

        summary = session.run(merged2,
                    feed_dict={g_train_loss_critic_ph: np.mean(avg0),
                                g_train_loss_cpc_ph: np.mean(avg1),
                                d_train_loss_ph: np.mean(avg2),
                                raw_train_image_ph: image[:1],
                                recon_train_image_ph: recon[:1]
                            })

        writer.add_summary(summary, epoch)
        writer.flush()

        avg0, avg1, avg2 = [], [], []

        for i in range(len(validation_data) // 1):
            validation_batch = next(validation_data)


            preds, _ = model.predict(validation_batch[0][:2], batch_size=batch_size)
            noise = np.random.normal(0, 1, (batch_size, args.code_size))
            image = validation_batch[0][2][:, 0]
            d_loss = gan.critic_model.test_on_batch([image, noise, preds[:,0]], [valid, fake, dummy])
            avg2.append(d_loss[0])
            g_loss = gan.generator_model.test_on_batch([noise, preds[:, 0]], [valid, cpc_true])
            _, _, recon = gan.generator_model.predict([noise, preds[:, 0]], batch_size=batch_size)
            avg0.append(g_loss[1])
            avg1.append(g_loss[2])

            sys.stdout.write(
                '\r Epoch {}: validation[{} / {}]'.format(epoch, i, len(validation_data)))


        print('\n')
        print('Validation -- Generator Critic loss: %.2f, Generator CPC loss: %.2f, Discriminator: %.2f' % (100.0 * np.mean(avg0), 100.0 * np.mean(avg1), 100.0 * np.mean(avg2)))
        print('%s' % ('-' * 40))

        summary = session.run(merged3,
                        feed_dict={g_test_loss_critic_ph: np.mean(avg0),
                                    g_test_loss_cpc_ph: np.mean(avg1),
                                    d_test_loss_ph: np.mean(avg2),
                                    raw_test_image_ph: image[:1],
                                    recon_test_image_ph: recon[:1]
                                })

        writer.add_summary(summary, epoch)
        writer.flush()

        if epoch % 10 == 0:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            if image.shape[-1] == 1:
                image = np.concatenate([image, image, image], axis=-1)
            if recon.shape[-1] == 1:
                recon = np.concatenate([recon, recon, recon], axis=-1)
            ax1.imshow(image[0] * 0.5 + 0.5)
            ax2.imshow(recon[0] * 0.5 + 0.5)
            plt.savefig('fig/' + args.name + '_epoch' + str(epoch) + '.png')
    
    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    gan.generator_model.save(join(output_dir, 'generator_' + args.name + '.h5'))

    # Saves the encoder alone
    gan.critic_model.save(join(output_dir, 'dis_' + args.name + '.h5'))

    # for i in range(batch_size):
    #     # print(image[0].shape)
    #     #plot
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1, 2, 1)
    #     ax2 = fig.add_subplot(1, 2, 2)
    #     if image.shape[-1] == 1:
    #         image = np.concatenate([image, image, image], axis=-1)
    #     if recon.shape[-1] == 1:
    #         recon = np.concatenate([recon, recon, recon], axis=-1)
    #     ax1.imshow(image[i] * 0.5 + 0.5)
    #     ax2.imshow(recon[i] * 0.5 + 0.5)
    #     plt.show()

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='CPC')
    argparser.add_argument(
        '--name',
        default='cpc',
        help='name')
    argparser.add_argument(
        '--load-name',
        default='',
        help='loadpath')
    argparser.add_argument(
        '-e', '--cpc-epochs',
        default=0,
        type=int,
        help='cpc epochs')
    argparser.add_argument(
        '-g', '--gan-epochs',
        default=1,
        type=int,
        help='gan epochs')
    argparser.add_argument(
        '--lr',
        default=1e-3,
        type=float,
        help='Learning rate')
    argparser.add_argument(
        '--gan-weight',
        default=1.0,
        type=float,
        help='GAN Weight')
    argparser.add_argument('--doctor', action='store_true', default=False, help='Doctor')
        
    args = argparser.parse_args()

    if args.doctor:
        predict_terms = 1
    else:
        predict_terms = 1

    args.predict_terms = predict_terms
    args.code_size = 10
    args.batch_size = 128
    args.color = False

    train_model(
        args, 
        batch_size=args.batch_size,
        output_dir='models/',
        code_size=args.code_size,
        lr=args.lr,
        terms=4,
        predict_terms=predict_terms,
        image_size=28,
        color=args.color
    ) 

