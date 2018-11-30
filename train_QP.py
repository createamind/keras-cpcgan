
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
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, Lambda, merge
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, TimeDistributed, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import RMSprop, Adam
from keras.utils import multi_gpu_model

from data_utils1 import SortedNumberGenerator
from data_utils_generated import GeneratedNumberGenerator
from data_video import VideoDataGenerator
from os.path import join, basename, dirname, exists

from tqdm import tqdm
import random
from functools import partial
import numpy as np
import sys, time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
if not os.path.exists('images/args.name'):
    os.makedirs('images/args.name')
#if not os.path.exists('date/args.name'):
#    os.makedirs('date/args.name')
# if not os.path.exists('models/args.name'):
#     os.makedirs('models/args.name')
if not os.path.exists('models/'):
    os.makedirs('models/')



def time_distributed(model, inputs, per_inputs=None):
    ''' alternative for keras.layer.TimeDistributed '''

    # per_inputs: other inputs 
    if per_inputs is None:
        outputs = [model(Lambda(lambda data: data[:,i])(inputs)) for i in range(inputs.shape[1])]
    else:
        outputs = [model([Lambda(lambda data: data[:,i])(inputs), per_inputs]) for i in range(inputs.shape[1])]

    if len(outputs) == 1:
        output = Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])   # if the length of the list is 1, get the element and expand its dimension.
    else:
        output = Lambda(lambda x: K.stack(x, axis=1))(outputs)            # if the length of the list is larger than 1, stack the elements along the axis.

    return output


class GANQP_CPC():
    def __init__(self, args, pc, encoder, cpc_sigma):
        self.img_rows = args.image_size
        self.img_cols = args.image_size
        self.channels = 3 if args.color else 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = args.code_size
        self.predict_terms = args.predict_terms
        self.terms = args.terms
        self.dataset = args.dataset

        pc.trainable = False
        encoder.trainable = False

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator(self.img_rows)
        self.critic = self.build_critic(self.img_rows)

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        y_img = Input(shape=(self.predict_terms, self.img_rows, self.img_cols, self.channels))

        # Noise input
        z_disc = Input(shape=(self.predict_terms, self.latent_dim))

        x_img = Input(shape=(self.terms, self.img_rows, self.img_cols, self.channels))
        x_img_last = Lambda(lambda x_img: x_img[:, -1])(x_img)
        pred = pc(x_img)  # pred = W_k * C_t

        z_disc_con = concatenate([z_disc, pred],-1)

        # Generate image based of noise (fake sample)
        fake_img = time_distributed(self.generator, z_disc_con, per_inputs=x_img_last)
        self.gen_model = Model(inputs=[x_img, z_disc], outputs=[fake_img])

        # Discriminator determines validity of the real and fake images
        x_fake_score = time_distributed(self.critic,fake_img)
        x_real_score = time_distributed(self.critic,y_img)
        self.critic_model = Model(inputs=[x_img, y_img, z_disc],
                            outputs=[x_real_score, x_fake_score])
        d_loss = x_real_score - x_fake_score
        d_loss = d_loss[...,0]
        d_norm = 10 * K.mean(K.abs(y_img - fake_img), axis=[2, 3, 4])
        d_loss = K.mean(- d_loss + 0.5 * d_loss ** 2 / d_norm)

        self.critic_model.add_loss(d_loss)
        self.critic_model.compile(optimizer=Adam(2e-4, 0.5))


        self.critic_model.summary()


        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.predict_terms, self.latent_dim))
        #pred = Input(shape=(self.predict_terms, self.latent_dim))   # pred = W_k * C_t

        z_gen_con = concatenate([z_gen, pred],-1)

        # Generate images based of noise
        img_gen = time_distributed(self.generator, z_gen_con, per_inputs=x_img_last)
        # Discriminator determines validity
        img_gen_score = time_distributed(self.critic,img_gen)
        # Defines generator model

        z = time_distributed(encoder,img_gen)

        if args.cpc_weight == 0.0:
            cpc_loss = Lambda(lambda x: x[:,:1,0])(z_gen)
        else:
            cpc_loss = cpc_sigma([pred, z])


        self.generator_model = Model(inputs=[x_img, z_gen], outputs=[img_gen_score, cpc_loss])

        cpc_loss = K.mean(cpc_loss)
        g_loss = K.mean(-img_gen_score)
        g_cpc_loss = g_loss * args.gan_weight + cpc_loss * args.cpc_weight

        self.generator_model.add_loss(g_cpc_loss)
        self.generator_model.compile(optimizer=Adam(2e-4, 0.5))


    # def build_generator(self, image_size):

    #     model = Sequential()

    #     model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim * 2))
    #     model.add(Reshape((7, 7, 128)))
    #     model.add(UpSampling2D())
    #     model.add(Conv2D(128, kernel_size=4, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Activation("relu"))
    #     model.add(UpSampling2D())

    #     if image_size >= 64:
    #         model.add(Conv2D(128, kernel_size=4, padding="same"))
    #         model.add(BatchNormalization(momentum=0.8))
    #         model.add(Activation("relu"))
    #         model.add(UpSampling2D())
    #     if image_size >= 112:
    #         model.add(Conv2D(128, kernel_size=4, padding="same"))
    #         model.add(BatchNormalization(momentum=0.8))
    #         model.add(Activation("relu"))
    #         model.add(UpSampling2D())
    #     if image_size >= 224:
    #         model.add(Conv2D(64, kernel_size=4, padding="same"))
    #         model.add(BatchNormalization(momentum=0.8))
    #         model.add(Activation("relu"))
    #         model.add(UpSampling2D())

    #     model.add(Conv2D(64, kernel_size=4, padding="same"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Activation("relu"))
        
    #     model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
    #     model.add(Activation("tanh"))

    #     #model.summary()

    #     noise = Input(shape=(self.latent_dim * 2,))
    #     img = model(noise)

    #     return Model(noise, img)
 


    def build_generator(self, image_size):

        # model = Sequential()

        # model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim * 2))
        # model.add(Reshape((7, 7, 128)))
        # model.add(UpSampling2D())
        # model.add(Conv2D(128, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(UpSampling2D())

        # if image_size >= 64:
        #     model.add(Conv2D(128, kernel_size=4, padding="same"))
        #     model.add(BatchNormalization(momentum=0.8))
        #     model.add(Activation("relu"))
        #     model.add(UpSampling2D())
        # if image_size >= 112:
        #     model.add(Conv2D(128, kernel_size=4, padding="same"))
        #     model.add(BatchNormalization(momentum=0.8))
        #     model.add(Activation("relu"))
        #     model.add(UpSampling2D())
        # if image_size >= 224:
        #     model.add(Conv2D(64, kernel_size=4, padding="same"))
        #     model.add(BatchNormalization(momentum=0.8))
        #     model.add(Activation("relu"))
        #     model.add(UpSampling2D())

        # model.add(Conv2D(64, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        
        # model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        # model.add(Activation("tanh"))

        #model.summary()

        noise = Input(shape=(self.latent_dim * 2,))
        refimg = Input(shape=self.img_shape)

        conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(refimg)
        conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        if self.dataset == 'kth':
            middle = Reshape(target_shape=(7 * 7 * 8,))(drop5)
            middle = Dense(units=self.latent_dim, activation='relu')(middle)
            middle = concatenate([middle, noise], axis=-1)
            middle = Dense(units=8 * 7 * 7, activation='relu')(middle)
            middle = Reshape(target_shape=(7, 7, 8,))(middle)
        else:
            middle = Reshape(target_shape=(14 * 14 * 8,))(drop5)
            middle = Dense(units=self.latent_dim, activation='relu')(middle)
            middle = concatenate([middle, noise], axis=-1)
            middle = Dense(units=8 * 14 * 14, activation='relu')(middle)
            middle = Reshape(target_shape=(14, 14, 8,))(middle)

        up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(middle))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'tanh')(conv9)

        model = Model(input = [noise, refimg], output = conv10)

        model.summary()

        return model


    def build_critic(self, image_size):

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

        if image_size >= 64:
            model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
        if image_size >= 112:
            model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
        if image_size >= 224:
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

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)



'''
This module describes the contrastive predictive coding model from DeepMind
'''
def network_encoderx(x, code_size, image_size):

    ''' Define the network mapping images to embeddings '''

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    if image_size >= 64:
        x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

    if image_size >= 112:
        x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

    if image_size >= 224:
        x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x


def network_encoder(x, code_size, image_size):

    ''' Define the network mapping images to embeddings '''

    #x = keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, activation='linear')(x)
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    if image_size >= 64:
        x = keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

    if image_size >= 112:
        x = keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

    if image_size >= 224:
        x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x


def network_autoregressive(x, context_size):

    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=context_size, return_sequences=False, name='ar_context')(x)

    return x

def network_prediction_rnn(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple predictions '''

    context_size = context.shape[-1].value

    x = RepeatVector(predict_terms, input_shape=(context_size,))(context)
    x = LSTM(context_size, return_sequences=True)(x)
    output = Dense(units=code_size, activation="linear")(x)

    return output



def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

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
        dot_product = K.mean(y_encoded * preds, axis=-1)

        # Keras loss functions take probabilities
        dot_product = K.sigmoid(dot_product)
        dot_product_probs = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension
    

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def network_cpc(image_shape, terms, predict_terms, code_size, context_size, learning_rate):

    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)
    encoder_output = network_encoder(encoder_input, code_size, image_shape[0])
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))
    x_encoded = TimeDistributed(encoder_model)(x_input)
    context = network_autoregressive(x_encoded, context_size)

    #preds = network_prediction_rnn(context, code_size, predict_terms)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    y_encoded = TimeDistributed(encoder_model)(y_input)

    # Loss
    cpc_layer = CPCLayer()
    dot_product_probs = cpc_layer([preds, y_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)
    pc_model = keras.models.Model(inputs=x_input, outputs=preds)

    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    cpc_model.summary()

    return (cpc_model, pc_model, encoder_model)




def train_model(args, batch_size, output_dir, code_size, context_size, lr=1e-4, terms=4, predict_terms=4, image_size=28, color=False, dataset='ucf'):

    if dataset == 'ucf' or dataset == 'walking' or dataset == 'kth':
        # Prepare data
        train_data = VideoDataGenerator(batch_size=batch_size, subset='train', terms=terms,
                                        positive_samples=batch_size  //2, predict_terms=predict_terms,
                                        image_size=image_size, color=color, rescale=True, dataset=dataset)

        validation_data = VideoDataGenerator(batch_size=batch_size, subset='val', terms=terms,
                                                positive_samples=batch_size  // 2, predict_terms=predict_terms,
                                                image_size=image_size, color=color, rescale=True, dataset=dataset)

    elif dataset == 'generated':
        train_data = GeneratedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                        positive_samples=batch_size // 2, predict_terms=predict_terms,
                                        image_size=image_size, color=color, rescale=True)

        validation_data = GeneratedNumberGenerator(batch_size=batch_size, subset='val', terms=terms,
                                                positive_samples=batch_size // 2, predict_terms=predict_terms,
                                                image_size=image_size, color=color, rescale=True)

    elif dataset == 'mnist':
        train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                        positive_samples=batch_size // 2, predict_terms=predict_terms,
                                        image_size=image_size, color=color, rescale=True)

        validation_data = SortedNumberGenerator(batch_size=batch_size, subset='val', terms=terms,
                                                positive_samples=batch_size // 2, predict_terms=predict_terms,
                                                image_size=image_size, color=color, rescale=True)
    else:
        raise NotImplementedError


    channel = 1 * (3 if color else 1)
    model, pc, encoder = network_cpc(image_shape=(image_size, image_size, channel), terms=terms, predict_terms=predict_terms, code_size=code_size, context_size=context_size, learning_rate=lr)

    if args.plan > 1:
        print('Loading CPC models')
        pc = keras.models.load_model(join(args.load_name, 'pc_' + args.name + '_best.h5'))#,custom_objects={'CPCLayer': CPCLayer})
        encoder = keras.models.load_model(join(args.load_name, 'encoder_' + args.name + '_best.h5'))
        model = keras.models.load_model(join(args.load_name, 'cpc_' + args.name + '_best.h5'),custom_objects={'CPCLayer': CPCLayer})
    if args.plan == 3:
        print('Evaluating CPC models')
        #model = keras.models.load_model(join(args.load_name, 'cpc_' + args.name + '_best.h5'),custom_objects={'CPCLayer': CPCLayer})
        eval_results = model.evaluate_generator(
            generator=validation_data,
            steps=len(validation_data),
            verbose=1
        )
        print("eval_results:", eval_results)
    else:
        if args.plan == 1:
            print('Start Training CPC')
        else:
            print('Resume Training CPC')

        #model = keras.models.load_model(join("handwaving_best_model", 'cpc_' + args.name + '_best.h5'),custom_objects={'CPCLayer': CPCLayer})

        # Callbacks

        class SaveModel(keras.callbacks.Callback):
            loss_min = 1000
            def on_epoch_end(self, epoch, logs={}):
                loss_now = (logs.get('loss') + logs.get('val_loss')) / 2
                if loss_now < SaveModel.loss_min and epoch > 0:
                    SaveModel.loss_min = loss_now
                    print(SaveModel.loss_min)
                    model.save(join(output_dir, 'cpc_' + args.name + '_best.h5'))#'_%d.h5'% epoch))
                    pc.save(join(output_dir, 'pc_' + args.name + '_best.h5'))#'_%d.h5'% epoch))
                    encoder.save(join(output_dir, 'encoder_' + args.name + '_best.h5'))#'_%d.h5'% epoch))
        save_model = SaveModel()

        callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4), save_model]


        # Trains the model
        model.fit_generator(
            generator=train_data,
            steps_per_epoch=len(train_data),
            validation_data=validation_data,
            validation_steps=len(validation_data),
            epochs=args.cpc_epochs,
            verbose=1,
            callbacks=callbacks
        )

        # Saves the model
        # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
        #model.save(join('models', args.name, 'cpc.h5'))   #'images', args.name, 'epoch%d.png' % epoch))

        # Saves the encoder alone
        #pc.save(join(output_dir, 'pc.h5'))
        #encoder.save(join(output_dir, 'encoder.h5'))
        # time.sleep(3)
        # pc = keras.models.load_model(join('cpc_models', 'pc.h5'))
        # encoder = keras.models.load_model(join('cpc_models', 'encoder.h5'))

        # Saves the model
        # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
        model.save(join(output_dir, 'cpc_' + args.name + '.h5'))
        pc.save(join(output_dir, 'pc_' + args.name + '.h5'))
        # Saves the encoder alone
        #encoder = model.layers[1].layer
        encoder.save(join(output_dir, 'encoder_' + args.name + '.h5'))

    print("Start Training GAN")
    cpc_sigma = CPCLayer()
    gan = GANQP_CPC(args, pc, encoder, cpc_sigma)
    gen_model = gan.gen_model

    for epoch in range(args.gan_epochs):
        #print(len(train_data))
        for i in range(len(train_data) // 1): # len(train_data): number of batches
            #print("xxxxxxxxxxxxxxx")
            #t0 = time.time()

            [x_img, y_img], labels = next(train_data)

            #t1 = time.time()

            for _ in range(2):
                #print("yyyyyyyyyyyyyyyyyyy")
                #[x_img, y_img], labels = next(train_data)
                noise = np.random.normal(0, 1, (batch_size, predict_terms, args.code_size))
                d_loss = gan.critic_model.train_on_batch([x_img, y_img, noise], None)

            #t2 = time.time()
            g_loss = gan.generator_model.train_on_batch([x_img, noise], None)

            #t3 = time.time()
            #print("time elapsed:", t1 - t0, t2 - t1, t3 - t2)
            sys.stdout.write(
                '\r Epoch {}: training[{} / {}]'.format(epoch, i, len(train_data)))

        ###################  Validation   ###################

        print("\nepoch: ", epoch, "\nd_loss: ", d_loss, "\ng_loss: ", g_loss)

        rows = min(5, args.batch_size)
        init_img = x_img[0:rows, ...]
        init_img = (init_img + 1)*0.5
        gen_img = gen_model.predict([x_img[0:rows,...],noise[0:rows,...]])
        gen_img = (gen_img + 1)*0.5

        imgs = np.concatenate((init_img,gen_img), axis=1)

        if imgs.shape[-1] == 1:
            imgs = np.concatenate([imgs, imgs, imgs], axis=-1)

        cols = args.terms + args.predict_terms
        fig, axs = plt.subplots(max(2,rows),cols)
        for i in range(rows):
            for j in range(cols):
                axs[i,j].imshow(imgs[i,j] )
                axs[i,j].axis('off')

        if not os.path.exists('images/'):
            os.makedirs('images/')
        if not os.path.exists(os.path.join('images', args.name)):
            os.makedirs(os.path.join('images', args.name))
        fig.savefig(os.path.join('images', args.name, 'epoch%d.png' % epoch), dpi=1500)
        gan.generator_model.save(join(output_dir, 'generator_' + args.name + '.h5'))
        gan.critic_model.save(join(output_dir, 'dis_' + args.name + '.h5'))

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
        help='load name (use "models" if wanted)')
    argparser.add_argument(
        '--dataset',
        default='ucf',
        help='ucf[default], walking, mnist, generated')
    argparser.add_argument(
        '-e', '--cpc-epochs',
        default=500,
        type=int,
        help='cpc epochs')
    argparser.add_argument(
        '-g', '--gan-epochs',
        default=1000,
        type=int,
        help='gan epochs')
    argparser.add_argument(
        '--predict-terms',
        default=4,
        type=int,
        help='predict-terms')
    argparser.add_argument(
        '--batch-size',
        default=8,
        type=int,
        help='batch_size')
    argparser.add_argument(
        '--terms',
        default=4,
        type=int,
        help='terms')
    argparser.add_argument(
        '--code-size',
        default=64,
        type=int,
        help='code size')
    argparser.add_argument(
        '--lr',
        default=5e-4,
        type=float,
        help='Learning rate')
    argparser.add_argument(
        '--cpc-weight',
        default=10.0,
        type=float,
        help='Learning rate')
    argparser.add_argument(
        '--gan-weight',
        default=1.0,
        type=float,
        help='Learning rate')
    argparser.add_argument('--doctor', action='store_true', default=False, help='Doctor')
    argparser.add_argument('--color', action='store_true', default=False, help='Color')

    args = argparser.parse_args()

    args.cpc_epochs = 0#1000
    args.gan_weight = 1.0
    args.cpc_weight = 5.0
    args.predict_terms = 5
    args.code_size = 128
    args.context_size = 512
    args.batch_size = 1#64
    args.color = False
    args.terms = 5
    args.load_name = "models"
    args.plan = 1 # {1: train from scratch, 2: resume training, 3: evaluate}



    # args.dataset = "ucf" # 
    # args.dataset = "mnist" # 
    # args.dataset = "generated" # 
    args.dataset = "kth" #

    if args.dataset == 'ucf' or args.dataset == 'baby':
        args.image_size = 224
    elif args.dataset == 'walking':
        args.image_size = 112
    elif args.dataset == 'kth':
        args.image_size = 112
        args.color = False
    else:
        args.image_size = 28

    print(args)
    train_model(
        args,
        batch_size=args.batch_size,
        output_dir='models',
        code_size=args.code_size,
        context_size = args.context_size,
        lr=args.lr,
        terms=args.terms,
        predict_terms=args.predict_terms,
        image_size=args.image_size,
        color=args.color,
        dataset=args.dataset
    )