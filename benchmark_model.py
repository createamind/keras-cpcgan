''' This module evaluates the performance of a trained CPC encoder '''

from data_utils import MnistGenerator
from os.path import join, basename, dirname, exists
import keras
import datetime
import argparse
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def build_model(encoder_path, image_shape, learning_rate):

    # Read the encoder
    encoder = keras.models.load_model(encoder_path)

    # Freeze weights
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    # Define the classifier
    x_input = keras.layers.Input(image_shape)
    x = encoder(x_input)
    x = keras.layers.Dense(units=128, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=10, activation='softmax')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()

    return model


def benchmark_model(args, encoder_path, epochs, batch_size, output_dir, lr=1e-4, image_size=28, color=False):

    # Prepare data
    train_data = MnistGenerator(batch_size, subset='train', image_size=image_size, color=color, rescale=True)

    validation_data = MnistGenerator(batch_size, subset='valid', image_size=image_size, color=color, rescale=True)

    # Prepares the model
    channel = 3 if color else 1
    model = build_model(encoder_path, image_shape=(image_size, image_size, channel), learning_rate=lr)

    # Callbacks
    callbacks = [
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4),
        keras.callbacks.TensorBoard(log_dir='./logs/bm_' + args.name + '_' +datetime.datetime.now().strftime('%d_%H-%M-%S ') , histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    ]
    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Saves the model
    model.save(join(output_dir, 'supervised.h5'))


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='CPC')
    # argparser.add_argument(
    #     '--host',
    #     metavar='H',
    #     default='localhost',
    #     help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '--name',
        default='cpc',
        help='name')
    # argparser.add_argument(
    #     '-p', '--port',
    #     metavar='P',
    #     default=2000,
    #     type=int,
    #     help='TCP port to listen to (default: 2000)')
    # argparser.add_argument(
    #     '-s', '--step-limit',
    #     default=256,
    #     type=int,
    #     help='Step limit of each run. (default: 256)')
    # argparser.add_argument(
    #     '-i', '--image-size',
    #     default=160,
    #     type=int,
    #     help='Size of images (default: 320).')
    # argparser.add_argument(
    #     '-b', '--batch-size',
    #     default=32,
    #     type=int,
    #     help='Size of batches.')
    # argparser.add_argument(
    #     '-t', '--train-epoch',
    #     default=100,
    #     type=int,
    #     help='Times of train.')
    # argparser.add_argument(
    #     '--vaealpha',
    #     default=1,
    #     type=int,
    #     help='Times of train.')
    # argparser.add_argument(
    #     '--mse-weight',
    #     default=0.01,
    #     type=float,
    #     help='Weight of MSE.')
    # argparser.add_argument(
    #     '--name',
    #     default=0.01,
    #     type=float,
    #     help='Weight of MSE.')
    # argparser.add_argument('--include-throttle', action='store_true', default=False, help='Include Throttle')
        
    args = argparser.parse_args()

    benchmark_model(
        args, 
        encoder_path='models/encoder_' + args.name + '.h5',
        epochs=15,
        batch_size=64,
        output_dir='models/',
        lr=1e-3,
        image_size=64,
        color=False
    )
