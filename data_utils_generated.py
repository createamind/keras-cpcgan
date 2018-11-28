''' This module contains code to handle data '''

import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageFilter


class Code(object):
    def __init__(self, imgSize=(28, 28), fontsize=20, bgColor=(225,) * 4, \
                 fontColor=(0, 0, 0)):
        self.imgSize = imgSize
        self.fontsize = fontsize
        self.bgColor = bgColor
        self.fontColor = fontColor
        self.font = ImageFont.truetype('./arial.ttf', self.fontsize)

    # def getDigit(self, digit):
    #     return str(digit)


    # def getFont(self, fontFile='./arial.ttf'):
    #     return ImageFont.truetype(fontFile, self.fontsize)

    def getTextPos(self, text, font):
        textWidth, textHeight = font.getsize(text)
        imgWidth, imgHeight = self.imgSize
        textPos = ((imgWidth - textWidth) / 2, (imgHeight - textHeight) / 2)
        return textPos

    def rotateImg(self, image, angle=0, expand=0):
        rot = image.rotate(angle, expand)
        fff = Image.new('RGBA', rot.size, self.bgColor)
        image = Image.composite(rot, fff, rot)
        return image

    #def createImg(self, digit):
    def createImg(self, text, angle):
        # '=' = 10 '+' = 11 '*' = 12
        if text == '10':
            text = '='
        if text == '11':
            text = '+'
        if text == '12':
            text = '*'

        #codeImg = Image.new('RGBA', self.imgSize, self.bgColor).convert("RGBA")
        codeImg = Image.new('RGBA', self.imgSize, self.bgColor)
        draw = ImageDraw.Draw(codeImg)
        textPos = self.getTextPos(text, self.font)
        draw.text(xy=textPos, text=text, fill=self.fontColor, font=self.font)
        codeImg = self.rotateImg(codeImg, angle)
        return np.array(codeImg)

    def get_batch_by_labels(self, subset, labels, image_size=28, color=False, rescale=True):

        images = []
        for i, label in enumerate(labels):
            images.append(self.createImg(str(int(label)), np.random.uniform(-30, 30))[:, :, :1])

        return (np.array(images).astype('float32') - 127) / 128.0, labels.astype('int32')


class GeneratedNumberGenerator(object):

    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=28, color=False, rescale=True):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.mnist_handler = Code()
        self.n_samples = 3000 if subset == 'train' else 600
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Build sentences
        image_labels = np.zeros((self.batch_size, self.terms + self.predict_terms + self.predict_terms))
        # true_image_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        positive_samples_n = self.positive_samples
        for b in range(self.batch_size):

            # Set ordered predictions for positive samples
            seed = np.random.randint(0, 10)

            # '=' = 10 '+' = 11 '*' = 12
            number = np.random.randint(0, 49999)
            left_number = list(map(int,list('%.5d' % number))) + [12, 2, 10]
            right_number = list(map(int,list('%.5d' % (number * 2))))
            sentence = np.array(left_number + right_number + right_number)
            # sentence = np.mod(np.arange(seed, seed + self.terms + self.predict_terms), 10)
            # true_sentence = np.copy(sentence)
            # sentence = np.concatenate([sentence, sentence[-self.predict_terms:]], axis=0)
            assert self.predict_terms == 5
            assert self.terms == 8
            if positive_samples_n <= 0:

                # Set random predictions for negative samples
                # Each predicted term draws a number from a distribution that excludes itself
                numbers = np.arange(0, 10)
                predicted_terms = sentence[-self.predict_terms:]
                for i, p in enumerate(predicted_terms):
                    predicted_terms[i] = np.random.choice(numbers[numbers != p], 1)
                sentence[-self.predict_terms:] = np.mod(predicted_terms, 10)
                sentence_labels[b, :] = 0

            # Save sentence
            image_labels[b, :] = sentence
            # true_image_labels[b, :] = true_sentence
            positive_samples_n -= 1

        # Retrieve actual images
        images, _ = self.mnist_handler.get_batch_by_labels(self.subset, image_labels.flatten(), self.image_size, self.color, self.rescale)
        # true_images, _ = self.mnist_handler.get_batch_by_labels(self.subset, true_image_labels.flatten(), self.image_size, self.color, self.rescale)

        # Assemble batch
        images = images.reshape((self.batch_size, self.terms + self.predict_terms + self.predict_terms, images.shape[1], images.shape[2], images.shape[3]))
        # true_images = true_images.reshape((self.batch_size, self.terms + self.predict_terms, true_images.shape[1], true_images.shape[2], true_images.shape[3]))
        
        x_images = images[:, :self.terms, ...]
        z_images = images[:, self.terms: -self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]
        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)

        # return [x_images[idxs, ...], y_images[idxs, ...], z_images[idxs, ...]], [sentence_labels[idxs, ...], np.zeros((self.batch_size, 1))]
        return [x_images[idxs, ...], y_images[idxs, ...]], [sentence_labels[idxs, ...]]


# def plot_sequences(x, y, labels=None, output_path=None):

#     ''' Draws a plot where sequences of numbers can be studied conveniently '''

#     images = np.concatenate([x, y], axis=1)
#     if images.shape[-1] == 1:
#         images = np.concatenate([images,images,images], axis=-1)
#     n_batches = images.shape[0]
#     n_terms = images.shape[1]
#     counter = 1

#     for n_b in range(n_batches):
#         for n_t in range(n_terms):
#             plt.subplot(n_batches, n_terms, counter)
#             plt.imshow(images[n_b, n_t, :, :, :])
#             plt.axis('off')
#             counter += 1
#         if labels is not None:
#             plt.title(labels[n_b, 0])

#     if output_path is not None:
#         plt.savefig(output_path, dpi=600)
#     else:
#         plt.show()


# if __name__ == "__main__":

#     # Test SortedNumberGenerator
#     ag = SortedNumberGenerator(batch_size=8, subset='train', terms=4, positive_samples=4, predict_terms=4, image_size=64, color=True, rescale=False)
#     for (x, y), labels in ag:
#         plot_sequences(x, y, labels, output_path=r'resources/batch_sample_sorted.png')
#         break

