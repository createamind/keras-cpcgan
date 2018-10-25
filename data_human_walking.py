import scipy.ndimage
import scipy.misc
import os
import random
import numpy as np
class HumanActivityDataGenerator(object):

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
        self.images = []

        for _, _, files in os.walk(os.path.join('./data/', subset)):
            for name in sorted(files):
                # print('Reading from ' + name)
                image = scipy.ndimage.imread(os.path.join('./data/', subset, name))
                image = (scipy.misc.imresize(image, [112, 112]).astype(float) - 127) / 128.0
                self.images.append(image[:, :, :1])
        
        self.tot = len(self.images)
        self.n_samples = 3000 if subset == 'train' else 600
        self.n_batches = self.n_samples // batch_size

        # self.batches = [self.get_data() for i in range(self.n_batches)]
        # self.batch_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches
        
    # def next(self):
    #     self.batch_index = (self.batch_index + 1) % self.n_batches
    #     return self.batches[self.batch_index]

    def next(self):
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        for b in range(self.batch_size):
            sentence_labels[b] = 1 if random.randint(1,5) == 1 else 0
        x, y, z = [], [], []
        for b in range(self.batch_size):
            random_pos = random.randint(self.terms, self.tot - self.predict_terms)
            term_images = self.images[random_pos - self.terms: random_pos]
            true_images = self.images[random_pos: random_pos + self.predict_terms]
            false_images=  list(reversed(true_images))
            #false_images = [self.images[random.randint(0, self.tot - 1)] for i in range(self.predict_terms)]
            if sentence_labels[b] == 0:
                x.append(term_images)
                y.append(false_images)
                z.append(true_images)
            else:
                x.append(term_images)
                y.append(true_images)
                z.append(true_images)

        return [np.array(x), np.array(y)], sentence_labels
