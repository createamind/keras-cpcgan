import scipy.ndimage
import scipy.misc
import os
import random
import numpy as np
import copy
class VideoDataGenerator(object):

    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=28, color=False, rescale=True, dataset='ucf', frame_stack = 2):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        self.videos = []

        for _, dirs, files in os.walk(os.path.join('./data/', dataset, subset)):
            for dir_name in dirs:
                if True:
                    images = []
                    for _, _, files in os.walk(os.path.join('./data/', dataset, subset, dir_name)):
                        images = []
                        c = 0
                        frames = []
                        for name in sorted(files)[:30]:
                            # print('Reading from ' + name)
                            image = scipy.ndimage.imread(os.path.join('./data/', dataset, subset, dir_name, name))
                            if dataset == 'walking':
                                image = (scipy.misc.imresize(image, [112, 112]).astype(float) - 127) / 128.0
                            else:
                                image = (scipy.misc.imresize(image, [224, 224]).astype(float) - 127) / 128.0
                            c = (c + 1) % frame_stack
                            frames.append(image[:, :, :1])  
                            if c == 0:
                                images.append(copy.deepcopy(frames))
                                frames = []
                    self.videos.append(copy.deepcopy(images))
            break
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
            random_video = random.randint(0, len(self.videos) - 1)
            random_pos = random.randint(self.terms, len(self.videos[random_video]) - self.predict_terms)
            term_images = self.videos[random_video][random_pos - self.terms: random_pos]
            true_images = self.videos[random_video][random_pos: random_pos + self.predict_terms]
            false_images = [self.videos[random_video][random.randint(0, len(self.videos[random_video]) - 1)] for i in range(self.predict_terms)]
            if sentence_labels[b] == 0:
                x.append(term_images)
                y.append(false_images)
                z.append(true_images)
            else:
                x.append(term_images)
                y.append(true_images)
                z.append(true_images)

        return [np.array(x), np.array(y)], sentence_labels
