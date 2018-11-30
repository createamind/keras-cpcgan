import scipy.ndimage
import scipy.misc
import os
import random
import numpy as np
import copy
import datetime


class VideoDataGenerator(object):

    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=28, color=False, rescale=True, dataset='ucf'):


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
        self.n_samples = 0

        print('load date start')
        print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print(dataset,subset)


        # data/dataset(kth)/subset(train)/dir_name/imgs
        data_dir = os.path.join('./data/', dataset, subset)
        if dataset == 'kth':
            if subset in ['train','val']:

                for dir_path, dir_names, file_names in os.walk(data_dir):

                    #print(_, [type(img_dirs),img_dirs], files)
                    #continue
                    
                    if len(dir_names) == 0:
                        print(dir_path)
                        images = []
                        c = 0
                        frames = []
                        for name in sorted(file_names):
                            # print('Reading from ' + name)
                            image = scipy.ndimage.imread(os.path.join(dir_path, name))[4:116, 24:136, :]

                            # if dataset == 'kth':
                            #     image = (scipy.misc.imresize(image, [112, 112]).astype(float) - 127) / 128.0
                            # else:
                            #     image = (scipy.misc.imresize(image, [224, 224]).astype(float) - 127) / 128.0

                            image = (scipy.misc.imresize(image, [112, 112]).astype(float) - 127) / 128.0

                            images.append(image[:, :, :1])
			
			# concatenate two images as input	
                        #images_np = np.array(images)
                        #images_aligned = np.concatenate((images_np[1:, ...], images_np[:-1, ...]), axis=-1)
                        #images = list(images_aligned)


                        self.videos.append(copy.deepcopy(images))
                        self.n_samples += len(images)




        print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print('load date over . ')
        print('len videos')
        print(len(self.videos))
        self.n_videos = len(self.videos)

        #self.n_samples = 3000 if subset == 'train' else 200
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
        sentence_labels = np.zeros((self.batch_size, 1)).astype('int32')
        # for b in range(self.batch_size):
        #     sentence_labels[b] = 1 if random.randint(1,5) == 1 else 0
        for b in range(self.positive_samples):
            sentence_labels[b] = 1

        x, y, z = [], [], []
        for b in range(self.batch_size):
            random_video = random.randint(0, self.n_videos - 1)     # the video number
            #random_pos = random.randint(self.terms, len(self.videos[random_video]) - self.predict_terms)   # the image position
            random_pos = random.randint(self.terms + 1, len(self.videos[random_video]) - self.predict_terms - 1)   # the image position
            term_images = self.videos[random_video][random_pos - self.terms: random_pos]
            true_images = self.videos[random_video][random_pos: random_pos + self.predict_terms]

            #if b%3 == 0:
            #    false_images = [self.videos[random_video][random.randint(0, len(self.videos[random_video]) - 1)] for i in range(self.predict_terms)]
            if b%2 == 1:
                false_images = self.videos[random_video][random_pos + 1: random_pos + self.predict_terms + 1]
            elif b%2 == 0:
                false_images = self.videos[random_video][random_pos - 1: random_pos + self.predict_terms - 1]

            #false_images = [self.videos[random_video][random.randint(0, len(self.videos[random_video]) - 1)] for i in range(self.predict_terms)]

            #false_images = self.videos[random_video][random_pos - self.predict_terms: random_pos]
            
            #if b%3 == 0:
            #    false_images = [self.videos[random_video][random.randint(0, len(self.videos[random_video]) - 1)] for i in range(self.predict_terms)]
            #elif b%3 == 1:
            #    false_images = self.videos[random_video][random_pos - self.predict_terms: random_pos]
            #else:
            #    false_images = [self.videos[random_video][random_pos - 1] for i in range(self.predict_terms)]

            #rand_video = random.randint(0, self.n_videos - 1)
            #false_images = [self.videos[rand_video][random.randint(0, len(self.videos[rand_video]) - 1)] for i in range(self.predict_terms)]


            x.append(term_images)
            if sentence_labels[b] == 0:
                y.append(false_images)
                #z.append(true_images)
            else:
                y.append(true_images)
                #z.append(true_images)

        return [np.array(x), np.array(y)], sentence_labels
