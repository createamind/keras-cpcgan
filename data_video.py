import scipy.ndimage
import scipy.misc
import os
import random
import numpy as np
import copy
import datetime


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
        self.framesall = []
        print('load date start')
        print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print(dataset,subset)





        for _, dirs, files in os.walk(os.path.join('./data/', dataset, subset)):
            a=0
            for dir_name in dirs:
                
                #framesall = []
                if dataset ==  'vkitty' or dataset =='kth' or dataset == 'vkittytest' : # or dataset =='ucfbig' :
                  if subset == 'train':
                   if dir_name != '0001' :
                    print(dir_name)

                    #framesall = []
                    for _, dirs2, files in os.walk(os.path.join('./data/', dataset, subset, dir_name)):
                        for dir_name2 in dirs2:
                            print(dir_name2)
                            for _, _, files in os.walk(os.path.join('./data/', dataset, subset, dir_name,dir_name2)):
                                #framesall = []
                                c = 0
                                frames = []
                                for name in sorted(files):
                                    # print('Reading from ' + name)
                                    image = scipy.ndimage.imread(
                                        os.path.join('./data/', dataset, subset, dir_name,dir_name2, name))
                                    if dataset == 'vkitty'  or dataset == 'vkittytest' :
                                        image = (scipy.misc.imresize(image, [64,200]).astype(float) - 127) / 128.0
                                    else:
                                        image = (scipy.misc.imresize(image, [112, 112]).astype(float) - 127) / 128.0
                                    c = (c + 1) % frame_stack
                                    if self.color:
                                        frames.append(image)
                                    else:
                                        frames.append(image[:, :, :1])
                                    if c == 0:
                                        self.framesall.append(copy.deepcopy(frames))
                                        frames = []
                                    #print('1len framesframesall: ',len(images))
                                print('2len framesframesall : ', len(self.framesall))
                            #self.videos.append(copy.deepcopy(framesall))
                            #print('len videos: ',  len(self.videos))

                  else:
                      print(dir_name)
                      #framesall = []
                      for _, _, files in os.walk(os.path.join('./data/', dataset, subset, dir_name)):
                          #framesall = []
                          c = 0
                          frames = []
                          for name in sorted(files):
                              # print('Reading from ' + name)
                              image = scipy.ndimage.imread(os.path.join('./data/', dataset, subset, dir_name, name))

                              if dataset == 'vkitty' or dataset== 'vkittytest'  :
                                  image = (scipy.misc.imresize(image, [64, 200]).astype(float) - 127) / 128.0
                              else:
                                  image = (scipy.misc.imresize(image, [112, 112]).astype(float) - 127) / 128.0

                              c = (c + 1) % frame_stack
                              if self.color:
                                  frames.append(image)
                              else:
                                  frames.append(image[:, :, :1])
                              if c == 0:
                                  self.framesall.append(copy.deepcopy(frames))
                                  frames = []
                          #a = (a + 1 )
                          #if a > 10:
                          #    break
                          print('2len framesframesall : ', len(self.framesall))
                      #self.videos.append(copy.deepcopy(framesall))

                else:
                    print(dir_name)
                    #framesall = []
                    for _, _, files in os.walk(os.path.join('./data/', dataset, subset, dir_name)):
                        #framesall = []
                        c = 0
                        frames = []
                        for name in sorted(files):
                            # print('Reading from ' + name)
                            image = scipy.ndimage.imread(os.path.join('./data/', dataset, subset, dir_name, name))
                            if dataset == 'walking' or dataset == 'ucfbig' :
                                image = (scipy.misc.imresize(image, [112, 112]).astype(float) - 127) / 128.0
                            else:
                                image = (scipy.misc.imresize(image, [224, 224]).astype(float) - 127) / 128.0
                            c = (c + 1) % frame_stack
                            if self.color :
                                frames.append(image)
                            else:
                                frames.append(image[:, :, :1])

                            if c == 0:
                                self.framesall.append(copy.deepcopy(frames))
                                frames = []
                        print('len framesall: ', len(self.framesall))
                    a = (a + 1)
                    print(a,'s dirs')
                    if a > 5010:
                        break
                        #print('2len framesframesall : ', len(self.framesall))
                    #self.videos.append(copy.deepcopy(framesall))
            break

        print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print('load date over . ')
        #print('len videos')
        #print(len(self.videos))

        self.n_samples = 3000 if subset == 'train' else 200
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

        # print('len videos')
        # print(len(self.videos))


        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        for b in range(self.batch_size):
            sentence_labels[b] = 1 if random.randint(1,5) == 1 else 0
        x, y, z = [], [], []
        for b in range(self.batch_size):
            random_pos = random.randint(self.terms, len(self.framesall)  - self.predict_terms)
            term_frames = self.framesall[random_pos - self.terms: random_pos]
            true_frames = self.framesall[random_pos: random_pos + self.predict_terms]

            #print('random_pos: ', random_pos)

            random_pos = random.randint(self.terms, len(self.framesall) - self.predict_terms)

            false_frames = [ self.framesall[random.randint(0, len(self.framesall) - 1)] for i in range(self.predict_terms)]
            #print('random_pos: ' , random_pos)

            if sentence_labels[b] == 0:
                x.append(term_frames)
                y.append(false_frames)
            else:
                x.append(term_frames)
                y.append(true_frames)

        return [np.array(x), np.array(y)], sentence_labels


if __name__ == "__main__":

    # Test SortedNumberGenerator
    ag = VideoDataGenerator(batch_size=8, subset='train', terms=4, positive_samples=4, predict_terms=4, image_size=64, color=False, rescale=False, dataset='vkitty', frame_stack = 8)
    for (x, y), labels in ag:
        VideoDataGenerator.next()
        break
