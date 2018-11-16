import scipy.ndimage
import scipy.misc
import os
import random
import numpy as np
import copy
import datetime
from matplotlib import pyplot as plt





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
        self.cnt = 1

        self.sceneframes = []
        self.scenes = []
        self.roads_sences = []
        self.info = []
        print('load date start')
        print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print(dataset,subset)





        for _, dirs, files in os.walk(os.path.join('./data/', dataset, subset)):
            a=0
            for dir_name in dirs:
                
                #framesall = []
                if dataset ==  'vkitty' or dataset =='kth' or dataset == 'vkittytest' : # or dataset =='ucfbig' :
                    if subset == 'train':
                        # if dir_name != '0001' :
                        print(dir_name)

                        # framesall = []
                        for _, dirs2, files in os.walk(os.path.join('./data/', dataset, subset, dir_name)):
                            for dir_name2 in dirs2:
                                print(dir_name2)
                                for _, _, files in os.walk(
                                        os.path.join('./data/', dataset, subset, dir_name, dir_name2)):
                                    # framesall = []
                                    c = 0
                                    frames = []
                                    for name in sorted(files):
                                        # print('Reading from ' + name)
                                        image = scipy.ndimage.imread(
                                            os.path.join('./data/', dataset, subset, dir_name, dir_name2, name))
                                        if dataset == 'vkitty' or dataset == 'vkittytest':
                                            image = (scipy.misc.imresize(image, [64, 200]).astype(float) - 127) / 128.0
                                        else:
                                            image = (scipy.misc.imresize(image, [112, 112]).astype(float) - 127) / 128.0
                                        c = (c + 1) % frame_stack
                                        if self.color:
                                            frames.append(image)
                                        else:
                                            frames.append(image[:, :, :1])
                                        if c == 0:
                                            self.sceneframes.append(copy.deepcopy(frames))
                                            frames = []
                                print('len sceneframes : ', len(self.sceneframes))
                                self.scenes.append(copy.deepcopy(self.sceneframes))
                                self.info.append(len(self.sceneframes))
                                self.sceneframes = []
                                print('len scenes: ', len(self.scenes))
                                print()

                        self.roads_sences.append(copy.deepcopy(self.scenes))
                        self.scenes = []
                        print('len roads_sences: ', len(self.roads_sences))
                        print()


                    else:
                        print(dir_name)

                        # framesall = []
                        for _, dirs2, files in os.walk(os.path.join('./data/', dataset, subset, dir_name)):
                            for dir_name2 in dirs2:
                                print(dir_name2)
                                for _, _, files in os.walk(
                                        os.path.join('./data/', dataset, subset, dir_name, dir_name2)):
                                    # framesall = []
                                    c = 0
                                    frames = []
                                    for name in sorted(files):
                                        # print('Reading from ' + name)
                                        image = scipy.ndimage.imread(
                                            os.path.join('./data/', dataset, subset, dir_name, dir_name2, name))
                                        if dataset == 'vkitty' or dataset == 'vkittytest':
                                            image = (scipy.misc.imresize(image, [64, 200]).astype(float) - 127) / 128.0
                                        else:
                                            image = (scipy.misc.imresize(image, [112, 112]).astype(float) - 127) / 128.0
                                        c = (c + 1) % frame_stack
                                        if self.color:
                                            frames.append(image)
                                        else:
                                            frames.append(image[:, :, :1])
                                        if c == 0:
                                            self.sceneframes.append(copy.deepcopy(frames))
                                            frames = []
                                print('len sceneframes : ', len(self.sceneframes))
                                self.scenes.append(copy.deepcopy(self.sceneframes))
                                self.sceneframes = []
                                print('len scenes: ', len(self.scenes))
                                print()
                        self.roads_sences.append(copy.deepcopy(self.scenes))
                        self.scenes = []
                        print('len roads_sences: ', len(self.roads_sences))
                        print()


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
                                image = (scipy.misc.imresize(image, [64, 64]).astype(float) - 127) / 128.0
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

        self.n_samples = 300 if subset == 'train' else 20
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
        self.cnt += 1
        sentence_labels = np.zeros((self.batch_size, 1)).astype('int32')
        for b in range(self.batch_size):
            sentence_labels[b] = (1 if random.randint(1, 2) == 1 else 0)
            #print("sentence_labels[b]:",sentence_labels[b])

        x, y, z = [], [], []
        for b in range(self.batch_size):
            #if self.cnt  > 20000:
                random_road = random.randint(0,len(self.roads_sences) - 1)
                #print('random_road: ', random_road)
                random_scene = random.randint(0,len(self.roads_sences[random_road]) - 1)
                #print('random_scene: ', random_scene)
                random_frames = random.randint(self.terms, len(self.roads_sences[random_road][random_scene]) - self.predict_terms)
                #print("len(self.roads_sences[random_road][random_scene])", len(self.roads_sences[random_road][random_scene]))
                #print("info:", self.info)
                #print('random_frames: ', random_frames)

                term_frames = self.roads_sences[random_road][random_scene][random_frames - self.terms: random_frames]
                if sentence_labels[b] == 0:
                    # random_pos = random.randint(self.terms, len(self.framesall) - self.predict_terms)
                    false_frames = [self.roads_sences[random_road][random_scene][  random.randint(0, len(self.roads_sences[random_road][random_scene]) - 1)] for i   in  range(self.predict_terms)]
                    x.append(term_frames)
                    y.append(false_frames)
                else:
                    true_frames = self.roads_sences[random_road][random_scene][
                                  random_frames: random_frames + self.predict_terms]
                    x.append(term_frames)
                    y.append(true_frames)

            # if self.cnt > 40000:
            #     random_road = random.randint(0,len(self.roads_sences) - 1)
            #     random_scene = random.randint(0,len(self.scenes) - 1)
            #     random_frames = random.randint(self.terms, len(self.sceneframes) - self.predict_terms)
            #
            #     term_frames = self.roads_sences[random_road][random_scene][random_frames - self.terms: random_frames]
            #
            #     print('random_road: ', random_road)
            #     print('random_scene: ', random_scene)
            #     print('random_frames: ', random_frames)
            #     print("len(self.roads_sences[random_road][random_scene])",
            #           len(self.roads_sences[random_road][random_scene]))
            #     print("info:", self.info)
            #
            #     if sentence_labels[b] == 0:
            #         # random_pos = random.randint(self.terms, len(self.framesall) - self.predict_terms)
            #         false_frames = [self.roads_sences[random_road][random_scene][
            #                             random.randint(0, len(self.roads_sences[random_road][random_scene]) - 1)] for i
            #                         in
            #                         range(self.predict_terms)]
            #         x.append(term_frames)
            #         y.append(false_frames)
            #     else:
            #         true_frames = self.roads_sences[random_road][random_scene][
            #                       random_frames: random_frames + self.predict_terms]
            #         x.append(term_frames)
            #         y.append(true_frames)
            #
            # if epoch > 7:
            #     random_road = random.randint(0,len(self.roads_sences) - 1)
            #     random_scene = random.randint(0,len(self.scenes) - 1)
            #     random_frames = random.randint(self.terms, len(self.sceneframes) - self.predict_terms)
            #
            #     term_frames = self.roads_sences[random_road][random_scene][random_frames - self.terms: random_frames]
            #
            #     print('random_road: ', random_road)
            #     print('random_scene: ', random_scene)
            #     print('random_frames: ', random_frames)
            #     print("len(self.roads_sences[random_road][random_scene])",
            #           len(self.roads_sences[random_road][random_scene]))
            #     print("info:", self.info)
            #
            #     if sentence_labels[b] == 0:
            #         # random_pos = random.randint(self.terms, len(self.framesall) - self.predict_terms)
            #         false_frames = [self.roads_sences[random_road][random_scene][
            #                             random.randint(0, len(self.roads_sences[random_road][random_scene]) - 1)] for i
            #                         in
            #                         range(self.predict_terms)]
            #         x.append(term_frames)
            #         y.append(false_frames)
            #     else:
            #         true_frames = self.roads_sences[random_road][random_scene][
            #                       random_frames: random_frames + self.predict_terms]
            #         x.append(term_frames)
            #         y.append(true_frames)



        return [np.array(x), np.array(y)], sentence_labels


if __name__ == "__main__":

    # Test SortedNumberGenerator
    ag = VideoDataGenerator(batch_size=8, subset='train', terms=4, positive_samples=4, predict_terms=4, image_size=64, color=False, rescale=False, dataset='vkitty', frame_stack = 8)
    for (x, y), labels in ag:
        VideoDataGenerator.next()
        break
