import json

import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
import torch as pt

from preprocessing import read_split_images
from word2vec import train_w2v, use_w2v

PATH = "input/"


class FLICKR30K(Dataset):
    c_vec = CountVectorizer(stop_words='english', min_df=1, max_df=100000)
    w2v_model = None

    def __init__(self, mode, limit=-1):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        internal_set_images = read_split_images(path=PATH, mode=self.mode, limit=limit)
        internal_set_captions = json.load(open('output/data/{}.json'.format(self.mode), 'r'))

        self.image_labels = internal_set_images.iloc[:, 0]
        internal_set_images = internal_set_images.drop(0, 1)
        self.images = internal_set_images.to_numpy()

        self.caption_labels = list(internal_set_captions.keys())
        if mode == 'train':
            self.captions, FLICKR30K.w2v_model = train_w2v(internal_set_captions.values())
        else:
            self.captions = use_w2v(internal_set_captions.values(), FLICKR30K.w2v_model)

        if limit > -1:
            self.captions = self.captions[:limit * 5]
            self.caption_labels = self.caption_labels[:limit * 5]

        self.captions_per_image = len(self.caption_labels) / len(self.image_labels)
        self.images = np.repeat(self.images, repeats=self.captions_per_image,
                                axis=0)

        self.caption_indices = np.random.permutation(len(self.images))
        self.image_indices = np.random.permutation(len(self.images))

        self.captions = self.captions[self.caption_indices]
        self.caption_labels = self.captions[self.caption_indices]

        self.images = self.images[self.image_indices]
        self.image_labels = self.images[self.image_indices]

    def __getitem__(self, index):
        return self.image_indices[index], pt.tensor(self.images[index]).float(), self.caption_indices[index], pt.tensor(
            self.captions[index]).float()

    def __len__(self):
        return len(self.captions)

    def get_dimensions(self):
        return self.images.shape[1], self.captions.shape[1]
