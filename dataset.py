import json

import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
import torch as pt

from preprocessing import read_split_images

PATH = "input/"


class FLICKR30K(Dataset):
    c_vec = CountVectorizer(stop_words='english', min_df=1, max_df=100000)

    def __init__(self, mode, limit=-1):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        internal_set_images = read_split_images(path=PATH, mode=self.mode, limit=limit)
        internal_set_captions = json.load(open('output/data/{}.json'.format(self.mode), 'r'))

        self.image_labels = internal_set_images.iloc[:, 0]
        internal_set_images = internal_set_images.drop(0,1)
        self.images = internal_set_images.to_numpy()

        self.caption_labels = list(internal_set_captions.keys())
        if mode == 'train':
            self.captions = np.stack(FLICKR30K.c_vec.fit_transform(internal_set_captions.values()).todense(), axis=0)
        else:
            self.captions = np.stack(FLICKR30K.c_vec.transform(internal_set_captions.values()).todense(), axis=0)

        if limit > -1:
            self.captions = self.captions[:limit * 5]
            self.caption_labels = self.caption_labels[:limit * 5]

        self.captions_per_image = len(self.caption_labels) / len(self.image_labels)
        self.images = np.repeat(self.images, repeats=self.captions_per_image,
                                             axis=0)

    def __getitem__(self, index):
        return pt.tensor(self.images[index]).float(), pt.tensor(self.captions[index]).float()

    def __len__(self):
        return len(self.captions)

    def get_dimensions(self):
        return self.images.shape[1], self.captions.shape[1]
