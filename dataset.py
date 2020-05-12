import json

import torch
import numpy as np
from torch.utils.data import Dataset

from preprocessing import read_split_images

PATH = "input/"


class FLICKR30K(Dataset):
    def __init__(self, mode):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.internal_set_images = read_split_images(path=PATH, verbose=True, mode=self.mode)
        self.internal_set_captions = json.load(open('output/data/{}.json'.format(self.mode), 'r'))
        captions_per_image = len(self.internal_set_captions) / len(self.internal_set_images)
        self.internal_set_images = np.repeat(self.internal_set_images, repeats=captions_per_image, axis=0)

    def __getitem__(self, index):
        return self.internal_set_images[index], self.internal_set_captions[index]

    def __len__(self):
        return len(self.internal_set_images)

    def get_dimensions(self):
        return self.internal_set_images.shape, self.internal_set_captions.shape
