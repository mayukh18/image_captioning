import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import cv2
import random
import numpy as np

mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.224]).reshape(1, 1, 3)


class CaptionDataset(Dataset):
    def __init__(self, data_folder, word_map, split):
        print("HERE")
        self.split = split
        assert self.split in {'train', 'val', 'test'}
        self.word_map = word_map

        # getting only 1 of each pair
        self.image_files = [f for f in os.listdir(os.path.join(data_folder, "resized_images")) if "_" not in f]
        self.captions = json.load(open(os.path.join(data_folder, self.split + '.json'), 'r'))

        self.data_folder = data_folder
        self.dataset_size = len(self.captions)
        # print("WORD MAP", self.word_map)

    def __getitem__(self, i):

        data = self.captions[i]
        img_id = data['img_id']
        caption = data['sentences'][random.randint(0, len(data['sentences']) - 1)]

        caption = [0] + [self.word_map[w] if w in self.word_map else 2 for w in caption.split(" ")] + [1]
        caption = caption + [2 for _ in range(62 - len(caption))]

        img1 = cv2.imread(os.path.join(self.data_folder, 'resized_images', img_id + ".png"))
        img2 = cv2.imread(os.path.join(self.data_folder, 'resized_images', img_id + "_2.png"))

        # img1 = (img1 / 255.0 - mean) / std
        # img2 = (img2 / 255.0 - mean) / std

        img1 = torch.from_numpy(img1).float().permute(2, 0, 1)
        img2 = torch.from_numpy(img2).float().permute(2, 0, 1)

        caplen = len(caption)
        caption = torch.LongTensor(caption)

        all_captions = []
        if self.split == 'train':
            return img1, img2, caption, caplen
        elif self.split == 'val':
            for cap in data['sentences']:
                cap = [0] + [self.word_map[w] for w in cap.split(" ")] + [1]
                cap = cap + [2 for _ in range(62 - len(cap))]
                all_captions.append(cap)
            return img1, img2, caption, caplen, torch.LongTensor(all_captions)
        elif self.split == 'test':
            for cap in data['sentences']:
                # print("cap", cap)
                # cap = [0] + [self.word_map[w] if w in self.word_map else 2 for w in cap.split(" ")] + [1]
                cap = [0] + [self.word_map[w] for w in cap.split(" ") if w in self.word_map] + [1]
                cap = cap + [2 for _ in range(62 - len(cap))]
                all_captions.append(cap)
            # all_captions = torch.LongTensor(data['sentences'])
            return img1, img2, caption, caplen, torch.LongTensor(all_captions)

    def __len__(self):
        return len(self.captions)





