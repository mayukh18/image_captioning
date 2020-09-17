import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import cv2


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

    def __getitem__(self, i):

        data = self.captions[i]
        img_id = data['img_id']
        caption = data['sentences'][0]

        caption = [0] + [self.word_map[w] for w in caption.split(" ")] + [1]
        caption = caption + [2 for _ in range(62 - len(caption))]

        img1 = torch.from_numpy(
            cv2.imread(os.path.join(self.data_folder, 'resized_images', img_id + ".png"))).float().permute(2, 0, 1)
        img2 = torch.from_numpy(
            cv2.imread(os.path.join(self.data_folder, 'resized_images', img_id + "_2.png"))).float().permute(2, 0, 1)

        caplen = len(caption)
        caption = torch.LongTensor(caption)

        all_captions = []
        if self.split is 'train':
            return img1, img2, caption, caplen
        elif self.split is 'val':
            for cap in data['sentences']:
                cap = [0] + [self.word_map[w] for w in cap.split(" ")] + [1]
                cap = cap + [2 for _ in range(62 - len(cap))]
                all_captions.append(cap)
            return img1, img2, caption, caplen, torch.LongTensor(all_captions[:1])
        elif self.split is 'test':
            for cap in data['sentences']:
                cap = [0] + [self.word_map[w] for w in cap.split(" ")] + [1]
                cap = cap + [2 for _ in range(62 - len(cap))]
                all_captions.append(cap)
            # all_captions = torch.LongTensor(data['sentences'])
            return img1, img2, caption, caplen, torch.LongTensor(all_captions[:1])

    def __len__(self):
        return len(self.captions)





