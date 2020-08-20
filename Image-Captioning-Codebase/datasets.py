import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from vocabulary import Vocabulary
import nltk
import random
from PIL import Image
import numpy as np

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
    
    
class Flickr8kDataset(Dataset):
    
    def __init__(self, annot_path, img_path, split, transform=None):
        
        self.annot_path = annot_path
        self.img_path = img_path
        
        self.transform = transform
        self.mode = split
        
        self.load()
            
    def load(self):
        
        vocab_threshold = 50
        
        with open(os.path.join(self.annot_path, "Flickr8k.token.txt")) as f:
            out = f.read()
    
        lines = out.split("\n")
        image_ids = []
        captions = {}
        
        for line in lines:
            split = line.split("\t")
            im_id = split[0][:-2]
            
            try:
                cap = split[1]
            except:
                print(line)
            
            if im_id not in image_ids:
                image_ids.append(im_id)
                captions[im_id] = []
            captions[im_id].append(cap)
        
        self.captions = captions
        #print(captions)
        self.image_ids = image_ids
        
        with open(os.path.join(self.annot_path, "Flickr_8k."+self.mode+"Images.txt")) as fp:
            files = fp.read()
        self.mode_files = [f for f in files.split("\n")]
        #print(self.mode_files)
        
        self.vocab = Vocabulary(vocab_threshold, vocab_file="./vocab.pkl", captions=captions)
        
        all_tokens = [nltk.tokenize.word_tokenize(str(captions[im_id][i]).lower()) for i in range(len(captions[im_id])) for im_id in captions]
        self.caption_lengths = [len(token) for token in all_tokens]
        #print("MAX Caption Length", np.max(self.caption_lengths))
            
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train':
            
            img_id = self.mode_files[index]
            if img_id == '':
                return self.__getitem__(random.randint(0,len(self.mode_files)))
            
            caption = random.choice(self.captions[img_id])

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_path, img_id)).convert('RGB')
            if image != None and image.size[0] * image.size[1] > 0:
                image = self.transform(image)
            else:
                image = self.transform(Image.fromarray(np.zeros(224,224,3)))

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            
            caption.extend([self.vocab(self.vocab.end_word) for i in range(len(caption), 37)])
            
            caption = torch.Tensor(caption).long()
            
            caplen = torch.LongTensor([37])
            return image, caption, caplen

        # obtain image if in test mode
        else:
            img_id = self.mode_files[index]
            if img_id == '':
                return self.__getitem__(random.randint(0,len(self.mode_files)))
            
            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_path, img_id)).convert('RGB')
            if image != None and image.size[0] * image.size[1] > 0:
                image = self.transform(image)
            else:
                image = self.transform(Image.fromarray(np.zeros(224,224,3)))
            
            
            all_captions = []
            for i in range(len(self.captions[img_id])):
                tokens = nltk.tokenize.word_tokenize(str(self.captions[img_id][i]).lower())
                caption = []
                caption.append(self.vocab(self.vocab.start_word))
                caption.extend([self.vocab(token) for token in tokens])
                caption.append(self.vocab(self.vocab.end_word))
                caption.extend([self.vocab(self.vocab.end_word) for i in range(len(caption), 37)])
                
                all_captions.append(caption)
            
            caption = random.choice(all_captions)
            caption = torch.Tensor(caption).long()
            caplen = torch.LongTensor([37])
            
            all_captions = torch.Tensor(all_captions).long()
            
            # return original image and pre-processed image tensor
            return image, caption, caplen, all_captions

    def get_train_indices(self):
        batch_size = 8
        
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=batch_size))
        return indices

    def __len__(self):
        return len(self.mode_files)
