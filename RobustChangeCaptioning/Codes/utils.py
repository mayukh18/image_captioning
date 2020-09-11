import os
import numpy as np
import h5py
import json
import torch

from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

def create_input_files(dataset, data_json_path, img_feature_path, captions_per_image, min_word_freq, output_folder, viewpoint, max_len=100):
  assert dataset in {'3dcc'}

  with open(data_json_path+'3dcc_v0-4_generated_captions_train.json', 'r') as f:
    data_train = json.load(f)

  with open(data_json_path+'3dcc_v0-4_generated_captions_val.json', 'r') as f:
    data_val = json.load(f)

  with open(data_json_path+'3dcc_v0-4_generated_captions_test.json', 'r') as f:
    data_test = json.load(f)

  # Read image captions for each image

  train_image_ids_0 = []
  train_image_ids_1 = []
  train_image_captions = []
  train_caption_types = []

  val_image_ids_0 = []
  val_image_ids_1 = []
  val_image_captions = []
  val_caption_types = []

  test_image_ids_0 = []
  test_image_ids_1 = []
  test_image_captions = []
  test_caption_types = []

  word_freq = Counter()
  
  add_count = 0
  move_count = 0

  for caption_index in range(0, len(data_train), 5):
    train_image_ids_0.append(data_train[caption_index]['image_index'])

    if data_train[caption_index]['caption_type'] == 'add_object':
      train_image_ids_1.append(add_count)
      add_count += 1

    elif data_train[caption_index]['caption_type'] == 'move_object':
      train_image_ids_1.append(move_count)
      move_count += 1

    else:
      train_image_ids_1.append(data_train[caption_index]['image_index'])

    captions = []
    for cap_inx in range(captions_per_image):
      temp_caption = data_train[cap_inx + caption_index]['caption']
      temp_caption = temp_caption.replace('.', '')
      temp_caption = temp_caption.split(' ')

      word_freq.update(temp_caption)

      captions.append(temp_caption)

    train_image_captions.append(captions)
    train_caption_types.append(data_train[caption_index]['caption_type'])

  add_count = 0
  move_count = 0

  for caption_index in range(0, len(data_val), 5):
    val_image_ids_0.append(data_val[caption_index]['image_index'])

    if data_val[caption_index]['caption_type'] == 'add_object':
      val_image_ids_1.append(add_count)
      add_count += 1

    elif data_val[caption_index]['caption_type'] == 'move_object':
      val_image_ids_1.append(move_count)
      move_count += 1

    else:
      val_image_ids_1.append(data_val[caption_index]['image_index'])

    captions = []
    for cap_inx in range(captions_per_image):
      temp_caption = data_val[cap_inx + caption_index]['caption']
      temp_caption = temp_caption.replace('.', '')
      temp_caption = temp_caption.split(' ')

      word_freq.update(temp_caption)
  
      captions.append(temp_caption)

    val_image_captions.append(captions)
    val_caption_types.append(data_val[caption_index]['caption_type'])

  add_count = 0
  move_count = 0

  for caption_index in range(0, len(data_test), 5): 
    test_image_ids_0.append(data_test[caption_index]['image_index'])
    
    if data_test[caption_index]['caption_type'] == 'add_object':
      test_image_ids_1.append(add_count)
      add_count += 1

    elif data_test[caption_index]['caption_type'] == 'move_object':
      test_image_ids_1.append(move_count)
      move_count += 1

    else:
      test_image_ids_1.append(data_test[caption_index]['image_index'])

    captions = []
    for cap_inx in range(captions_per_image):
      temp_caption = data_test[cap_inx + caption_index]['caption']
      temp_caption = temp_caption.replace('.', '')
      temp_caption = temp_caption.split(' ')

      word_freq.update(temp_caption)
      captions.append(temp_caption)

    test_image_captions.append(captions)
    test_caption_types.append(data_test[caption_index]['caption_type'])

  assert len(train_image_ids_0) == len(train_image_captions)
  assert len(val_image_ids_0) == len(val_image_captions)
  assert len(test_image_ids_0) == len(test_image_captions)

  assert len(train_image_ids_0) == len(train_image_ids_1)
  assert len(val_image_ids_0) == len(val_image_ids_1)
  assert len(test_image_ids_0) == len(test_image_ids_1)

  words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
  word_map = {k: v + 1 for v, k in enumerate(words)}
  word_map['<unk>'] = len(word_map) + 1
  word_map['<start>'] = len(word_map) + 1
  word_map['<end>'] = len(word_map) + 1
  word_map['<pad>'] = 0

  base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

  with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as f:
    json.dump(word_map, f)

  for imgids0, imgids1, imgcaps, captypes, split in [(train_image_ids_0, train_image_ids_1, train_image_captions, train_caption_types, 'TRAIN'), (val_image_ids_0, val_image_ids_1, val_image_captions, val_caption_types, 'VAL'), (test_image_ids_0, test_image_ids_1, test_image_captions,test_caption_types,'TEST')]:

    tt = 0
    h1 = h5py.File(os.path.join(output_folder, split+'_IMAGE_FEATURES_1_')+base_filename + '.h5', 'a')
    h2 = h5py.File(os.path.join(output_folder, split+'_IMAGE_FEATURES_2_')+base_filename + '.h5', 'a')

    images1 = h1.create_dataset('images_features', (len(imgids0), 1024, 4, 4), dtype=np.float32)
    images2 = h2.create_dataset('images_features', (len(imgids1), 1024, 4, 4), dtype=np.float32)

    print('\nReading %s images and captions, storing to file...\n' % split)

    enc_captions = []
    caplens = []

    if split == 'TRAIN':
      feature_dir = img_feature_path + "train/"
    elif split == 'VAL':
      feature_dir = img_feature_path + "val/"
    elif split == 'TEST':
      feature_dir = img_feature_path + "test/"

    original_file = h5py.File(feature_dir+'original/0.h5','r')
    distract_file = h5py.File(feature_dir+'original/' + str(viewpoint) + '.h5','r')
    add_file = h5py.File(feature_dir+'add/' + str(viewpoint) + '.h5','r')
    delete_file = h5py.File(feature_dir+'delete/' + str(viewpoint) + '.h5','r')
    move_file = h5py.File(feature_dir+'move/' + str(viewpoint) + '.h5', 'r')
    replace_file = h5py.File(feature_dir+'replace/' + str(viewpoint) + '.h5', 'r')
    swap_file = h5py.File(feature_dir+'swap/' + str(viewpoint) + '.h5','r')

    for sample_number in range(len(imgids0)):
      temp_feat0 = original_file['images_features'][imgids0[sample_number],:,:,:]

      if captypes[sample_number] == 'distract':
        temp_feat1 = distract_file['images_features'][imgids1[sample_number],:,:,:]

      elif captypes[sample_number] == 'add_object':
        temp_feat1 = add_file['images_features'][imgids1[sample_number],:,:,:]

      elif captypes[sample_number] == 'drop_object':
        temp_feat1 = delete_file['images_features'][imgids1[sample_number],:,:,:]

      elif captypes[sample_number] == 'move_object':
        temp_feat1 = move_file['images_features'][imgids1[sample_number],:,:,:]

      elif captypes[sample_number] == 'swap_objects':
        temp_feat1 = swap_file['images_features'][imgids1[sample_number],:,:,:]

      elif captypes[sample_number] == 'replace_object':
        temp_feat1 = replace_file['images_features'][imgids1[sample_number],:,:,:]

      h1['images_features'][tt:tt+1] = temp_feat0
      h2['images_features'][tt:tt+1] = temp_feat1
      tt = tt + 1
      temp_feat0 = None
      temp_feat1 = None

    print(len(imgcaps))

    for caption_index in range(len(imgcaps)):
      captions = imgcaps[caption_index]

      for j, c in enumerate(captions):
        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']] + [word_map['<pad>']]*(max_len-len(c))

        c_len = len(c) + 2

        enc_captions.append(enc_c)
        caplens.append(c_len)

    assert len(enc_captions) == len(caplens)

    with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as f:
      json.dump(enc_captions, f)

    with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as f:
      json.dump(caplens, f)  


def init_embedding(embeddings):
  bias = np.sqrt(3.0/embeddings.size(1))
  torch.nn.init.uniform_(embeddings, -bias, bias)

def load_embeddings(emb_file, word_map):
  # Find embedding dimension
  with open(emb_file, 'r') as f:
    emb_dim = len(f.readline().split(' ')) - 1

  vocab = set(word_map.keys())

  embeddings = torch.FloatTensor(len(vocab), emb_dim)
  init_embedding(embeddings)

  # Read embedding file
  print("\nLoading embeddings...")
  for line in open(emb_file, 'r'):
    line = line.split(' ')

    emb_word = line[0]
    embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

    if emb_word not in vocab:
      continue

    embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

  return embeddings, emb_dim

def save_checkpoint(root_dir, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best):
  state = {'epoch': epoch,
           'epochs_since_improvement': epochs_since_improvement,
           'bleu-4': bleu4,
           'encoder':encoder,
           'decoder':decoder,
           'encoder_optimizer': encoder_optimizer,
           'decoder_optimizer': decoder_optimizer}

  filename = root_dir + 'checkpoint_' + data_name + '.pth.tar'
  torch.save(state, filename)
  if is_best:
    torch.save(state, root_dir + 'BEST_checkpoint_' + data_name + '.pth.tar')



class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
  print("\nDecaying learning rate.")
  for param_group in optimizer.param_groups:
    param_group['lr'] = param_group['lr'] * shrink_factor
  print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
  batch_size = targets.size(0)
  _, ind = scores.topk(k, 1, True, True)
  correct = ind.eq(targets.view(-1,1).expand_as(ind))

  correct_total = torch.sum(correct.view(-1))
  
  return correct_total.item()*(100.0/batch_size)








