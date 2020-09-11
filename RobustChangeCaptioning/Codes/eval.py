import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json

import argparse

# Parameters

data_name = '3dcc_5_cap_per_img_0_min_word_freq' # base name shared by data files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # sets device for model and PyTorch tensors
cudnn.benchmark = True # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

captions_per_image = 5
batch_size = 1

# model_name


def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def evaluate(args, beam_size, n_gram):
  # Load model
  checkpoint = torch.load(args.checkpoint,map_location='cuda:0')

  encoder = checkpoint['encoder']
  encoder = encoder.to(device)
  encoder.eval()

  decoder = checkpoint['decoder']
  decoder = decoder.to(device)
  decoder.eval()

  # Load word map (word2ix)
  with open(args.word_map_file, 'r') as f:
    word_map = json.load(f)

  rev_word_map = {v: k for k, v in word_map.items()}
  vocab_size = len(word_map)


  ## total
  result_file_name = 'eval_results_fortest/' + args.model_name + '_BEST.txt'
  reference_file_name1 = 'eval_results_fortest/' + args.model_name + '_BEST_ref1.txt'
  reference_file_name2 = 'eval_results_fortest/' + args.model_name + '_BEST_ref2.txt'
  reference_file_name3 = 'eval_results_fortest/' + args.model_name + '_BEST_ref3.txt'
  reference_file_name4 = 'eval_results_fortest/' + args.model_name + '_BEST_ref4.txt'
  reference_file_name5 = 'eval_results_fortest/' + args.model_name + '_BEST_ref5.txt'

  result_file = open(result_file_name, 'a')
  reference_file1 = open(reference_file_name1, 'a')
  reference_file2 = open(reference_file_name2, 'a')
  reference_file3 = open(reference_file_name3, 'a')
  reference_file4 = open(reference_file_name4, 'a')
  reference_file5 = open(reference_file_name5, 'a')

  result_json_file = {}
  reference_json_file = {}

  ## add
  result_file_name_add = 'eval_results_fortest/' + args.model_name + '_add_BEST.txt'
  reference_file_name1_add = 'eval_results_fortest/' + args.model_name + '_add_BEST_ref1.txt'
  reference_file_name2_add = 'eval_results_fortest/' + args.model_name + '_add_BEST_ref2.txt'
  reference_file_name3_add = 'eval_results_fortest/' + args.model_name + '_add_BEST_ref3.txt'
  reference_file_name4_add = 'eval_results_fortest/' + args.model_name + '_add_BEST_ref4.txt'
  reference_file_name5_add = 'eval_results_fortest/' + args.model_name + '_add_BEST_ref5.txt'

  result_file_add = open(result_file_name_add, 'a')
  reference_file1_add = open(reference_file_name1_add, 'a')
  reference_file2_add = open(reference_file_name2_add, 'a')
  reference_file3_add = open(reference_file_name3_add, 'a')
  reference_file4_add = open(reference_file_name4_add, 'a')
  reference_file5_add = open(reference_file_name5_add, 'a')

  result_json_file_add = {}
  reference_json_file_add = {}

  ## delete
  result_file_name_delete = 'eval_results_fortest/' + args.model_name + '_delete_BEST.txt'
  reference_file_name1_delete = 'eval_results_fortest/' + args.model_name + '_delete_BEST_ref1.txt'
  reference_file_name2_delete = 'eval_results_fortest/' + args.model_name + '_delete_BEST_ref2.txt'
  reference_file_name3_delete = 'eval_results_fortest/' + args.model_name + '_delete_BEST_ref3.txt'
  reference_file_name4_delete = 'eval_results_fortest/' + args.model_name + '_delete_BEST_ref4.txt'
  reference_file_name5_delete = 'eval_results_fortest/' + args.model_name + '_delete_BEST_ref5.txt'

  result_file_delete = open(result_file_name_delete, 'a')
  reference_file1_delete = open(reference_file_name1_delete, 'a')
  reference_file2_delete = open(reference_file_name2_delete, 'a')
  reference_file3_delete = open(reference_file_name3_delete, 'a')
  reference_file4_delete = open(reference_file_name4_delete, 'a')
  reference_file5_delete = open(reference_file_name5_delete, 'a')

  result_json_file_delete = {}
  reference_json_file_delete = {}

  ## move
  result_file_name_move = 'eval_results_fortest/' + args.model_name + '_move_BEST.txt'
  reference_file_name1_move = 'eval_results_fortest/' + args.model_name + '_move_BEST_ref1.txt'
  reference_file_name2_move = 'eval_results_fortest/' + args.model_name + '_move_BEST_ref2.txt'
  reference_file_name3_move = 'eval_results_fortest/' + args.model_name + '_move_BEST_ref3.txt'
  reference_file_name4_move = 'eval_results_fortest/' + args.model_name + '_move_BEST_ref4.txt'
  reference_file_name5_move = 'eval_results_fortest/' + args.model_name + '_move_BEST_ref5.txt'

  result_file_move = open(result_file_name_move, 'a')
  reference_file1_move = open(reference_file_name1_move, 'a')
  reference_file2_move = open(reference_file_name2_move, 'a')
  reference_file3_move = open(reference_file_name3_move, 'a')
  reference_file4_move = open(reference_file_name4_move, 'a')
  reference_file5_move = open(reference_file_name5_move, 'a')

  result_json_file_move = {}
  reference_json_file_move = {}

  ## swap
  result_file_name_swap = 'eval_results_fortest/' + args.model_name + '_swap_BEST.txt'
  reference_file_name1_swap = 'eval_results_fortest/' + args.model_name + '_swap_BEST_ref1.txt'
  reference_file_name2_swap = 'eval_results_fortest/' + args.model_name + '_swap_BEST_ref2.txt'
  reference_file_name3_swap = 'eval_results_fortest/' + args.model_name + '_swap_BEST_ref3.txt'
  reference_file_name4_swap = 'eval_results_fortest/' + args.model_name + '_swap_BEST_ref4.txt'
  reference_file_name5_swap = 'eval_results_fortest/' + args.model_name + '_swap_BEST_ref5.txt'

  result_file_swap = open(result_file_name_swap, 'a')
  reference_file1_swap = open(reference_file_name1_swap, 'a')
  reference_file2_swap = open(reference_file_name2_swap, 'a')
  reference_file3_swap = open(reference_file_name3_swap, 'a')
  reference_file4_swap = open(reference_file_name4_swap, 'a')
  reference_file5_swap = open(reference_file_name5_swap, 'a')

  result_json_file_swap = {}
  reference_json_file_swap = {}

  ## replace
  result_file_name_replace = 'eval_results_fortest/' + args.model_name + '_replace_BEST.txt'
  reference_file_name1_replace = 'eval_results_fortest/' + args.model_name + '_replace_BEST_ref1.txt'
  reference_file_name2_replace = 'eval_results_fortest/' + args.model_name + '_replace_BEST_ref2.txt'
  reference_file_name3_replace = 'eval_results_fortest/' + args.model_name + '_replace_BEST_ref3.txt'
  reference_file_name4_replace = 'eval_results_fortest/' + args.model_name + '_replace_BEST_ref4.txt'
  reference_file_name5_replace = 'eval_results_fortest/' + args.model_name + '_replace_BEST_ref5.txt'

  result_file_replace = open(result_file_name_replace, 'a')
  reference_file1_replace = open(reference_file_name1_replace, 'a')
  reference_file2_replace = open(reference_file_name2_replace, 'a')
  reference_file3_replace = open(reference_file_name3_replace, 'a')
  reference_file4_replace = open(reference_file_name4_replace, 'a')
  reference_file5_replace = open(reference_file_name5_replace, 'a')

  result_json_file_replace = {}
  reference_json_file_replace = {}

  ## distract
  result_file_name_distract = 'eval_results_fortest/' + args.model_name + '_distract_BEST.txt'
  reference_file_name1_distract = 'eval_results_fortest/' + args.model_name + '_distract_BEST_ref1.txt'
  reference_file_name2_distract = 'eval_results_fortest/' + args.model_name + '_distract_BEST_ref2.txt'
  reference_file_name3_distract = 'eval_results_fortest/' + args.model_name + '_distract_BEST_ref3.txt'
  reference_file_name4_distract = 'eval_results_fortest/' + args.model_name + '_distract_BEST_ref4.txt'
  reference_file_name5_distract = 'eval_results_fortest/' + args.model_name + '_distract_BEST_ref5.txt'

  result_file_distract = open(result_file_name_distract, 'a')
  reference_file1_distract = open(reference_file_name1_distract, 'a')
  reference_file2_distract = open(reference_file_name2_distract, 'a')
  reference_file3_distract = open(reference_file_name3_distract, 'a')
  reference_file4_distract = open(reference_file_name4_distract, 'a')
  reference_file5_distract = open(reference_file_name5_distract, 'a')

  result_json_file_distract = {}
  reference_json_file_distract = {}

  """
  Evaluation

  :param beam_size: beam size at which to generate captions for evaluation
  :return: BLEU-4 score
  """

  # DataLoader
  loader = torch.utils.data.DataLoader(
      CaptionDataset(args.data_folder, data_name, 'TEST', captions_per_image),
      batch_size = batch_size, shuffle=False, num_workers=1, pin_memory=True)

  # TODO: Batched Beam Search
  # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

  # Lists to store references (true captions), and hypothesis (prediction) for each image
  # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
  # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
  references = list()
  hypotheses = list()

  references_0 = list()
  hypotheses_0 = list()

  references_1 = list()
  hypotheses_1 = list()

  references_2 = list()
  hypotheses_2 = list()

  references_3 = list()
  hypotheses_3 = list()

  references_4 = list()
  hypotheses_4 = list()

  references_5 = list()
  hypotheses_5 = list()



  # For each image
  ddd = 0
  for i, (image1, image2, caps, caplens, allcaps, captypes) in enumerate(
          tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

    if ddd == 5000:
      break
    current_index = i
    ddd += 1

    k = beam_size

    # Move to GPU device, if available
    image1 = image1.to(device) # (1, 768, 16, 16) 
    image2 = image2.to(device) # (1, 768, 16, 16) 

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)   # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device) # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()



    # Start decoding
    step = 1

    l_bef, l_aft, alpha_bef, alpha_aft = encoder(image1, image2)
      
    l_diff = torch.sub(l_aft,l_bef)

    l_total = torch.cat([l_bef,l_aft,l_diff],dim=1)

    l_total = decoder.relu(decoder.wd1(l_total))

    
    h_da = torch.zeros(1, decoder.hidden_dim).to(device)  ## TODO ## random?
    c_da = torch.zeros(1, decoder.hidden_dim).to(device)

    h_ds = torch.zeros(1, decoder.hidden_dim).to(device)
    c_ds = torch.zeros(1, decoder.hidden_dim).to(device)


    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
      embeddings = decoder.embedding(k_prev_words).squeeze(1) # (s, embed_dim)
      
      u_t = torch.cat([l_total, h_ds],dim=1)
      h_da, c_da = decoder.dynamic_att(u_t, (h_da, c_da))

      a_t = decoder.softmax(decoder.wd2(h_da)) 

      l_dyn = a_t[:,0].unsqueeze(1)*l_bef + a_t[:,1].unsqueeze(1)*l_aft + a_t[:,2].unsqueeze(1)*l_diff

      c_t = torch.cat([embeddings,l_dyn], dim=1)


      h_ds, c_ds = decoder.decode_step(c_t, (h_ds, c_ds)) # (s, decoder_dim)
   
      scores = decoder.wdc(h_ds) # (s, vocab_size)
      scores = F.log_softmax(scores, dim=1)


      # Add
      scores = top_k_scores.expand_as(scores) + scores # (s, vocab_size)      

      # For the first step, all k points will have the same scores (since same k previous words, h, c)
      if step == 1:
        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) # (s)
      else:
        # Unroll and find top scores, and their unrolled indices
        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) # (s)

      # Convert unrolled indices to actual indices of scores
      prev_word_inds = top_k_words / vocab_size # (s)
      next_word_inds = top_k_words % vocab_size # (s)

      # Add new words to sequences
      seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1) # (s, step + 1)

      # Which sequences are incomplete (didn't reach <end>)?
      incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
      complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

      # Set aside complete sequences
      if len(complete_inds) > 0:
        complete_seqs.extend(seqs[complete_inds].tolist())
        complete_seqs_scores.extend(top_k_scores[complete_inds])
      k -= len(complete_inds) # reduce beam length accordingly

      # Proceed with incomplete sequences
      if k == 0:
        break
      seqs = seqs[incomplete_inds]
      h_ds = h_ds[prev_word_inds[incomplete_inds]]
      c_ds = c_ds[prev_word_inds[incomplete_inds]]
      image1 = image1[prev_word_inds[incomplete_inds]]
      image2 = image2[prev_word_inds[incomplete_inds]]
      top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
      k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

      # Break if things have been going on too long
      if step > 50:
        break
      step += 1
    
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    # References
    img_caps = allcaps[0].tolist()
    img_captions = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps)) # remove <start> and pads
    references.append(img_captions)


    # Hypotheses
    temptemp = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
    hypotheses.append(temptemp)
 

    captype = captypes[0].item()

    if captype == 0:
      references_0.append(img_captions)
      hypotheses_0.append(temptemp)

    if captype == 1:
      references_1.append(img_captions)
      hypotheses_1.append(temptemp)

    if captype == 2:
      references_2.append(img_captions)
      hypotheses_2.append(temptemp)

    if captype == 3:
      references_3.append(img_captions)
      hypotheses_3.append(temptemp)

    if captype == 4:
      references_4.append(img_captions)
      hypotheses_4.append(temptemp)

    if captype == 5:
      references_5.append(img_captions)
      hypotheses_5.append(temptemp)

    assert len(references) == len(hypotheses)


  #-----------------------------------------------------------------
  kkk = -1
  for item in hypotheses:
    kkk += 1
    if kkk % 5 == 0:
      line_hypo = ""
      for word_idx in item:
        word = get_key(word_map, word_idx)
        #print(word)
        line_hypo += word[0] + " "

      result_json_file[str(kkk)] = []
      result_json_file[str(kkk)].append(line_hypo)


      line_hypo += "\r\n"
      result_file.write(line_hypo)
        
  kkk = -1
  for item in references:
    kkk += 1
    if kkk % 5 == 0:
      iii = 0
      reference_json_file[str(kkk)] = []

      for sentence in item:
        line_repo = ""
        for word_idx in sentence:
          word = get_key(word_map, word_idx)
          line_repo += word[0] + " "
              
        reference_json_file[str(kkk)].append(line_repo)

        line_repo += "\r\n"

        if iii == 0:
          reference_file1.write(line_repo)
        if iii == 1:
          reference_file2.write(line_repo)
        if iii == 2:
          reference_file3.write(line_repo)
        if iii == 3:
          reference_file4.write(line_repo)
        if iii == 4:
          reference_file5.write(line_repo)
        iii += 1

  #-----------------------------------------------------------------

  #----------------------------------------------------------------- add
  kkk = -1
  for item in hypotheses_0:
    kkk += 1
    if kkk % 5 == 0:
      line_hypo = ""
      for word_idx in item:
        word = get_key(word_map, word_idx)
        #print(word)
        line_hypo += word[0] + " "

      result_json_file_add[str(kkk)] = []
      result_json_file_add[str(kkk)].append(line_hypo)


      line_hypo += "\r\n"
      result_file_add.write(line_hypo)
        
  kkk = -1
  for item in references_0:
    kkk += 1
    if kkk % 5 == 0:
      iii = 0
      reference_json_file_add[str(kkk)] = []

      for sentence in item:
        line_repo = ""
        for word_idx in sentence:
          word = get_key(word_map, word_idx)
          line_repo += word[0] + " "
              
        reference_json_file_add[str(kkk)].append(line_repo)

        line_repo += "\r\n"

        if iii == 0:
          reference_file1_add.write(line_repo)
        if iii == 1:
          reference_file2_add.write(line_repo)
        if iii == 2:
          reference_file3_add.write(line_repo)
        if iii == 3:
          reference_file4_add.write(line_repo)
        if iii == 4:
          reference_file5_add.write(line_repo)
        iii += 1

  #-----------------------------------------------------------------

  #----------------------------------------------------------------- delete
  kkk = -1
  for item in hypotheses_1:
    kkk += 1
    if kkk % 5 == 0:
      line_hypo = ""
      for word_idx in item:
        word = get_key(word_map, word_idx)
        #print(word)
        line_hypo += word[0] + " "

      result_json_file_delete[str(kkk)] = []
      result_json_file_delete[str(kkk)].append(line_hypo)


      line_hypo += "\r\n"
      result_file_delete.write(line_hypo)
        
  kkk = -1
  for item in references_1:
    kkk += 1
    if kkk % 5 == 0:
      iii = 0
      reference_json_file_delete[str(kkk)] = []

      for sentence in item:
        line_repo = ""
        for word_idx in sentence:
          word = get_key(word_map, word_idx)
          line_repo += word[0] + " "
              
        reference_json_file_delete[str(kkk)].append(line_repo)

        line_repo += "\r\n"

        if iii == 0:
          reference_file1_delete.write(line_repo)
        if iii == 1:
          reference_file2_delete.write(line_repo)
        if iii == 2:
          reference_file3_delete.write(line_repo)
        if iii == 3:
          reference_file4_delete.write(line_repo)
        if iii == 4:
          reference_file5_delete.write(line_repo)
        iii += 1

  #-----------------------------------------------------------------

  #----------------------------------------------------------------- move
  kkk = -1
  for item in hypotheses_2:
    kkk += 1
    if kkk % 5 == 0:
      line_hypo = ""
      for word_idx in item:
        word = get_key(word_map, word_idx)
        #print(word)
        line_hypo += word[0] + " "

      result_json_file_move[str(kkk)] = []
      result_json_file_move[str(kkk)].append(line_hypo)


      line_hypo += "\r\n"
      result_file_move.write(line_hypo)
        
  kkk = -1
  for item in references_2:
    kkk += 1
    if kkk % 5 == 0:
      iii = 0
      reference_json_file_move[str(kkk)] = []

      for sentence in item:
        line_repo = ""
        for word_idx in sentence:
          word = get_key(word_map, word_idx)
          line_repo += word[0] + " "
              
        reference_json_file_move[str(kkk)].append(line_repo)

        line_repo += "\r\n"

        if iii == 0:
          reference_file1_move.write(line_repo)
        if iii == 1:
          reference_file2_move.write(line_repo)
        if iii == 2:
          reference_file3_move.write(line_repo)
        if iii == 3:
          reference_file4_move.write(line_repo)
        if iii == 4:
          reference_file5_move.write(line_repo)
        iii += 1
  #-----------------------------------------------------------------

  #----------------------------------------------------------------- swap
  kkk = -1
  for item in hypotheses_3:
    kkk += 1
    if kkk % 5 == 0:
      line_hypo = ""
      for word_idx in item:
        word = get_key(word_map, word_idx)
        #print(word)
        line_hypo += word[0] + " "

      result_json_file_swap[str(kkk)] = []
      result_json_file_swap[str(kkk)].append(line_hypo)


      line_hypo += "\r\n"
      result_file_swap.write(line_hypo)
        
  kkk = -1
  for item in references_3:
    kkk += 1
    if kkk % 5 == 0:
      iii = 0
      reference_json_file_swap[str(kkk)] = []

      for sentence in item:
        line_repo = ""
        for word_idx in sentence:
          word = get_key(word_map, word_idx)
          line_repo += word[0] + " "
              
        reference_json_file_swap[str(kkk)].append(line_repo)

        line_repo += "\r\n"

        if iii == 0:
          reference_file1_swap.write(line_repo)
        if iii == 1:
          reference_file2_swap.write(line_repo)
        if iii == 2:
          reference_file3_swap.write(line_repo)
        if iii == 3:
          reference_file4_swap.write(line_repo)
        if iii == 4:
          reference_file5_swap.write(line_repo)
        iii += 1
  #-----------------------------------------------------------------

  #----------------------------------------------------------------- replace
  kkk = -1
  for item in hypotheses_4:
    kkk += 1
    if kkk % 5 == 0:
      line_hypo = ""
      for word_idx in item:
        word = get_key(word_map, word_idx)
        #print(word)
        line_hypo += word[0] + " "

      result_json_file_replace[str(kkk)] = []
      result_json_file_replace[str(kkk)].append(line_hypo)


      line_hypo += "\r\n"
      result_file_replace.write(line_hypo)
        
  kkk = -1
  for item in references_4:
    kkk += 1
    if kkk % 5 == 0:
      iii = 0
      reference_json_file_replace[str(kkk)] = []

      for sentence in item:
        line_repo = ""
        for word_idx in sentence:
          word = get_key(word_map, word_idx)
          line_repo += word[0] + " "
              
        reference_json_file_replace[str(kkk)].append(line_repo)

        line_repo += "\r\n"

        if iii == 0:
          reference_file1_replace.write(line_repo)
        if iii == 1:
          reference_file2_replace.write(line_repo)
        if iii == 2:
          reference_file3_replace.write(line_repo)
        if iii == 3:
          reference_file4_replace.write(line_repo)
        if iii == 4:
          reference_file5_replace.write(line_repo)
        iii += 1

  #-----------------------------------------------------------------

  #----------------------------------------------------------------- distract
  kkk = -1
  for item in hypotheses_5:
    kkk += 1
    if kkk % 5 == 0:
      line_hypo = ""
      for word_idx in item:
        word = get_key(word_map, word_idx)
        #print(word)
        line_hypo += word[0] + " "

      result_json_file_distract[str(kkk)] = []
      result_json_file_distract[str(kkk)].append(line_hypo)


      line_hypo += "\r\n"
      result_file_distract.write(line_hypo)
        
  kkk = -1
  for item in references_5:
    kkk += 1
    if kkk % 5 == 0:
      iii = 0
      reference_json_file_distract[str(kkk)] = []

      for sentence in item:
        line_repo = ""
        for word_idx in sentence:
          word = get_key(word_map, word_idx)
          line_repo += word[0] + " "
              
        reference_json_file_distract[str(kkk)].append(line_repo)

        line_repo += "\r\n"

        if iii == 0:
          reference_file1_distract.write(line_repo)
        if iii == 1:
          reference_file2_distract.write(line_repo)
        if iii == 2:
          reference_file3_distract.write(line_repo)
        if iii == 3:
          reference_file4_distract.write(line_repo)
        if iii == 4:
          reference_file5_distract.write(line_repo)
        iii += 1

  #-----------------------------------------------------------------


  # Calculate BLEU-4 scores

  weights1 = (1.0/1.0, )
  weights2=(1.0/2.0, 1.0/2.0,)
  weights3=(1.0/3.0, 1.0/3.0, 1.0/3.0,)
  weights4=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, )
  
  bleu1 = corpus_bleu(references, hypotheses, weights1)
  bleu2 = corpus_bleu(references, hypotheses, weights2)
  bleu3 = corpus_bleu(references, hypotheses, weights3)
  bleu4 = corpus_bleu(references, hypotheses, weights4)

  bleu1_0 = corpus_bleu(references_0, hypotheses_0, weights1)
  bleu2_0 = corpus_bleu(references_0, hypotheses_0, weights2)
  bleu3_0 = corpus_bleu(references_0, hypotheses_0, weights3)
  bleu4_0 = corpus_bleu(references_0, hypotheses_0, weights4)

  bleu1_1 = corpus_bleu(references_1, hypotheses_1, weights1)
  bleu2_1 = corpus_bleu(references_1, hypotheses_1, weights2)
  bleu3_1 = corpus_bleu(references_1, hypotheses_1, weights3)
  bleu4_1 = corpus_bleu(references_1, hypotheses_1, weights4)

  bleu1_2 = corpus_bleu(references_2, hypotheses_2, weights1)
  bleu2_2 = corpus_bleu(references_2, hypotheses_2, weights2)
  bleu3_2 = corpus_bleu(references_2, hypotheses_2, weights3)
  bleu4_2 = corpus_bleu(references_2, hypotheses_2, weights4)

  bleu1_3 = corpus_bleu(references_3, hypotheses_3, weights1)
  bleu2_3 = corpus_bleu(references_3, hypotheses_3, weights2)
  bleu3_3 = corpus_bleu(references_3, hypotheses_3, weights3)
  bleu4_3 = corpus_bleu(references_3, hypotheses_3, weights4)

  bleu1_4 = corpus_bleu(references_4, hypotheses_4, weights1)
  bleu2_4 = corpus_bleu(references_4, hypotheses_4, weights2)
  bleu3_4 = corpus_bleu(references_4, hypotheses_4, weights3)
  bleu4_4 = corpus_bleu(references_4, hypotheses_4, weights4)

  bleu1_5 = corpus_bleu(references_5, hypotheses_5, weights1)
  bleu2_5 = corpus_bleu(references_5, hypotheses_5, weights2)
  bleu3_5 = corpus_bleu(references_5, hypotheses_5, weights3)
  bleu4_5 = corpus_bleu(references_5, hypotheses_5, weights4)

  with open('eval_results_fortest/' + args.model_name + '_res.json','w') as f:
    json.dump(result_json_file,f)

  with open('eval_results_fortest/' + args.model_name + '_gts.json','w') as f:
    json.dump(reference_json_file,f)


  ## add
  with open('eval_results_fortest/' + args.model_name + '_add_res.json','w') as f:
    json.dump(result_json_file_add,f)

  with open('eval_results_fortest/' + args.model_name + '_add_gts.json','w') as f:
    json.dump(reference_json_file_add,f)


  ## delete
  with open('eval_results_fortest/' + args.model_name + '_delete_res.json','w') as f:
    json.dump(result_json_file_delete,f)

  with open('eval_results_fortest/' + args.model_name + '_delete_gts.json','w') as f:
    json.dump(reference_json_file_delete,f)

  ## move
  with open('eval_results_fortest/' + args.model_name + '_move_res.json','w') as f:
    json.dump(result_json_file_move,f)

  with open('eval_results_fortest/' + args.model_name + '_move_gts.json','w') as f:
    json.dump(reference_json_file_move,f)

  ## swap
  with open('eval_results_fortest/' + args.model_name + '_swap_res.json','w') as f:
    json.dump(result_json_file_swap,f)

  with open('eval_results_fortest/' + args.model_name + '_swap_gts.json','w') as f:
    json.dump(reference_json_file_swap,f)

  ## replace
  with open('eval_results_fortest/' + args.model_name + '_replace_res.json','w') as f:
    json.dump(result_json_file_replace,f)

  with open('eval_results_fortest/' + args.model_name + '_replace_gts.json','w') as f:
    json.dump(reference_json_file_replace,f)

  ## distract
  with open('eval_results_fortest/' + args.model_name + '_distract_res.json','w') as f:
    json.dump(result_json_file_distract,f)

  with open('eval_results_fortest/' + args.model_name + '_distract_gts.json','w') as f:
    json.dump(reference_json_file_distract,f)




  return bleu1, bleu2, bleu3, bleu4, bleu1_0, bleu2_0, bleu3_0, bleu4_0, bleu1_1, bleu2_1, bleu3_1, bleu4_1, bleu1_2, bleu2_2, bleu3_2, bleu4_2, bleu1_3, bleu2_3, bleu3_3, bleu4_3, bleu1_4, bleu2_4, bleu3_4, bleu4_4, bleu1_5, bleu2_5, bleu3_5, bleu4_5

def r2(bleu):
  result = float(int(bleu*10000.0)/10000.0)
  
  result = str(result)
  while len(result) < 6:
    result += "0"

  return result



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  

  parser.add_argument('--data_folder', default='dataset/for_train/3dcc_v0-2/con-sub_too_r3')
  parser.add_argument('--checkpoint', default='results/v0-2_total_concat_subtractBEST_checkpoint_3dcc_5_cap_per_img_0_min_word_freq.pth.tar')
  parser.add_argument('--word_map_file', default='dataset/for_train/3dcc_v0-2/con-sub_too_r3/WORDMAP_3dcc_5_cap_per_img_0_min_word_freq.json')
  parser.add_argument('--model_name', default='con-sub_too_r3')

  args = parser.parse_args()

  beam_size = 1
  n_gram = 4
  bleu1, bleu2, bleu3, bleu4, bleu1_0, bleu2_0, bleu3_0, bleu4_0, bleu1_1, bleu2_1, bleu3_1, bleu4_1, bleu1_2, bleu2_2, bleu3_2, bleu4_2, bleu1_3, bleu2_3, bleu3_3, bleu4_3, bleu1_4, bleu2_4, bleu3_4, bleu4_4, bleu1_5, bleu2_5, bleu3_5, bleu4_5 = evaluate(args, beam_size, n_gram)

  print('\n original: BLEU-1 - {bleu11}, BLEU-2 - {bleu22}, BLEU-3 - {bleu33}, BLEU-4 - {bleu44},\n      add: BLEU-1 - {bleu11_0}, BLEU-2 - {bleu22_0}, BLEU-3 - {bleu33_0}, BLEU-4 - {bleu44_0},\n     drop: BLEU-1 - {bleu11_1}, BLEU-2 - {bleu22_1}, BLEU-3 - {bleu33_1}, BLEU-4 - {bleu44_1},\n     move: BLEU-1 - {bleu11_2}, BLEU-2 - {bleu22_2}, BLEU-3 - {bleu33_2}, BLEU-4 - {bleu44_2},\n     swap: BLEU-1 - {bleu11_3}, BLEU-2 - {bleu22_3}, BLEU-3 - {bleu33_3}, BLEU-4 - {bleu44_3},\n  replace: BLEU-1 - {bleu11_4}, BLEU-2 - {bleu22_4}, BLEU-3 - {bleu33_4}, BLEU-4 - {bleu44_4},\n distract: BLEU-1 - {bleu11_5}, BLEU-2 - {bleu22_5}, BLEU-3 - {bleu33_5}, BLEU-4 - {bleu44_5},\n'.format(beam_size, bleu11=r2(bleu1), bleu22=r2(bleu2), bleu33=r2(bleu3), bleu44=r2(bleu4), bleu11_0=r2(bleu1_0), bleu22_0=r2(bleu2_0), bleu33_0=r2(bleu3_0), bleu44_0=r2(bleu4_0), bleu11_1=r2(bleu1_1), bleu22_1=r2(bleu2_1), bleu33_1=r2(bleu3_1), bleu44_1=r2(bleu4_1), bleu11_2=r2(bleu1_2), bleu22_2=r2(bleu2_2), bleu33_2=r2(bleu3_2), bleu44_2=r2(bleu4_2), bleu11_3=r2(bleu1_3), bleu22_3=r2(bleu2_3), bleu33_3=r2(bleu3_3), bleu44_3=r2(bleu4_3), bleu11_4=r2(bleu1_4), bleu22_4=r2(bleu2_4), bleu33_4=r2(bleu3_4), bleu44_4=r2(bleu4_4), bleu11_5=r2(bleu1_5), bleu22_5=r2(bleu2_5), bleu33_5=r2(bleu3_5), bleu44_5=r2(bleu4_5)))











































