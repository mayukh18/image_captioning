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

data_name = '3dcc_5_cap_per_img_0_min_word_freq'  # base name shared by data files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

captions_per_image = 5
batch_size = 1

print(device)


# model_name


def get_key(dict_, value):
    return [k for k, v in dict_.items() if v == value]


def evaluate(args, beam_size, n_gram):
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location='cuda:0')

    net = checkpoint['net']
    net = net.to(device)
    net.eval()

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

    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """

    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, word_map, 'test'),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT
    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    # For each image
    ddd = 0
    for i, (image1, image2, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        if ddd == 5000:
            break
        current_index = i
        ddd += 1

        k = beam_size

        # Move to GPU device, if available
        image1 = image1.to(device)  # (1, 768, 16, 16)
        image2 = image2.to(device)  # (1, 768, 16, 16)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[0]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1

        img_feat1 = net(image1)
        img_feat2 = net(image2)

        l_bef, l_aft, alpha_bef, alpha_aft = encoder(img_feat1, img_feat2)

        l_diff = torch.sub(l_aft, l_bef)

        l_total = torch.cat([l_bef, l_aft, l_diff], dim=1)  # increase dim to fit k

        l_total = decoder.relu(decoder.wd1(l_total)).repeat(k, 1)

        h_da = torch.zeros(k, decoder.hidden_dim).to(device)  ## TODO ## random?
        c_da = torch.zeros(k, decoder.hidden_dim).to(device)

        h_ds = torch.zeros(k, decoder.hidden_dim).to(device)
        c_ds = torch.zeros(k, decoder.hidden_dim).to(device)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            u_t = torch.cat([l_total, h_ds], dim=1)
            h_da, c_da = decoder.dynamic_att(u_t, (h_da, c_da))

            a_t = decoder.softmax(decoder.wd2(h_da))

            l_dyn = a_t[:, 0].unsqueeze(1) * l_bef + a_t[:, 1].unsqueeze(1) * l_aft + a_t[:, 2].unsqueeze(1) * l_diff

            c_t = torch.cat([embeddings, l_dyn], dim=1)

            h_ds, c_ds = decoder.decode_step(c_t, (h_ds, c_ds))  # (s, decoder_dim)

            scores = decoder.wdc(h_ds)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # print(top_k_words, vocab_size)
            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # print(prev_word_inds, next_word_inds)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step + 1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != 1]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            h_ds = h_ds[prev_word_inds[incomplete_inds]]
            c_ds = c_ds[prev_word_inds[incomplete_inds]]
            h_da = h_da[prev_word_inds[incomplete_inds]]
            c_da = c_da[prev_word_inds[incomplete_inds]]
            l_total = l_total[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # print(seqs)
            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in [0, 1, 2]],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        temptemp = [w for w in seq if w not in [0, 1, 2]]
        hypotheses.append(temptemp)

    # Calculate BLEU-4 score
    weights1 = (1.0 / 1.0,)
    weights2 = (1.0 / 2.0, 1.0 / 2.0,)
    weights3 = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,)
    weights4 = (1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0,)

    # print("REFS", references)
    import numpy as np
    print(np.mean([len(x) for x in references]))
    # print("HYPO", hypotheses)

    bleu1 = corpus_bleu(references, hypotheses, weights1)
    bleu2 = corpus_bleu(references, hypotheses, weights2)
    bleu3 = corpus_bleu(references, hypotheses, weights3)
    bleu4 = corpus_bleu(references, hypotheses, weights4)
    return bleu1, bleu2, bleu3, bleu4


def r2(bleu):
    result = float(int(bleu * 10000.0) / 10000.0)

    result = str(result)
    while len(result) < 6:
        result += "0"

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/workspace/workspace/image_captioning')
    parser.add_argument('--checkpoint', default='BEST_checkpoint_3dcc_5_cap_per_img_0_min_word_freq.pth.tar')
    parser.add_argument('--word_map_file', default='/workspace/workspace/image_captioning/word_map.json')
    parser.add_argument('--model_name', default='con-sub_too_r3')

    args = parser.parse_args()

    beam_size = 1
    n_gram = 4
    bleu1, bleu2, bleu3, bleu4 = evaluate(
        args, beam_size, n_gram)

    print(
        '\n original: BLEU-1 - {bleu11}, BLEU-2 - {bleu22}, BLEU-3 - {bleu33}, BLEU-4 - {bleu44},\n'.format(
            beam_size, bleu11=r2(bleu1), bleu22=r2(bleu2), bleu33=r2(bleu3), bleu44=r2(bleu4)))







