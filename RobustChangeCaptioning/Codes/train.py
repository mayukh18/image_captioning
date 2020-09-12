import json
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import DualAttention, DynamicSpeaker, ImNet
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu  ##--

import argparse

# Data parameters
data_name = '3dcc_5_cap_per_img_0_min_word_freq'

# Model parameters
embed_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
captions_per_image = 5
feature_dim = 1024

# Training parameters
start_epoch = 0
epochs_since_improvement = 0
batch_size = 64
workers = 1
decoder_lr = 1e-4
encoder_lr = 1e-4
grap_clip = 5.
alpha_c = 1.
best_bleu4 = 0.
print_freq = 100
checkpoint = None


def get_key(dict_, value):
    return [k for k, v in dict_.items() if v == value]


def main(args):
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, data_name

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'word_map.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    # Initialize
    encoder = DualAttention(attention_dim=args.attention_dim,
                            feature_dim=feature_dim).to(device)

    decoder = DynamicSpeaker(feature_dim=feature_dim,
                             embed_dim=embed_dim,
                             vocab_size=len(word_map),
                             hidden_dim=args.hidden_dim,
                             dropout=dropout).to(device)

    net = ImNet().to(device)

    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=encoder_lr)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, word_map, 'train'),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, word_map, 'train'),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Epochs

    for epoch in range(start_epoch, args.epochs):
        print("epoch : " + str(epoch))

        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(encoder_optimizer, 0.8)
            adjust_learning_rate(decoder_optimizer, 0.8)

        train(train_loader=train_loader,
              net=net,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              word_map=word_map
              )

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                word_map=word_map)

        # check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(args.root_dir, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, net, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, word_map):
    encoder.train()
    decoder.train()
    net.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top3accs = AverageMeter()

    start = time.time()

    # Batches
    for i, (imgs1, imgs2, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        im1_enc = net(imgs1)
        im2_enc = net(imgs2)

        # Forward prop.
        l_bef, l_aft, alpha_bef, alpha_aft = encoder(im1_enc, im2_enc)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(l_bef, l_aft, caps, caplens)

        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)

        # TODO
        # Add doubly stochastic attention regularization

        # Back prop.
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()

        # TODO
        # Grad_clip ??

        # Update weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Keep track of metrics
        top3 = accuracy(scores, targets, 3)
        losses.update(loss.item(), sum(decode_lengths))
        top3accs.update(top3, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss=losses,
                                                                          top3=top3accs))


def validate(val_loader, encoder, decoder, criterion, word_map):
    encoder.eval()
    decoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top3accs = AverageMeter()

    start = time.time()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        # Batches
        for i, (imgs1, imgs2, caps, caplens, allcaps) in enumerate(val_loader):
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            l_bef, l_aft, alpha_bef, alpha_aft = encoder(imgs1, imgs2)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(l_bef, l_aft, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets)

            # TODO
            # Add doubly stochastic attention regularization

            losses.update(loss.item(), sum(decode_lengths))
            top3 = accuracy(scores, targets, 3)
            top3accs.update(top3, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top3=top3accs))

            # References
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}], img_caps))
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

            weights1 = (1.0, 0.0, 0.0, 0.0)
            weights2 = (0.5, 0.5, 0.0, 0.0)
            weights3 = (0.33, 0.33, 0.33, 0.0)
            weights4 = (0.25, 0.25, 0.25, 0.25)

            bleu1 = corpus_bleu(references, hypotheses, weights1)
            bleu2 = corpus_bleu(references, hypotheses, weights2)
            bleu3 = corpus_bleu(references, hypotheses, weights3)
            bleu4 = corpus_bleu(references, hypotheses, weights4)

            print(
                '\n * LOSS - {loss.avg:.3f}, TOP-3 ACCURACY - {top3.avg:.3f}, BLEU-1 - {bleu11}, BLEU-2 - {bleu22}, BLEU-3 - {bleu33}, BLEU-4 - {bleu44},\n'.format(
                    loss=losses,
                    top3=top3accs,
                    bleu11=bleu1,
                    bleu22=bleu2,
                    bleu33=bleu3,
                    bleu44=bleu4, ))

            return bleu4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/content/')
    parser.add_argument('--root_dir', default='./')
    parser.add_argument('--epochs', type=int, default=42)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=512)

    args = parser.parse_args()

    main(args)






