import torch
from torch import nn
from torchvision import models
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImNet(nn.Module):

    def __init__(self):
        super(ImNet, self).__init__()
        model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        # print(self.model)

    def forward(self, img):
        print(img.size())
        out = self.model(img)
        print(out.size())
        return out


class DualAttention(nn.Module):
    """
  Dual attention network.
  """

    def __init__(self, attention_dim, feature_dim):
        """
    """
        super(DualAttention, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim * 2, attention_dim, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(attention_dim, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feat1, img_feat2):
        # img_feat1 (batch_size, feature_dim, h, w)
        batch_size = img_feat1.size(0)
        feature_dim = img_feat1.size(1)

        img_diff = img_feat2 - img_feat1

        img_feat1_d = torch.cat([img_feat1, img_diff], dim=1)
        img_feat2_d = torch.cat([img_feat2, img_diff], dim=1)

        img_feat1_d = self.conv1(img_feat1_d)
        img_feat2_d = self.conv1(img_feat2_d)

        img_feat1_d = self.relu(img_feat1_d)
        img_feat2_d = self.relu(img_feat2_d)

        img_feat1_d = self.conv2(img_feat1_d)
        img_feat2_d = self.conv2(img_feat2_d)

        # To this point
        # img_feat1, img_feat2 have dimension
        # (batch_size, hidden_dim, h, w)

        alpha_img1 = self.sigmoid(img_feat1_d)
        alpha_img2 = self.sigmoid(img_feat2_d)

        # To this point
        # alpha_img1, alpha_img2 have dimension
        # (batch_size, 1, h, w)

        img_feat1 = img_feat1 * (alpha_img1.repeat(1, 2048, 1, 1))
        img_feat2 = img_feat2 * (alpha_img2.repeat(1, 2048, 1, 1))

        # (batch_size,feature_dim,h,w)

        img_feat1 = img_feat1.sum(-2).sum(-1).view(batch_size, -1)
        img_feat2 = img_feat2.sum(-2).sum(-1).view(batch_size, -1)

        return img_feat1, img_feat2, alpha_img1, alpha_img2


class DynamicSpeaker(nn.Module):
    """
  Dynamic speaker network.
  """

    def __init__(self, feature_dim, embed_dim, vocab_size, hidden_dim, dropout):
        """
    """
        super(DynamicSpeaker, self).__init__()

        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=1)  ##### TODO #####

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)

        self.dynamic_att = nn.LSTMCell(hidden_dim * 2, hidden_dim, bias=True)

        self.decode_step = nn.LSTMCell(embed_dim + feature_dim, hidden_dim, bias=True)

        self.relu = nn.ReLU()
        self.wd1 = nn.Linear(feature_dim * 3, hidden_dim)
        self.wd2 = nn.Linear(hidden_dim, 3, )  ##### TODO #####
        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
    Initializes some parameters with values from the uniform distribution, for easier convergence
    """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.wd1.bias.data.fill_(0)
        self.wd1.weight.data.uniform_(-0.1, 0.1)
        self.wd2.bias.data.fill_(0)
        self.wd2.weight.data.uniform_(-0.1, 0.1)
        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, l_bef, l_aft, encoded_captions, caption_lengths):
        # To this point,
        # l_bef, l_aft have dimension
        # (batch_size, feature_dim)

        batch_size = l_bef.size(0)

        l_diff = torch.sub(l_aft, l_bef)

        l_total = torch.cat([l_bef, l_aft, l_diff], dim=1)
        l_total = self.relu(self.wd1(l_total))  # (batch_size, hidden_dim)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        l_diff = l_diff[sort_ind]
        l_total = l_total[sort_ind]
        l_bef = l_bef[sort_ind]
        l_aft = l_aft[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        h_da = torch.zeros(batch_size, self.hidden_dim).to(device)  ## TODO ## random?
        c_da = torch.zeros(batch_size, self.hidden_dim).to(device)

        h_ds = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_ds = torch.zeros(batch_size, self.hidden_dim).to(device)

        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, np.max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, np.max(decode_lengths), 3).to(device)  # TODO  ## is three ok?

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            u_t = torch.cat([l_total[:batch_size_t], h_ds[:batch_size_t]], dim=1)
            h_da, c_da = self.dynamic_att(u_t[:batch_size_t], (h_da[:batch_size_t], c_da[:batch_size_t]))

            a_t = self.softmax(self.wd2(h_da))  #### (batch_size, 3)

            l_dyn = a_t[:, 0].unsqueeze(1) * l_bef[:batch_size_t] + a_t[:, 1].unsqueeze(1) * l_aft[:batch_size_t] + a_t[
                                                                                                                    :,
                                                                                                                    2].unsqueeze(
                1) * l_diff[:batch_size_t]

            c_t = torch.cat([embeddings[:batch_size_t, t, :], l_dyn[:batch_size_t]], dim=1)

            h_ds, c_ds = self.decode_step(c_t, (h_ds[:batch_size_t], c_ds[:batch_size_t]))

            preds = self.wdc(h_ds)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = a_t

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind




















