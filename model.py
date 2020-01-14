import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from modules.losses import MLMLoss, SOPLoss
from modules.layers import BertEmbedding, TransformerEncoder, SentenceOrderPrediction, MaskedLanguageModel, Mask


class BertModel(nn.Module):
    """Raw Bert Model
    """

    def __init__(
            self,
            vocab_size,
            emb_dim,
            hidden_dim,
            layer_num=8,
            head_num=4,
            max_len=512,
            dropout=0.1):
        super(BertModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.max_len = max_len
        self.dropout = dropout
        self.layer_num = layer_num

        self.emb_layer = BertEmbedding(vocab_size, emb_dim, max_len, dropout)
        if emb_dim != hidden_dim:
            self.factorization = True
            self.factorize = nn.Linear(emb_dim, hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoder(hidden_dim, head_num, hidden_dim * 4, dropout) for _ in range(layer_num)]
        )  # bert with 12 transformer encoder with different parameters

    def forward(self, seq, mask=None, seg_label=None):
        device = seq.device
        if seg_label is None:
            if mask:
                seg_label = mask.long()
            else:
                seg_label = torch.zeros(seq.shape).to(device)
        x = self.emb_layer(seq, seg_label)
        if self.factorization:  # embedding matrix factorization in albert
            x = self.factorize(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        return x


class Bert4Pretrain(nn.Module):
    """Bert model with mlm and sop output layer
    """

    def __init__(
            self,
            vocab_size,
            emb_dim=128,
            hidden_dim=768,
            layer_num=8,
            head_num=4,
            max_len=512,
            dropout=0.1):
        super(Bert4Pretrain, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.max_len = max_len
        self.dropout = dropout
        self.layer_num = layer_num

        self.mask = Mask()
        self.bert_model = BertModel(
            vocab_size,
            emb_dim,
            hidden_dim,
            layer_num,
            head_num,
            max_len,
            dropout)
        self.mlm = MaskedLanguageModel(hidden_dim, vocab_size)
        self.sop = SentenceOrderPrediction(hidden_dim)

    def forward(self, seq, seg_label=None):
        mask = self.mask(seq)
        bert_emb = self.bert_model(seq, mask, seg_label)
        mlm_out = self.mlm(bert_emb)
        sop_out = self.sop(bert_emb)
        return mlm_out, sop_out

    def get_bert_model(self):
        return self.bert_model


class BertTrainer(object):
    """
    BertTrainer
    args:
        bert_model: instance of Bert4Pretrain
        epoch: int, num of epoch
        lr: float, learning rate, default 0.0001
        device: torch.device or str, cpu or cuda, if None, use cuda if avaliable
        display_per_step: int
    """

    def __init__(
            self,
            bert_model,
            epoch,
            save_path,
            lr=0.0001,
            device=None,
            display_per_step=1000,
            save_per_step=10000
    ):
        self.epoch = epoch
        self.lr = lr
        self.save_path = save_path
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = bert_model.to(self.device)
        self.display_per_step = display_per_step
        self.save_per_step = save_per_step

        # mask language model and sentence order prediction loss, adamw
        # optimizer
        self.mlm_criterion = MLMLoss()
        self.sop_criterion = SOPLoss()
        self.optimizer = torch.optim.AdamW(self.bert.parameters(), lr=lr)

    def fit(self, train_loader):
        count = 0
        for _ in range(self.epoch):
            mlm_acc_list = []
            sop_acc_list = []
            for data in tqdm(train_loader):
                mlm_acc, sop_acc = self.iteration(data)
                mlm_acc_list.append(mlm_acc)
                sop_acc_list.append(sop_acc)
                count += 1
                if count % self.display_per_step == 0:
                    # display information
                    cur_mlm_acc = np.mean(mlm_acc_list)
                    cur_sop_acc = np.mean(sop_acc_list)
                    print(
                        'current mlm acc: {}\tcurrent sop acc:{}'.format(
                            cur_mlm_acc, cur_sop_acc))

                if count % self.save_per_step == 0:
                    # save bert model
                    self.save()

    def iteration(self, data_dict):
        # fetch data
        sentences = data_dict['sentences'].to(self.device)
        mlm_labels = data_dict['mlm_labels'].to(self.device)
        sop_labels = data_dict['sop_label'].to(self.device)
        seg_labels = data_dict['seg_labels'].to(self.device)

        # use bert to predict masked word and sentence order jointly
        mlm_pred, sop_pred = self.bert(sentences, seg_labels)

        # compute loss
        mlm_loss = self.mlm_criterion(mlm_pred, mlm_labels)
        sop_loss = self.sop_criterion(sop_pred, sop_labels)
        loss = mlm_loss + sop_loss

        # update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # compute accuracy of this batch
        mlm_acc_num = (mlm_pred[mlm_labels != 0].argmax(
            dim=1) == mlm_labels[mlm_labels != 0]).float().sum().item()
        sop_acc_num = (sop_pred.argmax(dim=1) == sop_labels).float().sum().item()
        mlm_acc = mlm_acc_num / mlm_labels[mlm_labels != 0].size(0)
        sop_acc = sop_acc_num / sop_labels.size(0)
        return mlm_acc, sop_acc

    def save(self, count):
        print('Saving model...')
        torch.save(
            self.bert.get_bert_model(),
            self.save_path +
            '/checkpoint{}.bin'.format(count))
        print('Done!')
