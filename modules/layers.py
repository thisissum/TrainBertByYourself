import torch
from torch import nn
import math
import torch.nn.functional as F

class Mask(nn.Module):
    """
    Mask layer, 0 for pad
    """

    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, x):
        return (x != 0).float()


class BertEmbedding(nn.Module):
    """Add TokenEmbedding, PositionEmbedding, SegmentEmbedding together
    """

    def __init__(self, vocab_size, emb_dim, max_len=512, dropout=0.1):
        super(BertEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.dropout = dropout
        self.pos_emb_layer = PositionEmbedding(emb_dim, max_len)
        self.seg_emb_layer = SegmentEmbedding(emb_dim)
        self.token_emb_layer = TokenEmbedding(vocab_size, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, seq, seg_label):
        x = self.pos_emb_layer(seq) + self.token_emb_layer(seq) + self.seg_emb_layer(seg_label)
        output = self.dropout_layer(x)
        return output


class TokenEmbedding(nn.Embedding):
    """The random initialized token embedding
    """

    def __init__(self, vocab_size, emb_dim):
        super(TokenEmbedding, self).__init__(vocab_size, emb_dim, padding_idx=0)


class PositionEmbedding(nn.Module):
    """The position embedding with 'cos' and 'sin'
    """

    def __init__(self, emb_dim, max_len=512):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len
        pos_emb = torch.zeros(max_len, emb_dim)

        pos = torch.arange(0, max_len).float().unsqueeze(-1)
        div = (torch.arange(0, emb_dim, 2).float() * (-1) * math.log(10000.0) / emb_dim).exp()
        pos_emb[:, 0::2] = torch.sin(pos * div)
        pos_emb[:, 1::2] = torch.cos(pos * div)
        self.pos_emb = pos_emb.unsqueeze(0)
        self.pos_emb.requires_grad = False

    def forward(self, x):
        return self.pos_emb[:, :x.size(1)].to(x.device)


class SegmentEmbedding(nn.Embedding):
    """Segment embedding is used to identify whether it's the first or second sentence
    """

    def __init__(self, emb_dim):
        super(SegmentEmbedding, self).__init__(3, emb_dim, padding_idx=0)


# class MultiHeadAttention(nn.Module):
#     """Multi head self attention in transformer
#     """
#
#     def __init__(self, hidden_dim, head_num=4, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.head_num = head_num
#         self.dropout = dropout
#         self.dim_per_head = hidden_dim // head_num
#         self.Q = nn.Linear(hidden_dim, self.dim_per_head * head_num)
#         self.K = nn.Linear(hidden_dim, self.dim_per_head * head_num)
#         self.V = nn.Linear(hidden_dim, self.dim_per_head * head_num)
#         self.dropout_layer = nn.Dropout(dropout)
#
#     def forward(self, query, key, value, mask=None):
#         # shape(mask) = batch_size, seq_len
#         # shape(query, key, value) = batch_size, seq_len, hidden_dim
#         batch_size, _, hidden_dim = query.shape
#
#         q = self.Q(query).view(batch_size, self.head_num, -1, self.dim_per_head)
#         k = self.K(key).view(batch_size, self.head_num, -1, self.dim_per_head)
#         v = self.V(value).view(batch_size, self.head_num, -1, self.dim_per_head)
#
#         # shape = batch_size, head_num, seq_len, seq_len
#         scores = q.matmul(k.permute(0, 1, 3, 2)) / math.sqrt(self.dim_per_head)
#         if mask is not None:
#             mask = mask.unsqueeze(1).unsqueeze(1).expand_as(scores)
#             scores = scores.masked_fill(mask.permute(0, 1, 3, 2) == 0, -1e9)
#
#         weights = torch.softmax(scores, dim=-1)
#
#         if self.dropout:
#             weights = self.dropout_layer(weights)
#
#         output = weights.matmul(v).contiguous().view(batch_size, -1, self.dim_per_head * self.head_num)
#         return output

class MultiHeadAttention(nn.Module):
    """
    Multi head self attention in transformer
    """

    def __init__(self, hidden_dim, head_num=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout = dropout

        self.dim_per_head = hidden_dim // head_num
        self.Q = nn.Linear(hidden_dim, self.dim_per_head*head_num)
        self.K = nn.Linear(hidden_dim, self.dim_per_head*head_num)
        self.V = nn.Linear(hidden_dim, self.dim_per_head*head_num)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # shape(x) = batch_size, seq_len, dk
        # shape(mask) = batch_size, seq_len

        #shape(q_list[i]) = batch_size, seq_len ,dim_per_head
        q_list = self.Q(q).split(self.dim_per_head, dim=-1)
        k_list = self.K(k).split(self.dim_per_head, dim=-1)
        v_list = self.V(v).split(self.dim_per_head, dim=-1)

        output = []
        for i in range(self.head_num):
            scores = q_list[i].matmul(k_list[i].permute(0,2,1)) / math.sqrt(10)
            if mask is not None:
                if i == 0: # for less computational cost
                    mask = mask.unsqueeze(1).expand_as(scores)
                scores.masked_fill_(mask==0, -1e9)
            logits = torch.softmax(scores, dim=-1)
            value = logits.matmul(v_list[i])
            output.append(value)
        output = torch.cat(output, dim=-1)
        return output





class Connection(nn.Module):
    """Implementation of  'add and layer norm'
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(Connection, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, sub_layer):
        return x + self.dropout_layer(sub_layer(self.layer_norm(x)))


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, projection_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout_layer(F.gelu(self.fc1(x))))


class TransformerEncoder(nn.Module):
    """Implementation of transformer encoder
    """

    def __init__(self, hidden_dim, head_num=4, projection_dim=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.head_num = head_num
        self.dropout = dropout
        self.attention_layer = MultiHeadAttention(
            hidden_dim=hidden_dim,
            head_num=head_num,
            dropout=dropout
        )
        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.first_skip = Connection(hidden_dim, dropout=dropout)
        self.second_skip = Connection(hidden_dim, dropout=dropout)

    def forward(self, x, mask=None):
        output = self.first_skip(x, lambda z: self.attention_layer(z, z, z, mask))
        output = self.second_skip(output, self.feed_forward)
        return self.dropout_layer(output)


class MaskedLanguageModel(nn.Module):
    """The Mask Language Model task output layer
    """

    def __init__(self, hidden_dim, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.fc(x)


class SentenceOrderPrediction(nn.Module):
    """The Sentence Order Prediction task output layer
    """

    def __init__(self, hidden_dim):
        super(SentenceOrderPrediction, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.fc(x[:, 0, :])
