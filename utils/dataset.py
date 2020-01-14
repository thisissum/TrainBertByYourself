import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from seq import pad_sequence
import random
from vocabs import Vocabulary, build_vocab


class BertDataset(IterableDataset):
    """Implementation of dataset used to load data from file
    """

    def __init__(self, path, vocab, cut_fc, max_len=512, sentence_splited=False, task='SOP', min_len=5):
        super(BertDataset, self).__init__()
        self.path = path
        self.task = task
        self.max_len = max_len
        self.min_len = min_len
        self.sentence_splited = sentence_splited
        self.cut_fc = cut_fc

        corpus_length = 0
        with open(path, 'r', encoding='utf-8') as f:
            for i in f:
                corpus_length += 1
        self.length = corpus_length

        if isinstance(vocab, str):
            self.vocab = Vocabulary.load(vocab)
        elif isinstance(vocab, Vocabulary):
            self.vocab = vocab

    # def __getitem__(self, index):
    #     with open(self.path, 'r', encoding='utf-8') as f:
    #         for sentence in f:
    #             if index == self.length:
    #                 raise IndexError()
    #             if len(sentence)> self.min_len:
    #                 yield self._pipeline(sentence)

    # def cut(self, sentence):
    #     """
    #     Method to be overwritten, cut sentence to list of words
    #     :param sentence: str, sentence to be cut into words or char
    #     :return: list[str], list of words or char
    #     """
    #     return sentence.replace('\n', '').split('_')

    def __iter__(self):
        # fetch a sentence
        with open(self.path, 'r', encoding='utf-8')  as f:
            for sentence in f:
                if len(sentence) > self.min_len:
                    yield self._pipeline(sentence)

    def _tokenize(self, words):
        return self.vocab.tokenize(words)

    def _pipeline(self, sentence):
        tokens = self._tokenize(self.cut_fc(sentence))

        sent1, sent2, is_right_order = self._SOP(tokens)
        sent1_random, sent1_label = self._MLM(sent1)
        sent2_random, sent2_label = self._MLM(sent2)

        sent1 = [self.vocab.get_sos()] + sent1_random + [self.vocab.get_eos()]
        sent2 = sent1_random + [self.vocab.get_eos()]
        sentence_tokens = pad_sequence(
            sent1 + sent2,
            padding=self.vocab.get_pad(),
            length=self.max_len
        )

        sent1_label = [self.vocab.get_pad()] + sent1_label + [self.vocab.get_pad()]
        sent2_label = sent2_label + [self.vocab.get_pad()]
        mlm_labels = pad_sequence(
            sent1_label + sent2_label,
            padding=self.vocab.get_pad(),
            length=self.max_len
        )

        seg_labels = pad_sequence(
            [1 for _ in range(len(sent1))] + [2 for _ in range(len(sent2))],
            padding=self.vocab.get_pad(),
            length=self.max_len
        )

        output = {
            'sentences': torch.tensor(sentence_tokens),
            'mlm_labels': torch.tensor(mlm_labels),
            'sop_label': is_right_order,
            'seg_labels': torch.tensor(seg_labels)
        }
        return output

    def _SOP(self, tokens):
        half = len(tokens) // 2
        sent1, sent2 = tokens[:half], tokens[half:]
        rand_num = random.random()
        if rand_num > 0.5:
            sent1, sent2 = sent2, sent1
            is_right_order = 0
        else:
            is_right_order = 1
        return sent1, sent2, is_right_order

    def _MLM(self, tokens):
        length = len(tokens)
        mlm_labels = [0 for i in range(length)]
        for i in range(length):
            replace_word = (random.random() > 0.85)
            if replace_word:
                mlm_labels[i] = tokens[i]
                replace_method_prob = random.random()
                if replace_method_prob < 0.8:
                    # replace with mask
                    tokens[i] = self.vocab.get_mask()
                elif replace_method_prob < 0.9:
                    # replace with random word
                    tokens[i] = self.vocab.sample()
                else:
                    pass
            else:
                pass
        return tokens, mlm_labels

    def __len__(self):
        return self.length

def build_dataloader(path, cut_fc, batch_size=64, max_len=512, sentence_splited=False, task='SOP', min_len=5):
    def reader():
        with open(path, 'r', encoding='utf-8') as f:
            for i in f:
                yield i

    vocab = build_vocab(reader(), cut_fc)
    dataset = BertDataset(path, vocab, cut_fc, max_len, sentence_splited, task, min_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader, vocab


if __name__ == '__main__':

    path = '../data/corpus.txt'

    def reader():
        with open(path, 'r', encoding='utf-8') as f:
            for i in f:
                yield i


    def cut(line):
        return line.replace('\n', '').split('_')


    vocab = build_vocab(reader(), cut)
    dataset = BertDataset(path, vocab, cut_fc=cut)
    dataloader = DataLoader(dataset, batch_size=64)
    print(dataset.length)
    count = 0
    for j in range(3):
        for i in dataloader:
            # for key, value in i.items():
            #     print(key)
            #     print(value)
            count += 1
            if count % 1000 == 0:
                print(count)
        print('---------------------------------------done{}-------------------------------------------'.format(count))
