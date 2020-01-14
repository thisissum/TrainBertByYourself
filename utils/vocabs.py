import json
import random
from collections import Counter


class Vocabulary(object):
    """vocabulary class used for build vocab and toknization
    """

    def __init__(self, vocab=None, pad=0, sos=1, eos=2, unk=3, mask=4):
        ori_dict = {
            '<pad>': pad, '<sos>': sos, '<eos>': eos, '<unk>': unk, '<mask>': mask
        }
        if vocab is None:
            self.vocab = ori_dict
        else:
            self.vocab = vocab.update(ori_dict)

    def sample(self):
        rand_num = random.randint(self.vocab['<mask>'] + 1, len(self.vocab) - 1)
        return list(self.vocab.values())[rand_num]

    def tokenize(self, words):
        """
        words:List[str]
        """
        return [self.vocab.get(word, self.vocab['<unk>']) for word in words]

    def add_word(self, word):
        _ = self.vocab.setdefault(word, len(self.vocab))

    def add_words(self, words):
        for word in set(words):
            self.add_word(word)

    def get_sos(self):
        return self.vocab['<sos>']

    def get_eos(self):
        return self.vocab['<eos>']

    def get_pad(self):
        return self.vocab['<pad>']

    def get_mask(self):
        return self.vocab['<mask>']

    def save(self, path):
        print('Saving vocabulary')
        with open(path, 'r', encoding='utf-8') as f:
            json.dump(self.vocab, f)
        print('Done')

    @classmethod
    def load(cls, path):
        print('Loading vocabulary')
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print('Done')
        return cls.__init__(vocab=vocab)

    @property
    def size(self):
        return len(self.vocab)


def build_vocab(iterator, cut_func, vocab=None, min_count=3):
    """Build a vocabulary
    args:
        iterator: iter, yield one sentence a time
        cut_func: function, used to cut sentence to List[str]
        vocab: Vocabulary, if not given, the default Vocabulary will be used
        min_count: int, min frequency of the word to add to the vocab
    return:
        vocab: Vocabulary, whose word id is reverse-sorted by it's freq in the corpus
    """
    print("Building Vocabulary")
    counter = Counter()
    if vocab is None:
        vocab = Vocabulary()
    for line in iterator:
        words = cut_func(line)
        counter.update(words)
    for key, freq in counter.most_common():
        if freq < min_count:
            continue
        vocab.add_word(key)
    print("Done")
    return vocab
