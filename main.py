from model import Bert4Pretrain, BertTrainer
from utils.dataset import build_dataloader

def cut(sentence):
    return sentence.replace('\n', '').split('_')

if __name__ == '__main__':
    path = './data/corpus.txt'
    dataloader, vocab = build_dataloader(path=path, cut_fc=cut, batch_size=16, max_len=128)
    bert_model = Bert4Pretrain(
        vocab_size=vocab.size,
        emb_dim=128,
        hidden_dim=512,
        layer_num=8,
        head_num=4,
        max_len=128
    )
    trainer = BertTrainer(bert_model=bert_model, epoch=20, save_path='./checkpoint', display_per_step=50)
    trainer.fit(dataloader)