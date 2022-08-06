from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.datasets import IWSLT2017
import pdb


import io
import re
from torchtext.vocab import build_vocab_from_iterator
import spacy
import pdb

CLEANR = re.compile('<.*?>') 

def load_data():
    
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # REPLACED By manually download files
    # train_iter, valid_iter, test_iter = IWSLT2017(
    #     root='.data',
    #     split=('train', 'valid', 'test'),
    #     language_pair=('en', 'de')
    # )


    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<PAD>'
    UNK_WORD = '<UNK>'

    src_file = '2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo/train.tags.en-de.en'
    tgt_file = '2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo/train.tags.en-de.de'


    def yield_tokens(file_path):
        with io.open(file_path, encoding = 'utf-8') as f:
            for line in f:
                yield re.sub(CLEANR, '', line.strip()).split()

    vocab_en = build_vocab_from_iterator(yield_tokens(src_file), specials=[BOS_WORD, EOS_WORD, BLANK_WORD, UNK_WORD], min_freq=5)
    vocab_en.set_default_index(-1)

    vocab_de = build_vocab_from_iterator(yield_tokens(tgt_file), specials=[BOS_WORD, EOS_WORD, BLANK_WORD, UNK_WORD], min_freq=5)
    vocab_de.set_default_index(-1)

    # Prepare train validation test dataset
    # with filter out some maxlen.
    
    
    pdb.set_trace()

    # SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    # TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
    #                  eos_token = EOS_WORD, pad_token=BLANK_WORD)

    # MAX_LEN = 100
    # train, val, test = datasets.IWSLT.splits(
    #     exts=('.de', '.en'), fields=(SRC, TGT), 
    #     filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
    #         len(vars(x)['trg']) <= MAX_LEN)
    # MIN_FREQ = 2
    # SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    # TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    return train, val, test, SRC, TGT

class MyIterator(Dataset):
    pass
    # def create_batches(self):
    #     if self.train:
    #         def pool(d, random_shuffler):
    #             for p in data.batch(d, self.batch_size * 100):
    #                 p_batch = data.batch(
    #                     sorted(p, key=self.sort_key),
    #                     self.batch_size, self.batch_size_fn)
    #                 for b in random_shuffler(list(p_batch)):
    #                     yield b
    #         self.batches = pool(self.data(), self.random_shuffler)
            
    #     else:
    #         self.batches = []
    #         for b in data.batch(self.data(), self.batch_size,
    #                                       self.batch_size_fn):
    #             self.batches.append(sorted(b, key=self.sort_key))