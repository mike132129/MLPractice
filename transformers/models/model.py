import copy
from .embedding import Embeddings
from .positional_encoding import PositionalEncoding
from .attention import MultiHeadAttention
from .positionwide_feedforward import PositionwiseFeedForward
from .encoder_decoder import (
    EncoderDecoder,
    Encoder,
    EncoderLayer,
    Decoder,
    DecoderLayer,
    Generator,
)

import torch.nn as nn


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

if __name__ == '__main__':
    tmp_model = make_model(10, 10, 2)
    import pdb
    pdb.set_trace()