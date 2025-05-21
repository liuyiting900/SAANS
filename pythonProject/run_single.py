import numpy as np
import bitarray
import sys
import re
import math
from utils import get_model, encode_context, encode_message

from arithmetic import encode_arithmetic, decode_arithmetic
from block_baseline import get_bins, encode_block, decode_block
from huffman_baseline import encode_huffman, decode_huffman
from sample import sample
from saac import encode_saac, decode_saac
from ADG import ADG_encoder
from SAANSNEW import encode_SAANS_ANS, decode_SAANSNEW


def main():
    global out, words_per_bit, kl, nll
    enc, model = get_model(model_name='gpt2')

    message_str = "This is secret message"
    mode = 'saans'  # algorithm
    block_size = 7   # for huffman and bins
    temp = 1.0  # for arithmetic and saans
    precision = 30  # for arithmetic ans s
    sample_tokens = 100  # for sample
    topk = 500  # for arithmetic
    nucleus = 0.9  # for saac and saans
    size = 8  # saans
    finish_sent = False


    ## VALIDATE PARAMETERS
    if mode not in ['arithmetic', 'huffman', 'bins', 'sample', 'saac', 'saans', 'ADG']:
        raise NotImplementedError

    if mode == 'bins':
        bin2words, words2bin = get_bins(len(enc.encoder), block_size)

    context = \
        """The US economy saw record growth in Q3.Analysts believe this was due to strong consumer spending.However, 
        some sectors remained below pre-pandemic levels."""
    context_tokens = encode_context(context, enc)
    message_bits = encode_message(message_str)
    print("Original message bits:", message_bits.tolist())

    # 方法一：按照Ascii码进行编码
    ba = bitarray.bitarray()
    ba.frombytes(message_str.encode('utf-8'))
    message = ba.tolist()
    # 利用任意上下文将比特编码为封面文本
    Hq = 0
    if mode == 'arithmetic':
        out, nll, kl, words_per_bit, Hq, time = encode_arithmetic(model, enc, message, context_tokens, temp=temp,
                                                                  finish_sent=finish_sent, precision=precision,
                                                                  topk=topk)
    elif mode == 'huffman':
        out, nll, kl, words_per_bit, time = encode_huffman(model, enc, message, context_tokens, block_size,
                                                           finish_sent=finish_sent)
    elif mode == 'bins':
        out, nll, kl, words_per_bit = encode_block(model, enc, message, context_tokens, block_size, bin2words,
                                                   words2bin, finish_sent=finish_sent)
    elif mode == 'sample':
        out, nll, kl, Hq = sample(model, enc, sample_tokens, context_tokens, temperature=temp, topk=topk)
        words_per_bit = 1
    elif mode == 'saac':
        out, nll, kl, words_per_bit, Hq, topk_list, case_studies = encode_saac(model, enc, message, context_tokens,
                                                                               temp=temp,
                                                                               precision=precision, topk=topk,
                                                                               nucleus=nucleus)
    elif mode == 'saans':
        out, nll, kl, words_per_bit, Hq, s= encode_SAANS_ANS(model, enc, message, context_tokens, size, nucleus,temp=temp,
                                                              finish_sent=finish_sent, precision=precision)

    elif mode == 'ADG':
        out, nll, kl, words_per_bit, Hq = ADG_encoder(model, enc, message, context_tokens, temp=temp,
                                                      finish_sent=finish_sent)

    text = enc.decode(out)

    print(message)
    print(len(message))
    print("=" * 40 + " Encoding " + "=" * 40)
    print(text)
    print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f' % (
        math.exp(nll), kl, words_per_bit, 1 / words_per_bit))

    # 由封面文本解码得到秘密消息
    if mode != 'sample':
        if mode == 'arithmetic':
            message_rec = decode_arithmetic(model, enc, text, context_tokens, temp=temp, precision=precision, topk=topk)
        elif mode == 'huffman':
            message_rec = decode_huffman(model, enc, text, context_tokens, block_size)
        elif mode == 'bins':
            message_rec = decode_block(model, enc, text, context_tokens, block_size, bin2words, words2bin)
        elif mode == 'saac':
            message_rec = decode_saac(model, enc, text, context_tokens, temp=temp,
                                      precision=precision, topk=topk, nucleus=nucleus)
        elif mode == 'saans':
            message_rec = decode_SAANSNEW(model, enc, text, context_tokens, s, size,nucleus,temp=temp, precision=precision)

    print("=" * 40 + " Recovered Message " + "=" * 40)
    print(message_rec)
    print(len(message_rec))
    print("=" * 90)
    message_rec = [bool(int(item)) for item in message_rec]
    ba = bitarray.bitarray(message_rec)
    reconst = ba.tobytes().decode('utf-8', 'ignore')
    print(reconst)


if __name__ == '__main__':
    main()
