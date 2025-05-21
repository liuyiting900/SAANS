import torch
import numpy as np
import bitarray
# from simctg.simctggpt import SimCTGGPT
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

def decode(self, token_ids, **kwargs):
    filtered_tokens = self.convert_ids_to_tokens(token_ids)
    text = self.convert_tokens_to_string(filtered_tokens)
    return text


GPT2Tokenizer.decode = decode


def _convert_token_to_id(self, token):
    return self.encoder.get(token, 0)


GPT2Tokenizer._convert_token_to_id = _convert_token_to_id

def limit_past(past):
    past = list(past)
    for i in range(len(past)):
        past[i] = past[i][:, :, :, -1022:]
    return past


def js(q, logq, logp):
    # q 和 logq 是分布 Q
    # logp 是分布 P 的对数概率（log P）
    # 所有 log 输入都是以 e 为底的对数

    m = 0.5 * (q + torch.exp(logp))  # 中间分布 M = 0.5 * (P + Q)
    logm = torch.log(m + 1e-12)

    # KL(Q || M)
    kl_qm = q * (logq - logm) / 0.69315
    kl_qm[q == 0] = 0

    # KL(P || M)
    p = torch.exp(logp)
    logp = logp  # already log P
    logm2 = logm  # same log M
    kl_pm = p * (logp - logm2) / 0.69315
    kl_pm[p == 0] = 0

    jsd = 0.5 * (kl_qm.sum() + kl_pm.sum())
    return jsd.item()



def kl(q, logq, logp):
    res = q * (logq - logp) / 0.69315
    res[q == 0] = 0
    return res.sum().item()


def entropy(q, logq):
    res = q * logq / 0.69315
    res[q == 0] = 0
    return -res.sum().item()  # in bits


# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


# 判断给定的标记索引对应的标记是否表示句子的结束
def is_sent_finish(token_idx, enc):
    token = enc.decoder[token_idx]
    return '.' in token or '!' in token or '?' in token


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break

    return i


def encode_context(raw_text, enc):
    context_tokens = [enc.encoder['<|endoftext|>']] + enc.encode(raw_text)
    return context_tokens


# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
def get_model(seed=1234, model_name='gpt2'):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    device = torch.device("cpu")  # 修改为 CPU

    enc = GPT2Tokenizer.from_pretrained(model_name)
    enc.unk_token = None
    enc.bos_token = None
    enc.eos_token = None

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    # model.double()

    return enc, model


enc32_itoc = ['\0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
              'u', 'v', 'w', 'x', 'y', 'z', '.', ',', "'", '!', ' ']
enc32_ctoi = {k: v for v, k in enumerate(enc32_itoc)}

# 将文本字符串编码为位表示形式
def enc32(text):
    bits = []
    for c in text:
        bits.extend(int2bits(enc32_ctoi[c], 5))
    return bits

# 将一系列位表示的文本转换为文本字符串。
def dec32(bits):
    text = ''
    for i in range(0, len(bits), 5):
        c = enc32_itoc[bits2int(bits[i:i + 5])]
        if c == '\0':
            break
        text += c
    return text


# message should be bit string
# encoded should be text string
def expansion_ratio(message, encoded):
    message_bits = len(message)
    encoded_ba = bitarray.bitarray()
    encoded_ba.frombytes(encoded.encode('utf-8'))
    encoded_bits = len(encoded_ba.tolist())
    return encoded_bits / message_bits


def count_elements(binary_string, split_size):
    # 补齐字符串长度，使得其长度是分割大小的整数倍
    remainder = len(binary_string) % split_size
    if remainder != 0:
        binary_string += '0' * (split_size - remainder)

    # 初始化元素字典，用于记录每个元素的出现次数
    element_counts = {}

    # 按指定大小分割字符串，并统计每个元素的出现次数
    for i in range(0, len(binary_string), split_size):
        element = binary_string[i:i + split_size]
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1

    # 添加出现次数为0的元素到字典中
    for i in range(2 ** split_size):
        element = format(i, '0' + str(split_size) + 'b')  # 格式化为固定长度的二进制字符串
        if element not in element_counts:
            element_counts[element] = 0

    # 按照出现次数从大到小对字典进行排序
    sorted_element_counts = dict(sorted(element_counts.items(), key=lambda item: item[1], reverse=True))

    return sorted_element_counts

# 找后一个元素
def get_next_element(element, cumulative_counts_dict):
    elements = list(cumulative_counts_dict.keys())
    index = elements.index(element)
    if index < len(elements) - 1:
        return elements[index + 1]
    else:
        return None

# 计算余数列表
def calculate_remainders(quotient,k):
    remainders = []
    while quotient >= k:
        quotient, remainder = divmod(quotient, k)
        remainders.append(remainder)
    remainders.append(quotient)
    return remainders[::-1]  # 将余数列表反转，使得最先计算的余数在列表的开头

def calculate_quotient(remainders):
    quotient = 0
    power = len(remainders) - 1
    for remainder in remainders:
        quotient += remainder * (10 ** power)
        power -= 1
    return quotient


# 将字符串编码为比特串
def encode_message(message_str, unicode_enc=True):
    if unicode_enc:
        ba = bitarray.bitarray()
        ba.frombytes(message_str.encode('utf-8'))
        return ba
    # else:
    # # 方法二：按照算术编码的形式编码
    #     message_ctx = [enc.encoder['<|endoftext|>']]
    #     message_str += '<eos>'
    #     message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=60000)








