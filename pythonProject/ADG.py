import torch
import torch.nn.functional as F
import time
from utils import limit_past, kl, entropy, bits2int, int2bits, is_sent_finish, num_same_from_beg,decode

# MSB
# e.g. [0, 1, 1, 1] looks like 0111=7
def msb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits[::-1]):
        res += bit * (2 ** i)
    return res

def near(alist, anum):
    up = len(alist) - 1
    # print("up", up)
    if up == 0:
        return 0

    bottom = 0
    while up - bottom > 1:
        index = int((up + bottom) / 2)
        if alist[index] < anum:
            up = index
        elif alist[index] > anum:
            bottom = index
        else:
            return index
    if up - bottom == 1:
        if alist[bottom] - anum < anum - up:
            index = bottom
        else:
            index = up

    return index

def ADG_encoder(model, enc, message, context, finish_sent=False, device='cpu', temp=1.0):
    torch.manual_seed(42)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    prev = context
    output = context
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0  # in bits
    total_entropy_ptau = 0
    total_num_sents = 0
    stega_sentence=[]


    with torch.no_grad():
        stega_text=[]
        j = 0
        sent_finish = False
        while j < len(message) or (finish_sent and not sent_finish):
            logits, past = model(prev.unsqueeze(0), past=past)  # 获取当前token的logits和past状态
            past = limit_past(past)  # 限制past状态的大小
            logits[0, -1, -1] = -1e20  # 禁止生成endoftext token
            logits[0, -1, 628] = -1e20  # 禁止生成两个新行的token
            # 获取最后一个token的logits并按降序排序
            sorted_logits, indices = logits[0, -1, :].sort(descending=True)
            # 将logits转换为双精度浮点数并根据温度参数进行缩放
            sorted_logits = sorted_logits.double()
            scaled_logits = sorted_logits / temp
            # 计算缩放后的概率分布
            probs_temp = F.softmax(scaled_logits, dim=0)

            # 计算缩放后的对数概率分布
            log_probs_temp = F.log_softmax(scaled_logits, dim=0)

            # 计算原始logits的对数概率分布
            log_probs = F.log_softmax(sorted_logits, dim=0)

            # 对logits进行数值稳定性处理
            log_prob = sorted_logits
            log_prob -= log_prob.max()
            prob = torch.exp(log_probs).reshape(-1)
            # prob[1] = 0
            #
            # # 重新归一化概率分布
            # prob = prob / prob.sum()  # 得到下一个词的条件概率分布
            # prob, indices = prob.sort(descending=True)
            # start recursion
            bit_tmp = 0
            #print(prob[0])
            while prob[0] <= 0.5:
                # print(prob[0])
                # embedding bit
                bit = 1
                while (1 / 2 ** (bit + 1)) > prob[0]:
                    bit += 1
                    # print(bit)
                mean = 1 / 2 ** bit
                #print('bit:',bit)
                # dp
                prob = prob.tolist()
                indices = indices.tolist()
                result = []
                for i in range(2 ** bit):
                    result.append([[], []])
                    # print(result)
                for i in range(2 ** bit - 1):
                    result[i][0].append(prob[0])
                    result[i][1].append(indices[0])
                    del (prob[0])
                    del (indices[0])
                    while sum(result[i][0]) < mean:
                        delta = mean - sum(result[i][0])
                        index = near(prob, delta)
                        if prob[index] - delta < delta:
                            result[i][0].append(prob[index])
                            result[i][1].append(indices[index])
                            del (prob[index])
                            del (indices[index])
                        else:
                            break
                    mean = sum(prob) / (2 ** bit - i - 1)
                result[2 ** bit - 1][0].extend(prob)
                result[2 ** bit - 1][1].extend(indices)
                # read secret message
                bit_embed = [int(_) for _ in message[bit_tmp: bit_tmp + bit]]

                # print("bit_embed:",bit_embed)
                int_embed = msb_bits2int(bit_embed)
                #print('int_embed:', int_embed)
                # updating
                prob = torch.FloatTensor(result[int_embed][0]).to(device)
                indices = torch.LongTensor(result[int_embed][1]).to(device)
                prob = prob / prob.sum()
                prob, _ = prob.sort(descending=True)
                indices = indices[_]
                bit_tmp += bit


                # Gather statistics
                total_log_probs += log_probs[int(torch.multinomial(prob, 1))].item()
                q = prob.double() / prob.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy(probs_temp, log_probs_temp)
                total_num_for_stats += 1

            # prev = indices[int(torch.multinomial(prob, 1))].view(1)
            prev = indices[0].view(1)
            # stega_sentence.append(int(prev))
            #print("stega", prev)
            # stega_text.append(enc.decode(stega_sentence))

            output = torch.cat((output, prev))

            j += bit_tmp

            # For text->bits->text
            partial = enc.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                break
            # break

        avg_NLL = -total_log_probs / total_num_for_stats
        avg_KL = total_kl / total_num_for_stats
        avg_Hq = total_entropy_ptau / total_num_for_stats
        words_per_bit = total_num_for_stats / j

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq