import torch
import torch.nn.functional as F

from utils import ( kl, entropy, bits2int
)


def encode_SAANS_ANS(
        model, enc, message_bits, context_ids, size,nucleus,
        finish_sent=True,device='cpu', temp=1.0,
        precision=30, verbose=True,
         epsilon_safe_gap=0.01
):
    context = torch.tensor(context_ids[-1022:], device=device).unsqueeze(0)
    past = None
    prev = context
    output_ids = []

    M = 2 ** precision
    s = bits2int(message_bits[:size])
    bit_pos = size

    total_logp = 0.0
    total_kl = 0.0
    total_Hq = 0.0
    count = 0

    def encode_one_step(s, prev, past):
        MIN_EPSILON = 1e-6
        FREQ_MIN = 32

        logits, past = model(prev, past=past)
        # past = limit_past(past)
        # logits[0, -1, -1] = -1e20  # endoftext token can't happen
        # logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
        logits = logits[0, -1] / temp
        probs = F.softmax(logits, dim=-1)
        logp = F.log_softmax(logits, dim=-1)

        sorted_probs, sorted_idx = probs.sort(descending=True)
        cum_probs = sorted_probs.cumsum(dim=0)
        mask = cum_probs <= nucleus
        k = max(int(mask.sum()), 2)


        top_probs = sorted_probs[:k]
        top_idx = sorted_idx[:k]
        top_probs = top_probs / top_probs.sum()

        freq = torch.round(top_probs * M).long()
        freq = torch.clamp(freq, min=FREQ_MIN)
        delta = M - freq.sum()
        freq[0] += delta
        cdf = torch.cat([torch.zeros(1, dtype=torch.long, device=device), freq.cumsum(dim=0)])

        u = s % M

        def find_valid_index(eps):
            if eps is not None:
                safe_threshold = int(eps * M)
                gap = cdf[1:] - cdf[:-1]
                sel = (u >= cdf[:-1]) & (u < cdf[1:]) & (gap >= safe_threshold)
            else:
                sel = (u >= cdf[:-1]) & (u < cdf[1:])
            idxs = sel.nonzero()
            return idxs.view(-1)[0].item() if idxs.numel() > 0 else None

        final_eps = epsilon_safe_gap
        idx = None
        while final_eps >= MIN_EPSILON:
            idx = find_valid_index(final_eps)
            if idx is not None:
                break
            final_eps /= 2

        if idx is None:
            idx = find_valid_index(None)
            if idx is None:
                raise ValueError(f"No valid interval found for u={u}")

        token_id = top_idx[idx].item()
        s_new = (s // M) * freq[idx].item() + (u - cdf[idx].item())

        return token_id, s_new, past, logp, top_probs, top_idx, u, freq[idx].item(), cdf[idx].item(), k, final_eps

    step = 0
    while True:
        if step > 0:
            for _ in range(size):
                b = message_bits[bit_pos] if bit_pos < len(message_bits) else 0
                s = (s << 1) | b
                bit_pos += 1

        s_before = s
        token_id, s, past, logp, top_probs, top_idx, u, f, base, k, final_eps = encode_one_step(s, prev, past)

        output_ids.append(token_id)
        prev = torch.tensor([[token_id]], device=device)

        if verbose:
            idx_tensor = (top_idx == token_id).nonzero()
            idx = idx_tensor.view(-1)[0].item()
            print(f"[Encode] step={step}, k={k}, token_id={token_id}, idx={idx}, prob={top_probs[idx].item():.5f}, "
                 f"u={u}, f={f}, base={base}, s={s}, s_before={s_before}, bit_pos={bit_pos}, final_eps={final_eps}")

        total_logp += logp[token_id].item()
        q = top_probs.cpu().double()
        logq = q.log()
        logp_top = logp[top_idx].cpu().double()
        total_kl += kl(q, logq, logp_top)
        total_Hq += entropy(q, logp_top)
        count += 1

        if finish_sent:
            text = enc.decode(output_ids)
            if '<eos>' in text:
                break
        else:
            if bit_pos >= len(message_bits):
                break

        step += 1

    avg_NLL = - total_logp / count
    avg_KL = total_kl / count
    avg_Hq = total_Hq / count
    bits_per_token = count /bit_pos

    return output_ids, avg_NLL, avg_KL, bits_per_token, avg_Hq, s

def decode_SAANSNEW(model, enc, text, context, s, size,nucleus,device='cpu', temp=1.0, precision=30,verbose=True):
    print("=" * 40 + " Decoding " + "=" * 40)
    M = 2 ** precision
    print("s:", s, "M:", M)

    inp = enc.encode(text)

    # BPE 修复：处理 token 628 → 198 + 198
    i = 0
    while i < len(inp):
        if inp[i] == 628:
            inp[i] = 198
            inp[i + 1:i + 1] = [198]
            i += 2
        else:
            i += 1

    context_tensor = torch.tensor(context[-1022:], device=device).unsqueeze(0)
    past = None

    cdf_list = []

    # 正向计算所有 token 的 CDF 信息
    with torch.no_grad():
        for t in range(1, len(inp)):
            token = inp[t]
            vocab_size = model.config.vocab_size
            if torch.any(context_tensor >= vocab_size):
                print("[Error] context_tensor 中出现超出 vocab 的 token！")
                print("最大 token id:", torch.max(context_tensor).item(), "词表上限:", vocab_size - 1)
                raise ValueError("Invalid token in context.")

            logits, past = model(context_tensor, past=past)
            logits = logits[0, -1] / temp
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_idx = probs.sort(descending=True)
            cum_probs = sorted_probs.cumsum(dim=0)

            mask = cum_probs <= nucleus
            k = max(int(mask.sum()), 2)
            top_probs = sorted_probs[:k]
            top_idx = sorted_idx[:k]
            top_probs /= top_probs.sum()

            freq = torch.round(top_probs * M).long()
            delta = M - freq.sum()
            freq[0] += delta
            cdf = torch.cat([torch.zeros(1, dtype=torch.long, device=device), freq.cumsum(0)], dim=0)

            cdf_list.append({
                'token': token,
                'top_idx': top_idx,
                'freq': freq,
                'cdf': cdf
            })

            # context_tensor = torch.cat([context_tensor, torch.tensor([[token]], device=device)], dim=1)
            context_tensor = torch.tensor([[token]], device=device)

            # print("cdf_list:", cdf_list)

    X = []

    # 反向执行 ANS 解码
    for t in range(len(cdf_list) - 1, -1, -1):
        token = cdf_list[t]['token']
        top_idx = cdf_list[t]['top_idx']
        freq = cdf_list[t]['freq']
        cdf = cdf_list[t]['cdf']

        idx_in_top = (top_idx == token).nonzero()
        if idx_in_top.numel() == 0:
            print(f"[Error] Token {token} not in top-k at t={t}. Skipping.")
            continue

        j = idx_in_top.view(-1)[0].item()
        f = freq[j].item()
        base = cdf[j].item()
        u = s % f
        s_prev=s

        try:
            s = (s // f) * M + u + base
            if verbose:
                print(f"[Decode] Token = {token}, idx = {j}, f = {f}, base = {base}, u = {u}, s_prev={s_prev}, s = {s}")
        except ZeroDivisionError:
            print(f"[Warning] f == 0 at t={t}, skipping token.")
            continue
        # 低位比特提取
        # while s > M:
        bit_count = size if t == 0 else size
        for _ in range(bit_count):
            X.append(s & 1)
            s >>= 1
        # print(f"[Bit Extract] {X}")

    print(f"Final recovered bitstream length: {len(X)}")
    X_rev = X[::-1]
    first4 = X_rev[:size][::-1]  # 反转前4位
    rest = X_rev[size:]  # 其余不变
    return first4 + rest

