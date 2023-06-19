

import torch
import torch.nn as nn
# seq_len=7
# emb_size=5
pe = torch.ones(2,4,4)
# print(pe,pe.size())
#
# dropout=nn.Dropout(p=0.5)
# a=dropout(pe)
# print(a,a.size())
# kk
batch_size=4
len_q=2
len_k=3
seq_k=torch.rand(4,3)
seq_k[1][1:]=0
seq_k[3][2:]=0

print(seq_k[1])

pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
# print(pad_attn_mask,pad_attn_mask.size())
pad_attn_mask=pad_attn_mask.expand(batch_size, len_q, len_k)
print(pad_attn_mask[1])

# return pad_attn_mask.expand(batch_size, len_q, len_k)
