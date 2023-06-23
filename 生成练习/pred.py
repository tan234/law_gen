import jieba
from config import *
import json
import torch
def generate_summary(doc_sentence, doc_field, sum_field, model, device, max_len = 50):
    # 将模型置为验证模式
    model.eval()

    # 对原文分词
    tokens=list(jieba.cut(doc_sentence))
    # nlp = spacy.load('en')
    #     nlp = spacy.load('en_core_web_md')
    # tokens = [token.text.lower() for token in nlp(doc_sentence)]

   # 为原文加上起始符号<sos>和结束符号<eos>
   #  tokens = [doc_field.init_token] + tokens + [doc_field.eos_token]

    # 将字符转换成序号
    with open(data_config['vocab_path'], "r", encoding="utf-8") as f:
        vocab = json.load(f)
    doc_indexes = [vocab[token] for token in tokens]

    # 转换成可以gpu计算的tensor
    doc_tensor = torch.LongTensor(doc_indexes).unsqueeze(1).to(device)

    doc_len = torch.LongTensor([len(doc_indexes)]).to(device)

    # 计算encoder
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(doc_tensor, doc_len)

    mask = model.create_mask(doc_tensor)
    # 生成摘要的一个单词 <sos>
    sum_indexes = [sum_field.vocab.stoi[sum_field.init_token]]

    # 构建一个attention tensor，存储每一步的attention
    attentions = torch.zeros(max_len, 1, len(doc_indexes)).to(device)

    for i in range(max_len):

        sum_tensor = torch.LongTensor([sum_indexes[-1]]).to(device)

        # 计算每一步的decoder
        with torch.no_grad():
            output, hidden, attention = model.decoder(sum_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention

        pred_token = output.argmax(1).item()

        # 如果出现了 <eos> 则直接结束计算
        if pred_token == sum_field.vocab.stoi[sum_field.eos_token]:
            break

        sum_indexes.append(pred_token)
    # 把序号转换成单词
    sum_tokens = [sum_field.vocab.itos[i] for i in sum_indexes]

    return sum_tokens[1:], attentions[:len(sum_tokens) - 1]