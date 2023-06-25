import jieba
from config import *
import json
import torch

from transformer_model import *

class Pred(object):

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'



        # 加载模型
        cur_dir = os.path.dirname(__file__)
        model_path = os.path.join(cur_dir, transformer_config['transformer_model_path'])
        self.model = Transformer()
        m_state_dict = torch.load(model_path)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(m_state_dict)
        self.tgt_len=data_config['dec_len']#需要定义这个吗
    def dec_input_pred(self,model, enc_input, start_symbol):
        '''
        预测时：
        encoder 和decoder,FFn都是拆开的
        decoder_input=[1,max_len]，是一个可变的list,每一步输出的decoeder,都改变了这个list

        :param model:
        :param enc_input:
        :param start_symbol:
        :return:
        '''

        enc_outputs, enc_self_attns = model.encoder(enc_input)  # enc_outputs:1,seq,emb

        dec_input = torch.zeros(1, self.tgt_len).type_as(enc_input.data)  # 1,max_tag_len
        # S,
        next_symbol = start_symbol
        for i in range(0, self.tgt_len):
            dec_input[0][i] = next_symbol

            dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)

            projected = model.projection(dec_outputs)
            # projected： [batch_size, tgt_len, tgt_vocab_size]
            y_hat = projected.view(-1, projected.size()[-1])  # [tgt_len, tgt_vocab_size]
            y_hat = y_hat.argmax(dim=1)  # tgt_len

            next_symbol = y_hat[i]
        return dec_input


    def start(self,doc):

        # 1 对原文分词
        tokens = list(jieba.cut(doc))

        # 2 将字符转换成序号
        with open(data_config['vocab_path'], "r", encoding="utf-8") as f:
            vocab = json.load(f)

        doc_indexes = [vocab.get(token, vocab["UNK"]) for token in tokens]
        doc_indexes = doc_indexes[:60]#有没有限制

        # 转换成可以gpu计算的tensor
        enc_inputs = torch.LongTensor(doc_indexes).unsqueeze(0).to(self.device)  # [seq_len,1]
        # tgt_len = 16

        # vocab里面要加上S,E
        predict_dec_input = self.dec_input_pred(self.model, enc_inputs, start_symbol=vocab["S"])

        # 问题：dec_input不就是输出了么，为什么还要再进行一次transformer
        predict, _, _, _ = self.model(enc_inputs, predict_dec_input)  # [batch_size, tgt_len, tgt_vocab_size]

        predict = predict.squeeze(0).argmax(dim=1)  # [tgt_len]

        with open(data_config['idx2word_dict'], "r", encoding="utf-8") as f2:
            idx2word = json.load(f2)
        res=[idx2word[str(i.item())] for i in predict]
        print(res)

        return res

if __name__=="__main__":

    # 测试数据
    doc_sentence= '''原标题：冒充监狱管理局民警以低价办卡为幌诈骗9人4000多万）法制晚报·看法新闻买来警服，私刻公章，冒充北京市监狱管理局的民警，"
                 "女子田某谎称可以从单位低价购买到中石化、中移动等储值卡卖，诈骗王某等9名被害人货款共计4000余万元。今天上午，"
                 "涉嫌诈骗罪的田某在北京二中院受审，在法庭上，田某表示认罪，并承认自己确实用被害人的钱买了宝马车、"
                 "商品房、手表和20多个名牌包。摄/通讯员王鑫刚33岁的田某是北京市西城区人。公诉机关指控，2016年至2017年11月间，"
                 "田某冒充北京市监狱管理局工作人员，私刻该单位公章，谎称能够通过内部政策低价从该单位购买中石化、"
                 "中国移动等储值卡然后卖给被害人。之后，田某采用高价进货低价卖出的方式，"
                 "诈骗被害人王某、冯某、潘某、罗某等九人货款共计4000余万元。田某将赃款用于个人消费。案发前被告人归还被害人少部分钱款。"
                 "2017年11月29日，被害人王某、冯某等人将田某扭送到公安机关并报案。公诉机关认为应当以诈骗罪追究田某的刑事责任。"
                 "“我没骗那么多钱，我自己算只有3400余万元。”庭审中，田某表示认罪，但对公诉机关指控的诈骗金额提出了异议。田某供述称，"
                 "她和被害人都是朋友，“我和王某是一起旅游时认识的，回来后经常来往就成了朋友，至于潘某，我俩住在一个小区里。”田某表示，"
                 "她得知有卖中石化、中石油的加油卡，自己认为可以赚钱，于是就从网上找到了上家张林（化名），从张林处购买卡后，转手卖给下家王某等人。"
                 "为了让被害人充分信任自己，田某告诉王某等下家，她在北京市监狱管理局工作，至于为何要冒充监狱工作人员'''

    Pred().start(doc_sentence)



def generate_summary(doc_sentence,doc_field,sum_field,model,device,max_len):
    '''
    :param doc_sentence: 输入内容
    :param doc_field:
    :param sum_field:
    :param model:
    :param device:
    :param max_len:
    :return:
    '''

    doc_sentence= '''原标题：冒充监狱管理局民警以低价办卡为幌诈骗9人4000多万）法制晚报·看法新闻买来警服，私刻公章，冒充北京市监狱管理局的民警，"
             "女子田某谎称可以从单位低价购买到中石化、中移动等储值卡卖，诈骗王某等9名被害人货款共计4000余万元。今天上午，"
             "涉嫌诈骗罪的田某在北京二中院受审，在法庭上，田某表示认罪，并承认自己确实用被害人的钱买了宝马车、"
             "商品房、手表和20多个名牌包。摄/通讯员王鑫刚33岁的田某是北京市西城区人。公诉机关指控，2016年至2017年11月间，"
             "田某冒充北京市监狱管理局工作人员，私刻该单位公章，谎称能够通过内部政策低价从该单位购买中石化、"
             "中国移动等储值卡然后卖给被害人。之后，田某采用高价进货低价卖出的方式，"
             "诈骗被害人王某、冯某、潘某、罗某等九人货款共计4000余万元。田某将赃款用于个人消费。案发前被告人归还被害人少部分钱款。"
             "2017年11月29日，被害人王某、冯某等人将田某扭送到公安机关并报案。公诉机关认为应当以诈骗罪追究田某的刑事责任。"
             "“我没骗那么多钱，我自己算只有3400余万元。”庭审中，田某表示认罪，但对公诉机关指控的诈骗金额提出了异议。田某供述称，"
             "她和被害人都是朋友，“我和王某是一起旅游时认识的，回来后经常来往就成了朋友，至于潘某，我俩住在一个小区里。”田某表示，"
             "她得知有卖中石化、中石油的加油卡，自己认为可以赚钱，于是就从网上找到了上家张林（化名），从张林处购买卡后，转手卖给下家王某等人。"
             "为了让被害人充分信任自己，田某告诉王某等下家，她在北京市监狱管理局工作，至于为何要冒充监狱工作人员'''

    doc_field=''
    sum_field=''

    # model_path = os.path.join(cur_dir, transformer_config['transformer_model_path'])
    # model = Transformer()
    # model.load_state_dict(torch.load(model_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    cur_dir = os.path.dirname(__file__)

    model_path = os.path.join(cur_dir, transformer_config['transformer_model_path'])
    model = Transformer()
    m_state_dict = torch.load(model_path)
    model = model.to(device)
    model.load_state_dict(m_state_dict)


    print(model)

    max_len = 50
    # 将模型置为验证模式
    # model.eval()

    # 对原文分词
    tokens=list(jieba.cut(doc_sentence))

   # 为原文加上起始符号<sos>和结束符号<eos>
   #  tokens = [doc_field.init_token] + tokens + [doc_field.eos_token]

    # 将字符转换成序号
    with open(data_config['vocab_path'], "r", encoding="utf-8") as f:
        vocab = json.load(f)

    doc_indexes = [vocab.get(token,vocab["UNK"]) for token in tokens]#[[1, 307, 1, 2985],,,,]
    # print(doc_indexes[:4])
    # kk
    # 转换成可以gpu计算的tensor
    doc_tensor = torch.LongTensor(doc_indexes).unsqueeze(1).to(device)#[seq_len,1]

    doc_len = torch.LongTensor([len(doc_indexes)]).to(device)#seq_len
    # print(doc_len,doc_len.size())
    # kk
    # 计算encoder
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(doc_tensor)

    # print(encoder_outputs)
    # print(hidden)
    # mask = model.create_mask(doc_tensor)
    # # 生成摘要的一个单词 <sos>
    # sum_indexes = [sum_field.vocab.stoi[sum_field.init_token]]
    #
    # # 构建一个attention tensor，存储每一步的attention
    attentions = torch.zeros(max_len, 1, len(doc_indexes)).to(device)
    #
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


# generate_summary()