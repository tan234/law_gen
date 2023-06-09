import jieba
from config import *
import json
import torch

from transformer_model import *
from jieba import posseg as psg
from paddlenlp.transformers import AutoModelForConditionalGeneration
from paddlenlp.transformers import AutoTokenizer
from config import *
from paddle import fluid
from paddlenlp.transformers import GPTModel, GPTTokenizer
from gpt2_model import Gpt2Model
import paddle
from config import *
from evaluating import top_k_top_p_filtering
'''Transformer 推理'''

class TransformerPred(object):

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.stopwords = self.stopw()

        # 加载模型
        self.model=self.load_model()
        # 需要小于dec_len
        self.max_tgt_len=data_config['dec_len']
        # idx2word_dict
        with open(data_config['idx2word_dict'], "r", encoding="utf-8") as f2:
            self.idx2word = json.load(f2)

        with open(data_config['vocab_path'], "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.config=Transformer_config
    '''读取训练好的模型'''
    def load_model(self):
        cur_dir = os.path.dirname(__file__)
        model_path = os.path.join(cur_dir, self.config['model_save_dir'])
        model = Transformer()
        m_state_dict = torch.load(model_path)
        model = model.to(self.device)
        model.load_state_dict(m_state_dict)
        return model
    '''停用词典'''
    def stopw(self):
        stopword = [line.replace('\n', '') for line in
                    open(data_config['stopword_path'], 'r+', encoding='utf-8').readlines()]
        return dict(zip(stopword, range(0, len(stopword))))

    def dec_input_pred(self,model, enc_input, start_symbol):
        '''
        预测时：
        encoder 和decoder,FFn都是拆开的
        decoder_input=[1,max_len]，是一个可变的list,每一步输出的decoeder,都改变了这个list
        '''

        enc_outputs, enc_self_attns = model.encoder(enc_input)  # enc_outputs:1,seq,emb

        dec_input = torch.zeros(1, self.max_tgt_len).type_as(enc_input.data)  # 1,max_tag_len
        # dec_input = torch.zeros(1, 0).type_as(enc_input.data)#空
        # S,
        next_symbol = start_symbol

        for i in range(0, self.max_tgt_len):
            dec_input[0][i] = next_symbol
            # dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype)], -1)
            # detach()不计算梯度,cat拼接类似于append，只是二维的 dec_input:[[a]]->[[a,b]]

            dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)

            projected = model.projection(dec_outputs)# projected： [batch_size, tgt_len, tgt_vocab_size]

            # y_hat = projected.squeeze(0).argmax(1)  # [tgt_len, tgt_vocab_size]

            # ----去掉 S-----
            projected = projected.squeeze(0)
            a = projected.size()[0]
            b = projected.size()[1]
            new_projected = torch.zeros(a, b - 1)
            new_projected[:, :next_symbol] = projected[:, :next_symbol]
            new_projected[:, next_symbol:] = projected[:, next_symbol+1:]
            y_hat = new_projected.argmax(dim=1)  # tgt_len
            next_symbol = y_hat[i]


            # y_hat = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            # next_symbol = y_hat[i]

            if self.idx2word[str(y_hat[i].item())] == 'SOE':
                    break

        return dec_input


    def start(self,doc):

        # 1 对原文分词
        # tokens=[word for word, p in psg.cut(doc) if word not in self.stopwords and len(word) > 1 and p not in ['m']]
        # tokens=[word for word in jieba.cut(doc)]
        tokens=[word for word in list(doc)]


        print(tokens)

        # 2 将字符转换成序号


        doc_indexes = [self.vocab.get(token, self.vocab["UNK"]) for token in tokens]
        doc_indexes = doc_indexes[:min(data_config['enc_len'],len(doc_indexes))] # 可以少，最多不能超过enc_len
        # print(doc_indexes)
        # print(len(doc_indexes))

        # 转换成可以gpu计算的tensor
        enc_inputs = torch.LongTensor(doc_indexes).unsqueeze(0).to(self.device)  # [seq_len,1]
        # tgt_len = 16

        # vocab里面要加上S,E
        predict_dec_input = self.dec_input_pred(self.model, enc_inputs, start_symbol=self.vocab["SOS"])# max_len
        print(predict_dec_input)
        # predict, _, _, _ = self.model(enc_inputs, predict_dec_input)

        # 问题：dec_input不就是输出了么，为什么还要再进行一次transformer
        # predict, _, _, _ = self.model(enc_inputs, predict_dec_input)  # [batch_size, tgt_len, tgt_vocab_size]
        # predict = predict.squeeze(0).argmax(dim=1)  # [tgt_len]
        res=[self.idx2word[str(i)]for i in predict_dec_input[0].tolist()]
        # res=[predict_dec_input]
        print(''.join(res))

        return res



'''GPT2 推理'''
class Gpt2Pred(object):
    '''
    模型预测通过top-k采样，每一步生成一个词不再是概率最大的一个（贪心搜索），从概率最大的k个中随机采样一个，这样子生成的效果不会太死板。
    '''
    def __init__(self):

        # 加载模型
        self.tokenizer = GPTTokenizer.from_pretrained("gpt2-medium-en")
        self.tokenizer.add_special_tokens({"sep_token": "<sep>"})


        self.config=GPT2_config
        self.model = Gpt2Model(self.tokenizer.vocab_size + 1)
        model_dic = paddle.load(self.config['model_save_path'])
        self.model.set_state_dict(model_dic)

        # self.infer(tokenizer,model,config,content)

    def infer(self,content):

        while True:
            # 输入内容编码
            input_id = [self.tokenizer.sep_token_id]

            input_id.extend(self.tokenizer(content)["input_ids"])
            input_id.append(self.tokenizer.sep_token_id)
            input_id = paddle.to_tensor([input_id])

            # logits = net(paddle.to_tensor([input_id]))
            response = []
            for _ in range(self.config['max_len']):
                logits = self.model(input_id)

                next_token_logits = logits[0, -1, :]# 取最后一个预测结果
                # 减小已经生成单词的概率
                for id in set(response):
                    next_token_logits[id] /= self.config['repetition_penalty']#给生成摘要中已经生成的词，降低这个词在本次预测结果中的概率

                next_token_logits = next_token_logits / self.config['temperature']#所有的概率都降低？
                next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')

                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=5, top_p=0)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(filtered_logits.softmax(dim=-1), num_samples=1)

                if next_token == self.tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    break
                response.append(next_token.item())
                input_id = paddle.concat((input_id, next_token.unsqueeze(0)), axis=1)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            # history.append(response)
            self.tokenizer.add_special_tokens(response)# tokenizer.add_special_tokens可以让tokenizer不给’[C1]’,’[C2]’,’[C3]’,’[C4]'进行分词
            text = self.tokenizer.convert_ids_to_string(response)
            print("模型预测摘要:" + text)


'''Pegasus 推理'''
class PegasusPred(object):

    def __init__(self,content,model_name):
        self.config = Pegasus_config

        # 加载训练好的模型
        self.model = AutoModelForConditionalGeneration.from_pretrained(model_name+'_'+self.config['model_save_dir'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name+'_'+self.config['model_save_dir'])
        self.content=content


    # -------------模型推理------------
    # 模型推理，针对单条文本，生成摘要
    def infer(self):
        tokenized = self.tokenizer(self.content,
                              truncation=True,
                              max_length=self.config['max_source_length'],
                              return_tensors='pd')
        preds, _ = self.model.generate(input_ids=tokenized['input_ids'],
                                  max_length=self.config['max_target_length'],
                                  min_length=self.config['min_target_length'],
                                  decode_strategy='beam_search',
                                  num_beams=4)
        res=self.tokenizer.decode(preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(res)
        return res

# if __name__=="__main__":
#
#     # 测试数据
#     doc_sentence= '''款共计4000余万元。今天上午，"
#                  "涉嫌诈骗罪的田某在北京二中院受审，在法庭上，田某表示认罪，并承认自己确实用被害人的钱买了宝马车、"
#                  "商品房、手表和20多个名牌包。摄/通讯员王鑫刚33岁的田某是北京市西城区人。公诉机关指控，2016年至2017年11月间，"
#                  '''
#     doc_sentence="""网易首页应用快速导航登录注册免费邮箱移动端网易公开课网易考拉网易严选支付电商邮箱网易首页>新闻中心>"
#              "新闻>正文交友48岁男子引诱强奸8名初中女学生一审获刑十四年半2019-03-2919:35:33来源:北京头条举报338易信微信QQ空间微博更多"
#              "（原标题：男子强奸8名初中女学生一审获刑十四年半）男子查某（化名）隐瞒真实姓名、年龄，谎称自己开公司、酒庄，通过qq、微信等方式，"
#              "认识多名初中女学生。查某利用初中生年幼无知、经济拮据的特点，通过唱歌、借钱、逼债等方式，引诱、逼迫多名未成年女性或幼女，与其在宾馆内发生性关系，"
#              "案发时6人尚不满14周岁，最小的受害者案发时仅12岁。3月29日，北京青年报记者从中国裁判文书网获悉，查某因犯强奸罪被法院一审判处有期徒刑14年6个月。"
#              "查某今年48岁。法院经审理查明，2017年下半年至2018年上半年期间，查某隐瞒真实姓名、年龄，称自己开公司、酒庄，通过QQ群、QQ空间、微信等多种方式，"
#              "认识了某县内多名初中女学生。在聊天过程中，查某利用初中生年幼无知、经济拮据，以男女朋友或兄妹相称，通过频繁发红包、转账、买东西、吃饭、唱歌、借钱"
#              "、逼债等方式，引诱、逼迫多名未成年女性或幼女，与其在宾馆内发生性关系。判决书显示，查某共强奸了8名未成年女性或幼女，其中因两名受害者拒绝犯罪中止。"
#              "他利用年幼无知、经济拮据的特点，通过发红包、借钱、逼债等方式，引诱、逼迫受害人与他在宾馆内发生性关系。而在与受害人发生性关系后，查某往往会给受害人一些钱，"
#              "对她们进行安抚，数额从1000多到1万多不等。庭审中，查某认为其不明知被害人未满14周岁，不构成强奸。法院认为，查某在侦查阶段的供述证实其隐瞒真实姓名和年纪，"
#              "目的就是想交往年纪小的女孩，其知道涉案多数被害人均在读初中，有的读初一，其中有受害人告诉过他其14岁，读初中。查某作为一个具有社会经验、正常认知判断能力的行为人，" \
#              "并非不知道被害人可能是幼女，甚至在主观上系积极追求状态，在此情况下，仍旧试图或者确实与被害人发生性关系，且事后以不同形式给予被害人相应的钱物，因此，应当认定为其明知，" \
#              "构成强奸罪。此外，在一些犯罪事实中，虽然查某没有采取暴力手段逼迫，但是以偿还借款等条件胁迫被害人与其发生性关系，这仍然构成强奸罪。" \
#              "法院认为，查某以胁迫或者其他手段多次强奸妇女、奸淫不满14周岁的幼女，其中在强奸两名受害者时系犯罪中止，其行为已构成强奸罪，" \
#              "公诉机关指控的罪名成立。法院以被告人查某犯强奸罪，判处有期徒刑14年6个月。根据我国刑法规定，奸淫不满十四周岁的幼女的" \
#              "，以强奸论，从重处罚。强奸妇女、奸淫幼女，强奸妇女、奸淫幼女情节恶劣，强奸妇女、奸淫幼女多人的，处十年以上有期徒刑、" \
#              "无期徒刑或者死刑。赵亚萍本文来源：北京头条责任编辑：杨强_NN6027跟贴34参与338帐号密码注册|跟贴广场|手机发跟贴登录并发贴网友评论仅供其表达个人看法，" \
#              "并不表明网易立场。为您推荐推荐娱乐体育财经时尚科技军事汽车＂长期性侵女教师＂的院长被免职曾明示学生陪自己新闻性侵华北电力大学|" \
#              "中国新闻周刊1天前4203跟贴4203浙江一区人武部部长被指在酒吧与人发生不正当关系新闻人武部纪委|新京报23小时前1507跟贴1507院长" \
#              "被曝性侵女教师还挑逗性骚扰女学生华电回应新闻纪委性侵|北京头条1天前36962跟贴3696216岁少女被囚禁性侵24天：4㎡地洞内贴婚庆饰品" \
#              "新闻婚庆性侵|澎湃新闻2天前1739跟贴1739媒体三问＂曹园＂事件:内部让人震惊谁是幕后保护伞新闻曹园违法|法制日报2天前1624跟" \
#              "贴162416岁少女被50岁单身男囚禁性侵24天案发现场曝光新闻性侵犯罪|重庆晨报上游新闻2天前14跟贴14优秀军人被控强奸服刑7年出" \
#              "狱喊冤:就要无罪两字新闻强奸服刑|新京报2天前61跟贴618000亿婚庆暴利背后，死要面子的中国式爱情槽值婚庆婚礼|网易槽值1天前" \
#              "4759跟贴4759歌手大壮成老赖被央视点名网友:大壮我们不一样新闻我们不一样大壮|重庆晨报上游新闻1天前1424跟贴142449岁大妈" \
#              "冒充90后＂军花＂诈骗90后小伙一家3万元新闻90后行骗|北京头条2天前373跟贴373＂流浪大师＂剃须理发换装后判若两人网友:要" \
#              "出道?新闻剃须理发|北京时间1天前1581跟贴1581响水爆炸头七:他们仍然在等待那些没有消息的人新闻爆炸响水|北京时间2天前3265跟贴" \
#              "3265女员工侵占银行1亿余元购置10多套房3年后才案发新闻外资银行金库|看看新闻1天前4066跟贴4066囚禁少女性侵24天嫌犯:记10余个qq标注＂" \
#              "我的女人＂新闻犯罪性侵|重庆晨报上游新闻2天前6404跟贴6404大学生搭顺风车出事身亡平台泄露乘客信息引乱象新闻顺风车拼车|南方都市报1天前1跟贴1抖音" \
#              "＂流浪大师＂火了，一条视频竟被卖上千元？新闻网红杜甫|曲一刀2天前3099跟贴3099牛弹琴：普京这次来真的了突然出兵委内瑞拉新闻普京委内瑞拉|牛弹琴1天" \
#              "前10770跟贴10770沈阳交警大队发生纵火爆炸袭警致3人受伤嫌犯死亡新闻纵火爆炸|沈阳市公安局1天前33110跟贴33110坚决说“不”！美盟友发声明批“戈兰高地" \
#              "公告”军事戈兰安理会|环球时报-环球网1天前737跟贴737＂团贷网＂被指非法集资145亿负责人被采取强制措施新闻团贷网网贷平台|上游新闻23小时前157跟贴157漫" \
#              "威官宣＂复联4＂内地定档4.24内地观众将全球首看新闻复联4复仇者联盟4|漫威影业官方微博13小时前3327跟贴3327公安部：彻底肃清周永康、孟宏伟的遗毒遗害新闻周永康" \
#              "孟宏伟|新京报1天前13跟贴13打开各地社保局官网网友被这些＂神仙项目＂惊呆了新闻就业社保局|观察者网20小时前632跟贴632比李云龙的意大利炮还暴力！美国飞机装105榴弹炮" \
#              "军事榴弹炮运输机|网易谈兵2天前941跟贴941韩国瑜返台千人接机台民众高举＂农民的救星＂标语新闻韩国瑜民进党|观察者网1天前296跟贴296德媒:为不让中国抢先载人登月美把" \
#              "安全放第二位军事载人登月中国|环球时报-环球网1天前5跟贴5加警方:加拿大遭绑架22岁中国留学生被发现还活着新闻加拿大中国|环球时报-环球网2天前764跟贴764陈国强被免去陕" \
#              "西副省长职务辞去省人大代表职务新闻陈国强陕西省|华商网5小时前277跟贴277美军列装全能防水袋可清洁武器防黑客军事防水袋武器|参考消息网1天前24跟贴24美国女生将闺蜜推下18" \
#              "米高桥摔成重伤被判入狱2天新闻入狱重伤|中国日报10小时前1013跟贴1013莫迪高调宣布“打卫星”成功巴基斯坦这样回应军事莫迪巴基斯" \
#              "坦|环球时报-环球网1天前3跟贴32个多月没消息的陈国强被免曾为赵正永＂大管家＂新闻赵正永陈国强|中国新闻周刊3小时前1跟贴1内蒙" \
#              "古呼和浩特市一居民楼发生爆炸有人员伤亡新闻居民楼海东路|腾格里新闻4小时前706跟贴706执导影片票房惨淡王小帅宣传文案现" \
#              "＂初夜＂露骨字眼新闻王小帅泡妞|观察者网23小时前46跟贴46美又要韩交付大笔“保护费”开口就是要数百亿军事美军韩美|环球时报-" \
#              "环球网1天前255跟贴25511岁男孩将70颗磁力珠塞尿道医生呼吁家长别再买新闻磁力珠手术|澎湃新闻2天前6170跟贴6170美放言5年内“" \
#              "重返月球”美媒：向中国发出挑战军事月球中国|环球时报-环球网1天前479跟贴479+加载更多新闻×大家都爱看进入新闻频道真正聪明人,饭局" \
#              "都用这8个套路说话课程|有多少人升职加薪，卡在了用不好excel人间|我以为的安稳，正在吞噬我财经|世行原副行长:人类平均" \
#              "寿命每2小时延长25分|大师科技|京东回应995工作制：不会强制要求但要全情投入体育|为了弗格森一句话，他狂奔80米换了" \
#              "1张红牌！23年的绝娱乐|漫威宣布《复联4》中国首映礼安排：4月18日举办时尚|向佐求婚郭碧婷她的清新初恋感谁看了不想" \
#              "娶！新闻推荐进入新闻频道浙江一区人武部部长被指在酒吧与人发生不正当关系科技|谁在使用拼多多？手机|华为P30Pro开箱" \
#              "图赏：徕卡四摄吊打友商旅游|全球知名红灯区花街柳巷不眠夜热点新闻进入新闻首页陶崇园坠亡一年导师终于道歉了马端斌举报村支" \
#              "书:县委否认调查组被操纵西安地铁电缆案责任人被判无期韩国瑜硕士论文:大陆解放台目标恒久不变热点新闻新加坡ins女王上日本综" \
#              "艺豪宅整面墙都是爱马仕包\"烂尾楼\"盖了136年终于要完工却因违章被罚3亿元\"港版王思聪\"家的音响比跑车还贵竟" \
#              "被嘲不豪华院长被曝性侵女教师还挑逗性骚扰女学生华电回应南京一已逝儿子托梦母亲说在水里难受，家人抽干池塘竟然.." \
#              ".南京老汉放生千年乌龟后，儿女遭遇车祸，真相令人崩溃!美女总裁惨遭调戏，特种兵未婚夫一个电话叫来200人...态度原创了不起的中" \
#              "国制造|牛!斯里兰卡钞票上印着中国造大国小民|门口的棒棒，消失了谈心社|流浪大师被扒光围观，到底是谁疯了王三三|这部剧比大尺度更有趣" \
#              "的是想象力网易公开课查看全部你的时间为什么总不够用你的时间为什么总不够用12个即兴表达的套路12个即兴表达的套路高质量人脉13招吸引过来高质量人脉1" \
#              "3招吸引过来销售新人最爱犯这些错误销售新人最爱犯这些错误精彩推荐海淘品牌春天怎么拍才上镜？春天怎么拍才上镜？记忆大师曝过目不忘法记忆大师曝过目不忘法" \
#              "5步不再越努力越焦虑5步不再越努力越焦虑百万人在用英语学习法百万人在用英语学习法跟贴排行点击排行132022陶崇园坠亡一年让学生叫\"爸爸\"的导师终于道歉" \
#              "了228047西安地铁电缆案责任人被判无期奥凯电缆被罚3千万32645222岁女子因父母潜逃被羁押388天17年后获赔16万423366四川3名未成年人因3元钱萌生抢劫意" \
#              "图捂死女店主58930死刑！男子假冒军官与女友交往分手后杀人报复68569求求你们，放过武大的樱花吧75909埃航空难调查结果:坠毁前自动预防失速系统已启动84706" \
#              "人间|90后进国企：我以为的安稳，正在吞噬我云阅读进入频道|原创|读书“繁花盛开”的相逢之爱“繁花盛开”的相逢之爱电影《南极之恋》原著电影《南极之恋》原著独家" \
#              "首发丨朱沐(Lydia)全新都市爱情传奇焦虑时代，如何像聪明人一样思考问题？哈利波特20周年一口气重温童年经典东野圭吾最负盛名之作一个跨越19年的迷案你要成为那个不" \
#              "可替代的人才有赢的机会时代在不断变化而“老炮儿”永远不死阅读下一篇2个多月没消息的陈国强被免曾为赵正永＂大管家＂陕西官场再起涟漪。两个多月没有消息的陈国强今天终于" \
#              "有了消息，他被免去陕西副省长职务，同时被终止省人大代表资格。这距他上任不过一年光景，而他尚不满57岁。据陕西传媒返回网易首页返回新闻首页©1997-2019网易公司版权所有About" \
#              "NetEase|公司简介|联系方法|招聘信息|客户服务|隐私政策|广告服务|网站地图|不良信息举报"""
#
#
#
#     Pred().start(doc_sentence)
# tensor([[70275,  2952, 70274,  1860,  1000,  4450,   514,  1967,  1290,  2435,
#            588,  1551,  7719,  2495,  1248,    32,    49,  1405,   107,   264,
#           2667,   634,  2637,  2021,  2323,  5930,   136,   443,   228,    37,
#            825,  4294,  4707,  3199,   484,   547,     8,   446,    47,    28,
#            128,   790,   112,   159,  1494,  2456,  1667,     0]])