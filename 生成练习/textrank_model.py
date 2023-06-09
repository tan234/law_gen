#encoding:utf-8

from textrank4zh import TextRank4Keyword, TextRank4Sentence
import pandas as pd

class TextRank(object):
    '''
    textrank 提取关键句子，作为文本摘要
    无监督？不需要保存模型，一条文本一预测？
    '''
    def __init__(self,df):
        self.df=df
        self.tr4s = TextRank4Sentence()
        self.df['pred_line'] = self.df['text'].map(lambda x: self.model(x))

    def model(self,x):

        self.tr4s.analyze(text=x, lower=True, source='all_filters')

        pred_tup = self.tr4s.get_key_sentences(num=3)

        y_pred = ','.join([i['sentence'] for i in sorted(pred_tup, key=lambda x: x['index'])])

        return y_pred
    # def model(self):
    #
    #
    #     self.df['pred_line']=self.df.apply(lambda row:self.abstrct_text(row))
    #
    #     for index,row in self.df.iterrows():
    #
    #         # 英文单词小写，进行词性过滤并剔除停用词
    #         self.tr4s.analyze(text=row['text'], lower=True, source='all_filters')
    #         # print('TextRank4Keyword-textrank-摘要提取:')
    #
    #         pred_tup = self.tr4s.get_key_sentences(num=3)
    #         y_pred = ','.join([i['sentence'] for i in sorted(pred_tup, key=lambda x: x['index'])])
    #         # print(y_pred)
    #         self.df.loc[index,'pred_line']=y_pred
    #         # print(res)
    #         # for item in tr4s.get_key_sentences(num=3):
    #         # 打印句子的索引、权重和内容
    #         #     print(item.index, item.weight, item.sentence)
    #
    #     return self.df
# df=pd.DataFrame([{'text':"不满祖先肖像被占用，王老吉后人将广州医药集团有限公司诉至法院。7月23日，新京报记者从北京知识产权法院获悉，王老吉后人因未及时按照法律规定申请续展注册“王老吉真像”，无法主张该的权。法院一审驳回原告的诉讼请求，目前该判决已生效。图为广药集团注册的王老吉。法院供图图为王老吉曾孙女注册的。法院供图判决显示，王老吉创始人名为王泽邦，乳名阿吉，年老后，大家唤他“王老吉”。1828年清道光年间，王泽邦开设了“王老吉凉茶铺”。王老吉去世后，其三子为了纪念父亲以及经营便利，了“王老吉真像”进行推广。1951年8月1日，王泽邦曾孙女王某申请注册了第7686号“王老吉真像”图形（简称引证）。1958年1月9日，该所有人由王某变更为王老吉联合制药厂，专用权至1971年7月31日。后专用权期满，王老吉联合制药厂未申请续展注册该。据判决书，2012年5月2日，广州医药集团有限公司（简称广药集团）向原国家工商行政管理总局局申请注册了第10855371号“王老吉真像”图形（简称诉争）。2017年7月12日，王某之子女胡某焕、胡某美、胡某业三人向原国家工商行政管理总局评审委员会（简称原商评委）提出对诉争的无效宣告请求。原商评委裁定诉争应予维持。胡某焕、胡某美、胡某业不服上述裁定，向北京知识产权法院提起行政诉讼，主张广药集团申请注册的诉争与其母王某申请注册的引证均为“王老吉真像”，二者完全一致。诉争对引证的摹仿、复制构成《法》“同一种商品或者类似商品上已经注册的相同或者近似”的规定，诉争应当被宣告无效。此外，诉争所使用的肖像王泽邦（王老吉）系三原告的外高祖父，广药集团以营利为目的使用其外高祖父的肖像，侵犯了三原告享有的已故外高祖父肖像权，从而构成《法》“损害他人现有的在先权利”之情形。2019年4月29日，北京知识产权法院经审理作出一审判决，驳回三原告胡某焕、胡某美、胡某业的诉讼请求。宣判后，三方当事人均未提出上诉。现判决已生效。"},
#                          {'text':'新浪上市公司研究院法说资本/恢恢近日，备受各界关注的中小投资者诉成都华泽钴镍材料股份有限公司、国信证券、瑞华会计师事务所（特殊普通合伙）证券虚假陈述责任纠纷案件又有新动向。（ST华泽维权入口）新浪从代理了中小投资者起诉索赔的广东奔犇主任刘国华律师处了解到，该案于2019年8月1日在成都市中级人民法院开庭审理，有多名律师到庭，这已是该系列案第二批开庭。除了中午休息一个多小时，庭审从早上9：15一直持续到下午5：30左右结束，各方围绕虚假陈述实施日、揭露日、国信、瑞华是否应承担连带责任、是否存在系统风险等焦点问题展开了激烈辩论，法院并未当庭宣判。对于虚假陈述揭露日，各方看法不一，分歧巨大，具体时间结点尚有待法院认定。2018年1月，证监会因虚假陈述依法对华泽钴镍作出行政处罚。同年8月，华泽涉嫌证券犯罪的相关人员被移送公安机关依法追究刑事责任。同年6月，国信证券作为华泽钴镍2013、2014年重大资产重组财务顾问和恢复上市的保荐机构，未能勤勉尽责，证监会对其进行了行政处罚。而担任审计机构的瑞华会计师事务所因华泽2013年和2014年年报存在虚假记载、重大遗漏于2018年12月同样被证监会采取行政处罚。因2015年、2016年、2017年连续三个会计年度经审计的净利润为负值，2016年、2017年连续两个会计年度的财务会计报告被出具无法表示意见的审计报告，2018年7月13日，ST华泽暂停上市。2019年5月17日，深交所决定对*ST华泽终止上市。7月9日，ST华泽被深交所摘牌。刘国华律师表示，华泽虽已退市，但并不影响投资者的索赔权利，其他受损投资者仍可索赔。关于国信应否担责，国信认为自己无过错，不应赔偿。原告方认为，其制作、出具的文件有虚假记载、误导性陈述或者重大遗漏，给他人造成损失的，应当与发行人、上市公司承担连带赔偿责任，但是能够证明自己没有过错的除外。由于证监会处罚决定书已下发，目前来看国信过错明显，应该承担相应的连带责任。瑞华则声称自己在2019年6月向北京市第一中级人民法院提起行政诉讼，要求撤销证监会的行政处罚决定书，故依据《民事诉讼法》第一百五十条第五款申请中止审理。原告方对此并不认可。原告方表示，本案的前置条件是证监会的行政处罚决定书，而不是法院的判决。申请中止审理请求未获法院支持，瑞华应该承担相应的连带责任。由于被告不同意调解，且存在多处分歧，各方将等待法院宣判。刘国华律师表示，虽判决结果还未下达，但是根据庭审情况，索赔条件调整为：在2013年9月20日到2015年11月24日之间买入华泽钴镍，并在2015年11月24日后卖出或继续持有的受损投资者。索赔条件最终由法院生效判决确定。在此时间段交易的投资者仍可进行索赔。对于争议较大的瑞华所是否应当承担连带责任。刘国华律师认为，现有的特殊普通合伙制，某些有错的会计师无力承担赔偿责任，某些“无错”的合伙人无须担责，收费动辄以百万千万计，风险和收益显然不成比例，某些会计师事务所和会计师也有忽视风险、大规模扩张的动力。因此，刘国华律师主张应提高中介机构的准入门槛，至少对于愿意承担无限连带责任的会计师事务所给予适当的政策倾斜。应让违法违规者承担相应的刑事、行政和民事责任。特别对于损失惨重的投资者来说，通过证券民事赔偿诉讼，让违规的会计师事务所和合伙人赔偿受损投资者的损失，才是对他们最好的保护。'}])
# df=TextRank(df).model()
# print(df)