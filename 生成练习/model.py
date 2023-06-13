# encoding:utf-8
import jieba.analyse
from textrank4zh import TextRank4Keyword, TextRank4Sentence

from dedal_data import DealData
from rouge  import Rouge


class TrainModel(object):
    def __init__(self):
        self.df=DealData().read_data()


    def model_evaluation(self,y_pred,y_true):

        rouge = Rouge()
        rouge_scores = rouge.get_scores(" ".join(jieba.cut(y_pred))," ".join(jieba.cut(y_true)))#"Installing collected packages", "Installing "
        print('rouge_scores:', rouge_scores)
        rouge_f=[rouge_scores[0][k]['f'] for k in rouge_scores[0]]
        score=0.2*rouge_f[0]+0.3*rouge_f[1]+0.5*rouge_f[2]

        # rl_p = rouge_scores[0]['rouge-l']['p']
        print("score", score)

    def text_rank(self):
        # 语料为一篇文档；文档用\n分割句子
        line=self.df.head(1)


        text=line['text'].values[0]

        # tags = jieba.analyse.extract_tags(text)
        # print("tfidf-关键词:", tags)
        # 2 关键词提取：textrank(jieba)
        # tags=jieba.analyse.textrank(text)
        # print('textrank-关键词:',tags)


        # 3 关键词提取：textrank(TextRank4Keyword)
        # 创建分词类的实例
        tr4w = TextRank4Keyword()
        # 对文本进行分析，设定窗口大小为2，并将英文单词小写
        tr4w.analyze(text=text, lower=True, window=2)
        # 从关键词列表中获取前20个关键词
        item=[item.word for item in tr4w.get_keywords(num=10, word_min_len=1)]
        # print(item.word, item.weight)打印每个关键词的内容及关键词的权重
        print('TextRank4Keyword-textrank-关键词:',item)


        # 4 关键词组提取：textrank(TextRank4Keyword)
        # phrase=[phrase for phrase in tr4w.get_keyphrases(keywords_num=5, min_occur_num=2)]
        # print('TextRank4Keyword-textrank-关键词组:',phrase)


        # 5 摘要提取：textrank(TextRank4Keyword)
        tr4s = TextRank4Sentence()
        # 英文单词小写，进行词性过滤并剔除停用词
        tr4s.analyze(text=text , lower=True, source='all_filters')
        print('TextRank4Keyword-textrank-摘要提取:')

        pred_tup=tr4s.get_key_sentences(num=3)
        y_pred=','.join([i['sentence'] for i in sorted(pred_tup,key=lambda x:x['index'] )])
        print(y_pred)

        # print(res)
        # for item in tr4s.get_key_sentences(num=3):
            # 打印句子的索引、权重和内容
            # print(item.index, item.weight, item.sentence)

            # y_pred=y_pred+''+item.sentence
        # print(y_pred)
        print(line['summary'].values[0])
        self.model_evaluation(y_pred,line['summary'].values[0])
        # return ','.join(item.sentence)


TrainModel().text_rank()