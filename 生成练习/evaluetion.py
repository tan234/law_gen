

from rouge  import Rouge
import jieba

class EvaluationTestData(object):

    def __init__(self,y_pred,y_true):

        self.y_pred=y_pred
        self.y_true=y_true

    def model_evaluation(self):
        rouge = Rouge()

        rouge_scores = rouge.get_scores(" ".join(jieba.cut(self.y_pred))," ".join(jieba.cut(self.y_true)))#"Installing collected packages", "Installing "
        # print('rouge_scores:', rouge_scores)
        rouge_f=[rouge_scores[0][k]['f'] for k in rouge_scores[0]]
        score=0.2*rouge_f[0]+0.3*rouge_f[1]+0.5*rouge_f[2]
        # rl_p = rouge_scores[0]['rouge-l']['p']
        # print("score", score)
        return score
