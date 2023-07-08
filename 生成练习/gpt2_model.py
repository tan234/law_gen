



import paddle
from paddlenlp.transformers import GPTModel, GPTTokenizer

class Gpt2Model(paddle.nn.Layer):
    # 使用的是 gpt2-medium-en ,当然也可以使用其他或者不适用预训练的模型

    def __init__(self,vocab_size):
        super().__init__()
        self.encoder = GPTModel.from_pretrained('gpt2-medium-en')
        self.linear = paddle.nn.Linear(1024,vocab_size)
    def forward(self,x):

        x = self.encoder(input_ids=x)
        x = self.linear(x)
        return x