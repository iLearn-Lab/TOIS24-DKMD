from constant import PAD_ID
from util.better_abc import ABCMeta, abstract_attribute


class TextEncoderConfig(metaclass=ABCMeta):
    pad_index = abstract_attribute()
    embed_size = abstract_attribute()
    hidden_size = abstract_attribute()
    num_directions = abstract_attribute()
    num_layers = abstract_attribute()
    dropout = abstract_attribute()
    text_feat_size = abstract_attribute()
    vocab_size = abstract_attribute()
    embed_init = abstract_attribute()

# TextEncoderConfig是父类；ContextTextEncoderConfig是子类
class ContextTextEncoderConfig(TextEncoderConfig):
    pad_index = PAD_ID
    embed_size = 300
    hidden_size = 256
    # 双向
    num_directions = 2
    num_layers = 1
    dropout = 0
    text_feat_size = hidden_size * num_layers * num_directions
    n_hyper = 2


    # def __init__(self, vocab_size, subgraphs_G, reversed_subgraphs_mapping_w, embed_init=None):
    def __init__(self, vocab_size, embed_init=None):
        # super(class,self).init()这是对继承自父类的属性进行初始化，而且是用父类的初始化方法来初始化继承的属性。也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。
        super(ContextTextEncoderConfig, self).__init__()
        self.vocab_size = vocab_size
        self.embed_init = embed_init
        self.train_len = 23369
        self.valid_len = 4325
        self.test_len = 6959

        # 随机生成64个数字--取出对应的subgraphs和reverse
        # self.index_context = torch.randint(1,100,(1, 64))
        # self.subgraphs_G = torch.index_select(subgraphs_G, dim, index, out=None)
        # self.subgraphs_G = subgraphs_G 
        # self.reversed_subgraphs_mapping_w = reversed_subgraphs_mapping_w

class ProductTextEncoderConfig(TextEncoderConfig):
    pad_index = PAD_ID
    embed_size = 300
    hidden_size = 256
    num_directions = 2
    num_layers = 1
    dropout = 0
    text_feat_size = hidden_size * num_layers * num_directions

    def __init__(self, vocab_size, embed_init=None):
        super(ProductTextEncoderConfig, self).__init__()
        self.vocab_size = vocab_size
        self.embed_init = embed_init
