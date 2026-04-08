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
    embed_img_size = 2048
    hidden_size = 256
    # 双向
    num_directions = 2
    num_layers = 1 #1 2
    dropout = 0
    text_feat_size = hidden_size * num_layers * num_directions
    n_hyper = 2
    hidden_size_RNN = 1024
    num_layers_RNN = 1
    num_directions_RNN = 2
    embed_size_RNN = 512

    # def __init__(self, vocab_size, subgraphs_G, reversed_subgraphs_mapping_w, embed_init=None):
    def __init__(self, vocab_size, word_size, embed_init=None):
        # super(class,self).init()这是对继承自父类的属性进行初始化，而且是用父类的初始化方法来初始化继承的属性。也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。
        super(ContextTextEncoderConfig, self).__init__()
        self.vocab_size = vocab_size
        self.word_size = word_size
        # self.img_size = img_size
        self.embed_init = embed_init
        # self.embed_img = embed_img
        self.train_len = 23369
        self.valid_len = 4325
        self.test_len = 6959
        # self.hidden_size_RNN = 1024
        # self.num_layers_RNN = 1
        # self.num_directions_RNN = 2
        # self.embed_size_RNN = 512
        self.hidden_size_RNN = 300 #300 1024
        self.num_layers_RNN = 1
        self.num_directions_RNN = 2
        self.embed_size_RNN = 300
        self.embed_size = 300 #300 #256 1024
        self.hid_router = 512
        self.hid_IMRC = 512
        self.num_head_IMRC = 16
        self.direction = 'i2t'


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
