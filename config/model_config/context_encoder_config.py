import operator
from config.model_config import ContextTextEncoderConfig
from config.model_config import ContextImageEncoderConfig


class ContextEncoderConfig:
    # operator.add(x, y)和表达式x+y是等效的
    # text_feat_size:512; image_feat_size:512--->加起来是1024
    input_size = operator.add(ContextTextEncoderConfig.text_feat_size,
                              ContextImageEncoderConfig.image_feat_size)
    hidden_size = 1024
    num_layers = 1
    num_directions = 2
    embed_size = 512
    # output_size = hidden_size * num_layers * num_directions
    output_size = 300